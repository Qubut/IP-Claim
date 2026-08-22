package hupd

import (
	"regexp"
	"strings"

	A "github.com/IBM/fp-go/v2/array"
	F "github.com/IBM/fp-go/v2/function"
	P "github.com/IBM/fp-go/v2/predicate"
)

// HUPDDateSuffixRE matches the trailing "-YYYYMMDD" segment HUPD appends to publication_number values.
var HUPDDateSuffixRE = regexp.MustCompile(`-\d{8}$`)

// epoKindRE matches the trailing kind code on a DOCDB-style EPO identifier (e.g. "A1", "B2").
var epoKindRE = regexp.MustCompile(`[A-Z]\d?$`)

// usptoPubRE matches a USPTO pre-grant publication number body: "US" followed
// by an 11-digit number (4-digit year + 7-digit serial), e.g. "US20100138160".
// EPO/DOCDB renders the same publication with a 6-digit serial ("US2010138160"),
// dropping the serial's leading zero. The 11 consecutive digits distinguish a
// USPTO publication from a granted number ("US9114971", ≤8 digits) and from the
// already-EPO-formatted publication ("US2010138160", 10 digits).
var usptoPubRE = regexp.MustCompile(`^US\d{11}`)

// NormalizeHUPDPatentNumber returns the granted-patent-number stem (bare numeric string in HUPD).
func NormalizeHUPDPatentNumber(s string) string {
	return strings.TrimSpace(s)
}

// NormalizeHUPDPublication converts a raw HUPD publication_number into the same
// canonical stem produced by [NormalizeUSID] for an EPO patent identifier.
//
// HUPD stores publication numbers in USPTO format with a trailing application
// date, e.g. "US20100138160A1-20100603". USPTO uses an 11-digit body (4-digit
// year + 7-digit serial) whereas EPO/DOCDB uses 10 (year + 6-digit serial),
// dropping the serial's leading zero. Without this reconciliation a HUPD
// publication ("US20100138160A1") and the matching EPO patent_id
// ("US2010138160A1") normalise to different stems ("20100138160" vs
// "2010138160") and never match — which previously left the EPO↔HUPD self-link
// (DatasetRecord.EPOHUPDPaths) empty for every application-style US patent.
//
// Mirrors the Python reference: publication_number.split('-')[0] then
// s[:6] + s[7:] (drop the extra serial digit at index 6).
func NormalizeHUPDPublication(raw string) string {
	s := HUPDDateSuffixRE.ReplaceAllString(strings.TrimSpace(raw), "")
	if usptoPubRE.MatchString(s) {
		s = s[:6] + s[7:]
	}
	return NormalizeUSID(s)
}

// NormalizeUSID returns the canonical numeric stem of a US-prefixed EPO identifier.
// "US9114971B2" → "9114971", "US20120043352A1" → "20120043352".
// Returns "" for non-US identifiers and unparseable inputs.
func NormalizeUSID(id string) string {
	if !strings.HasPrefix(id, "US") {
		return ""
	}
	s := id[2:]
	if loc := epoKindRE.FindStringIndex(s); loc != nil && loc[1] == len(s) {
		s = s[:loc[0]]
	}
	return s
}

// NormalizeFamily maps fps through NormalizeUSID, discarding empty results.
func NormalizeFamily(fps []string) []string {
	return F.Pipe2(fps, A.Map(NormalizeUSID), A.Filter(P.IsNonZero[string]()))
}
