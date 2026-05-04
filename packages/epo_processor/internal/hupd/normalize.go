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

// NormalizeHUPDPatentNumber returns the granted-patent-number stem (bare numeric string in HUPD).
func NormalizeHUPDPatentNumber(s string) string {
	return strings.TrimSpace(s)
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
