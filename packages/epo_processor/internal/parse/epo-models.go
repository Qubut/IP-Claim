package parse

// ExchangeDocument is the in-memory mirror of an EPO
// <exchange-document> element. It is the parser's internal staging
// type — the public, flat output is [PatentRecord].
type ExchangeDocument struct {
	Country               string
	DocNumber             string
	Kind                  string
	Status                string
	PatentClassifications []PatentClassification
	Citations             []Citation
	FamilyMembers         []FamilyMember
}

// PatentClassification is one <patent-classification> entry. Scheme is
// the classification system (e.g. "CPCI", "IPCR"); ClassificationSymbol
// is the canonical symbol within that scheme.
type PatentClassification struct {
	Scheme               string // e.g. "CPCI" or "IPCR"
	ClassificationSymbol string // canonical symbol, e.g. "H04L63/00"
}

// Citation is one <citation> under <references-cited>. CitedID is the
// concatenated country+number+kind of the cited document; Categories
// captures the EPO citation categories (X, Y, A, …).
type Citation struct {
	// CitedID is the country+doc-number+kind triple, e.g. "US9114971B2".
	CitedID string `parquet:"name=cited_id, type=BYTE_ARRAY, convertedtype=UTF8"`
	// Categories holds the EPO examiner relevance codes: X (highly relevant),
	// Y (relevant in combination), A (background), E, O, T, …
	Categories []string `parquet:"name=categories, type=LIST"`
}

// FamilyMember is one <family-member> under <patent-family>. Each
// member groups the publication references for a single patent within
// the same simple family.
type FamilyMember struct {
	PublicationReferences []PublicationReference
}

// PublicationReference is one <publication-reference> with its
// data-format attribute (typically "docdb" or "epodoc") and the
// resolved [DocumentID].
type PublicationReference struct {
	// DataFormat is the DOCDB identifier format: "docdb" or "epodoc".
	DataFormat string
	DocumentID DocumentID
}

// DocumentID is the country + doc-number + kind triple identifying a
// single patent publication.
type DocumentID struct {
	Country   string // two-letter country code, e.g. "US", "EP"
	DocNumber string // bare application or publication number
	Kind      string // kind code, e.g. "B2", "A1"
}

// PatentRecord is the flat Parquet row written by the streaming
// pipeline. Field order and parquet tags are stable contracts —
// changing them is a breaking change for downstream consumers.
type PatentRecord struct {
	// PatentID is the EPO identifier: country+doc-number+kind, e.g. "EP1234567A1".
	PatentID string `parquet:"name=patent_id, type=BYTE_ARRAY, convertedtype=UTF8"`
	// Status is the legal status of the document, e.g. "new", "update".
	Status string `parquet:"name=status, type=BYTE_ARRAY, convertedtype=UTF8"`
	// CPCList contains the CPC (Cooperative Patent Classification) symbols
	// assigned by the examining office.
	CPCList []string `parquet:"name=cpc_list, type=LIST"`
	// Citations lists all documents cited in the examination report.
	Citations []Citation `parquet:"name=citations, type=LIST"`
	// FamilyPatents lists the DOCDB patent identifiers of all simple-family
	// members, normalised to country+doc-number+kind.
	FamilyPatents []string `parquet:"name=family_patents, type=LIST"`
}
