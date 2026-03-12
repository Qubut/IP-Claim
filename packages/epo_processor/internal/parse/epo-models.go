package parse

type ExchangeDocument struct {
	Country               string
	DocNumber             string
	Kind                  string
	Status                string
	PatentClassifications []PatentClassification
	Citations             []Citation
	FamilyMembers         []FamilyMember
}

// PatentClassification from the XML
type PatentClassification struct {
	Scheme               string
	ClassificationSymbol string
}

// Citation in references-cited
type Citation struct {
	CitedID    string   `parquet:"name=cited_id, type=BYTE_ARRAY, convertedtype=UTF8"`
	Categories []string `parquet:"name=categories, type=LIST"`
}

// FamilyMember in patent-family
type FamilyMember struct {
	PublicationReferences []PublicationReference
}

// PublicationReference in family-member
type PublicationReference struct {
	DataFormat string
	DocumentID DocumentID
}

// DocumentID inside publication-reference
type DocumentID struct {
	Country   string
	DocNumber string
	Kind      string
}

// PatentRecord is the patent schema for Parquet output
type PatentRecord struct {
	PatentID      string     `parquet:"name=patent_id, type=BYTE_ARRAY, convertedtype=UTF8"`
	Status        string     `parquet:"name=status, type=BYTE_ARRAY, convertedtype=UTF8"`
	CPCList       []string   `parquet:"name=cpc_list, type=LIST"`
	Citations     []Citation `parquet:"name=citations, type=LIST"`
	FamilyPatents []string   `parquet:"name=family_patents, type=LIST"`
}
