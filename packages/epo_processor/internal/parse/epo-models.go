package parse

// ExchangeDocument represents a single patent document
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
	CitedID    string
	Categories []string
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
