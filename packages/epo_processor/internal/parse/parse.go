package parse

import (
	"fmt"
	"strings"

	"github.com/IBM/fp-go/v2/array"
	F "github.com/IBM/fp-go/v2/function"
	IOR "github.com/IBM/fp-go/v2/idiomatic/ioresult"
	IO "github.com/IBM/fp-go/v2/io"
	"github.com/IBM/fp-go/v2/option"
	O "github.com/IBM/fp-go/v2/ord"
	"github.com/antchfx/xmlquery"
)

// ExtractPatentRecord parses an EPO exchange-document XML node into a
// flat [PatentRecord]. Stateless and goroutine-safe; performs no I/O.
func ExtractPatentRecord(node *xmlquery.Node) (PatentRecord, error) {
	return exchangeDocumentFromNode(node)
}

func exchangeDocumentFromNode(node *xmlquery.Node) (PatentRecord, error) {
	country := node.SelectAttr("country")
	docNumber := node.SelectAttr("doc-number")
	kind := node.SelectAttr("kind")
	status := node.SelectAttr("status")
	if country == "" || docNumber == "" || kind == "" || status == "" {
		return PatentRecord{}, fmt.Errorf("missing required attributes")
	}
	classifications := extractAll(node, ".//*[local-name()='patent-classification']",
		func(n *xmlquery.Node) IOR.IOResult[PatentClassification] {
			schemeNode := xmlquery.FindOne(n, "*[local-name()='classification-scheme']")
			if schemeNode == nil {
				return IOR.Left[PatentClassification](
					fmt.Errorf("missing classification-scheme"),
				)
			}
			scheme := schemeNode.SelectAttr("scheme")
			if scheme == "" {
				return IOR.Left[PatentClassification](fmt.Errorf("missing scheme attribute"))
			}
			symbolNode := xmlquery.FindOne(n, "*[local-name()='classification-symbol']")
			if symbolNode == nil {
				return IOR.Left[PatentClassification](
					fmt.Errorf("missing classification-symbol"),
				)
			}
			symbol := strings.TrimSpace(symbolNode.InnerText())
			return IOR.Of(
				PatentClassification{Scheme: scheme, ClassificationSymbol: symbol},
			)
		})
	citations := extractAll(node, ".//*[local-name()='references-cited']/*[local-name()='citation']",
		func(n *xmlquery.Node) IOR.IOResult[Citation] {
			categories := F.Pipe2(
				xmlquery.Find(
					n,
					"*[local-name()='category'] | *[local-name()='rel-passage']/*[local-name()='category']",
				),
				array.Map(func(c *xmlquery.Node) string {
					return strings.TrimSpace(c.InnerText())
				}),
				array.Filter(func(s string) bool {
					return s != ""
				}),
			)
			citedID := F.Pipe2(
				option.FromNillable(
					xmlquery.FindOne(n, "*[local-name()='patcit']/*[local-name()='document-id']"),
				),
				option.Map(func(docIDNode *xmlquery.Node) string {
					c := getText(docIDNode, "*[local-name()='country']")
					d := getText(docIDNode, "*[local-name()='doc-number']")
					k := getText(docIDNode, "*[local-name()='kind']")
					if c != "" || d != "" || k != "" {
						return c + d + k
					}
					return ""
				}),
				option.GetOrElse(func() string { return "" }),
			)
			return IOR.Of(Citation{CitedID: citedID, Categories: categories})
		})
	familyMembers := extractAll(node, ".//*[local-name()='patent-family']/*[local-name()='family-member']",
		func(familyNode *xmlquery.Node) IOR.IOResult[FamilyMember] {
			listRefs := IOR.IOResult[[]*xmlquery.Node](func() ([]*xmlquery.Node, error) {
				return xmlquery.QueryAll(
					familyNode,
					"*[local-name()='publication-reference']",
				)
			})
			refs := F.Pipe1(
				listRefs,
				IOR.Chain(
					IOR.TraverseArray(
						func(pr *xmlquery.Node) IOR.IOResult[PublicationReference] {
							dataFormat := pr.SelectAttr("data-format")
							if dataFormat == "" {
								return IOR.Left[PublicationReference](
									fmt.Errorf("missing data-format attribute"),
								)
							}
							docIDNode := xmlquery.FindOne(pr, "*[local-name()='document-id']")
							if docIDNode == nil {
								return IOR.Left[PublicationReference](
									fmt.Errorf("no document-id found"),
								)
							}
							c := getText(docIDNode, "*[local-name()='country']")
							d := getText(docIDNode, "*[local-name()='doc-number']")
							k := getText(docIDNode, "*[local-name()='kind']")
							return IOR.Of(PublicationReference{
								DataFormat: dataFormat,
								DocumentID: DocumentID{Country: c, DocNumber: d, Kind: k},
							})
						},
					),
				),
			)
			return IOR.MonadMap(refs, func(refs []PublicationReference) FamilyMember {
				return FamilyMember{PublicationReferences: refs}
			})
		})
	doc := ExchangeDocument{
		Country:               country,
		DocNumber:             docNumber,
		Kind:                  kind,
		Status:                status,
		PatentClassifications: classifications,
		Citations:             citations,
		FamilyMembers:         familyMembers,
	}
	patentID := doc.Country + doc.DocNumber + doc.Kind
	strOrd := O.FromStrictCompare[string]()

	// CPC list: filter to CPCI scheme, project symbols, dedupe, sort.
	cpcList := F.Pipe4(
		doc.PatentClassifications,
		array.Filter(func(pc PatentClassification) bool { return pc.Scheme == "CPCI" }),
		array.Map(func(pc PatentClassification) string { return pc.ClassificationSymbol }),
		array.StrictUniq[string],
		array.Sort(strOrd),
	)

	filteredCitations := array.Filter(func(c Citation) bool { return c.CitedID != "" })(doc.Citations)

	// Family list: flatten members, keep docdb refs, project family ID,
	// drop self/empties, dedupe, sort.
	familyList := F.Pipe6(
		doc.FamilyMembers,
		array.Chain(func(fm FamilyMember) []PublicationReference { return fm.PublicationReferences }),
		array.Filter(func(pr PublicationReference) bool { return pr.DataFormat == "docdb" }),
		array.Map(func(pr PublicationReference) string {
			return pr.DocumentID.Country + pr.DocumentID.DocNumber + pr.DocumentID.Kind
		}),
		array.Filter(func(fid string) bool { return fid != "" && fid != patentID }),
		array.StrictUniq[string],
		array.Sort(strOrd),
	)
	return PatentRecord{
		PatentID:      patentID,
		Status:        doc.Status,
		CPCList:       cpcList,
		Citations:     filteredCitations,
		FamilyPatents: familyList,
	}, nil
}

func getText(parent *xmlquery.Node, selector string) string {
	n := xmlquery.FindOne(parent, selector)
	if n == nil {
		return ""
	}
	return strings.TrimSpace(n.InnerText())
}

// extractAll runs an XPath query against node and applies elem to each
// matched element. Any failure (the query itself, or any element)
// degrades the whole subsection to an empty slice (best-effort parsing).
func extractAll[T any](
	node *xmlquery.Node,
	xpath string,
	elem func(*xmlquery.Node) IOR.IOResult[T],
) []T {
	list := IOR.IOResult[[]*xmlquery.Node](func() ([]*xmlquery.Node, error) {
		return xmlquery.QueryAll(node, xpath)
	})
	return F.Pipe2(
		list,
		IOR.Chain(IOR.TraverseArray(elem)),
		IOR.GetOrElse(func(_ error) IO.IO[[]T] { return IO.Of([]T{}) }),
	)()
}
