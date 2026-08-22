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
	R "github.com/IBM/fp-go/v2/result"
	"github.com/antchfx/xmlquery"
	"github.com/antchfx/xpath"
)

// Precompiled XPath expressions. xmlquery's string-based Find/FindOne/QueryAll
// route through a process-global, mutex-guarded LRU cache; precompiling once
// and using QuerySelector(All) bypasses that lock entirely, which is what lets
// ExtractPatentRecord scale across many goroutines.
var (
	xpClassifications = xpath.MustCompile(".//*[local-name()='patent-classification']")
	xpClassScheme     = xpath.MustCompile("*[local-name()='classification-scheme']")
	xpClassSymbol     = xpath.MustCompile("*[local-name()='classification-symbol']")
	xpCitations       = xpath.MustCompile(".//*[local-name()='references-cited']/*[local-name()='citation']")
	xpCategories      = xpath.MustCompile("*[local-name()='category'] | *[local-name()='rel-passage']/*[local-name()='category']")
	xpPatcitDocID     = xpath.MustCompile("*[local-name()='patcit']/*[local-name()='document-id']")
	xpFamilyMembers   = xpath.MustCompile(".//*[local-name()='patent-family']/*[local-name()='family-member']")
	xpPubReference    = xpath.MustCompile("*[local-name()='publication-reference']")
	xpDocumentID      = xpath.MustCompile("*[local-name()='document-id']")
	xpCountry         = xpath.MustCompile("*[local-name()='country']")
	xpDocNumber       = xpath.MustCompile("*[local-name()='doc-number']")
	xpKind            = xpath.MustCompile("*[local-name()='kind']")
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
	// Validate all required attributes; short-circuit on the first missing
	// one with an error naming which attribute it was.
	req := func(name, val string) R.Result[string] {
		if val == "" {
			return R.Left[string](fmt.Errorf("missing required attribute %q", name))
		}
		return R.Of(val)
	}
	if _, err := R.UnwrapError(R.SequenceT4(
		req("country", country),
		req("doc-number", docNumber),
		req("kind", kind),
		req("status", status),
	)); err != nil {
		return PatentRecord{}, err
	}
	classifications := extractAll(node, xpClassifications,
		func(n *xmlquery.Node) IOR.IOResult[PatentClassification] {
			pc := F.Pipe3(
				option.FromNillable(xmlquery.QuerySelector(n, xpClassScheme)),
				option.Map(func(s *xmlquery.Node) string { return s.SelectAttr("scheme") }),
				option.Filter(func(scheme string) bool { return scheme != "" }),
				option.Chain(func(scheme string) option.Option[PatentClassification] {
					return F.Pipe1(
						option.FromNillable(xmlquery.QuerySelector(n, xpClassSymbol)),
						option.Map(func(sym *xmlquery.Node) PatentClassification {
							return PatentClassification{
								Scheme:               scheme,
								ClassificationSymbol: strings.TrimSpace(sym.InnerText()),
							}
						}),
					)
				}),
			)
			return option.Fold(
				func() IOR.IOResult[PatentClassification] {
					return IOR.Left[PatentClassification](fmt.Errorf("incomplete patent-classification"))
				},
				IOR.Of[PatentClassification],
			)(pc)
		})
	citations := extractAll(node, xpCitations,
		func(n *xmlquery.Node) IOR.IOResult[Citation] {
			categories := F.Pipe2(
				xmlquery.QuerySelectorAll(n, xpCategories),
				array.Map(func(c *xmlquery.Node) string {
					return strings.TrimSpace(c.InnerText())
				}),
				array.Filter(func(s string) bool {
					return s != ""
				}),
			)
			citedID := F.Pipe2(
				option.FromNillable(
					xmlquery.QuerySelector(n, xpPatcitDocID),
				),
				option.Map(func(docIDNode *xmlquery.Node) string {
					c := getText(docIDNode, xpCountry)
					d := getText(docIDNode, xpDocNumber)
					k := getText(docIDNode, xpKind)
					if c != "" || d != "" || k != "" {
						return c + d + k
					}
					return ""
				}),
				option.GetOrElse(func() string { return "" }),
			)
			return IOR.Of(Citation{CitedID: citedID, Categories: categories})
		})
	familyMembers := extractAll(node, xpFamilyMembers,
		func(familyNode *xmlquery.Node) IOR.IOResult[FamilyMember] {
			refs := F.Pipe1(
				xmlquery.QuerySelectorAll(familyNode, xpPubReference),
				IOR.TraverseArray(
					func(pr *xmlquery.Node) IOR.IOResult[PublicationReference] {
						docID := F.Pipe1(
							option.FromNillable(xmlquery.QuerySelector(pr, xpDocumentID)),
							option.Map(func(idNode *xmlquery.Node) DocumentID {
								return DocumentID{
									Country:   getText(idNode, xpCountry),
									DocNumber: getText(idNode, xpDocNumber),
									Kind:      getText(idNode, xpKind),
								}
							}),
						)
						pubRef := F.Pipe2(
							pr.SelectAttr("data-format"),
							option.FromPredicate(func(df string) bool { return df != "" }),
							option.Chain(func(df string) option.Option[PublicationReference] {
								return option.Map(func(id DocumentID) PublicationReference {
									return PublicationReference{DataFormat: df, DocumentID: id}
								})(docID)
							}),
						)
						return option.Fold(
							func() IOR.IOResult[PublicationReference] {
								return IOR.Left[PublicationReference](fmt.Errorf("incomplete publication-reference"))
							},
							IOR.Of[PublicationReference],
						)(pubRef)
					},
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

func getText(parent *xmlquery.Node, expr *xpath.Expr) string {
	n := xmlquery.QuerySelector(parent, expr)
	if n == nil {
		return ""
	}
	return strings.TrimSpace(n.InnerText())
}

// extractAll runs a precompiled XPath query against node and applies elem to
// each matched element. Any per-element failure degrades the whole subsection
// to an empty slice (best-effort parsing).
func extractAll[T any](
	node *xmlquery.Node,
	expr *xpath.Expr,
	elem func(*xmlquery.Node) IOR.IOResult[T],
) []T {
	return F.Pipe2(
		xmlquery.QuerySelectorAll(node, expr),
		IOR.TraverseArray(elem),
		IOR.GetOrElse(func(_ error) IO.IO[[]T] { return IO.Of([]T{}) }),
	)()
}
