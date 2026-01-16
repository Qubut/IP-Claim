package parse

import (
	"bufio"
	"context"
	"encoding/csv"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/IBM/fp-go/v2/array"
	ET "github.com/IBM/fp-go/v2/either"
	F "github.com/IBM/fp-go/v2/function"
	IO "github.com/IBM/fp-go/v2/io"
	IOE "github.com/IBM/fp-go/v2/ioeither"
	"github.com/IBM/fp-go/v2/option"
	"github.com/antchfx/xmlquery"
	"github.com/schollz/progressbar/v3"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"
	"golang.org/x/sync/semaphore"

	"github.com/Qubut/IP-Claim/packages/epo_processor/internal/config"
)

type Parser struct {
	Cfg              config.Config
	Logger           *zap.SugaredLogger
	Tracer           trace.Tracer
	Meter            metric.Meter
	progress         *progressbar.ProgressBar
	processedRecords *atomic.Uint64
	sessionDuration  metric.Int64Histogram
	xmlFilesTotal    metric.Int64Counter
	xmlFilesSuccess  metric.Int64Counter
	xmlFilesFailed   metric.Int64Counter
	recordsTotal     metric.Int64Counter
	bytesTotal       metric.Int64Counter
	fileDuration     metric.Int64Histogram
}

func NewParser(
	cfg config.Config,
	tracer trace.Tracer,
	logger *zap.SugaredLogger,
	meter metric.Meter,
) (*Parser, error) {
	p := &Parser{
		Cfg:              cfg,
		Logger:           logger,
		Tracer:           tracer,
		Meter:            meter,
		processedRecords: &atomic.Uint64{},
	}

	var err error
	p.sessionDuration, err = meter.Int64Histogram(
		"parse.session.duration",
		metric.WithDescription("Duration of the full parsing session"),
		metric.WithUnit("ms"),
	)
	if err != nil {
		return nil, err
	}

	p.xmlFilesTotal, err = meter.Int64Counter(
		"parse.xml_files.total",
		metric.WithDescription("Total number of XML files processed"),
	)
	if err != nil {
		return nil, err
	}

	p.xmlFilesSuccess, err = meter.Int64Counter(
		"parse.xml_files.success",
		metric.WithDescription("Number of successfully parsed XML files"),
	)
	if err != nil {
		return nil, err
	}

	p.xmlFilesFailed, err = meter.Int64Counter(
		"parse.xml_files.failed",
		metric.WithDescription("Number of failed XML file parses"),
	)
	if err != nil {
		return nil, err
	}

	p.recordsTotal, err = meter.Int64Counter(
		"parse.records.total",
		metric.WithDescription("Total number of records written to CSV"),
	)
	if err != nil {
		return nil, err
	}

	p.bytesTotal, err = meter.Int64Counter(
		"parse.bytes.total",
		metric.WithDescription("Total bytes parsed from XML files"),
		metric.WithUnit("By"),
	)
	if err != nil {
		return nil, err
	}

	p.fileDuration, err = meter.Int64Histogram(
		"parse.file.duration",
		metric.WithDescription("Duration of individual XML file parsing"),
		metric.WithUnit("ms"),
	)
	if err != nil {
		return nil, err
	}

	return p, nil
}

func (p *Parser) ParseAllToCSV(
	ctx context.Context,
	downloadDir, outputCSV string,
	maxWorkers int64,
) error {
	ctx, sessionSpan := p.Tracer.Start(ctx, "parse.session", trace.WithAttributes(
		attribute.String("download_dir", downloadDir),
		attribute.String("output_csv", outputCSV),
		attribute.Int64("max_workers", maxWorkers),
	))
	defer sessionSpan.End()

	startTime := time.Now()
	p.Logger.Info(
		"Starting parsing session",
		zap.String("download_dir", downloadDir),
		zap.String("output_csv", outputCSV),
	)

	ctxFind, findSpan := p.Tracer.Start(ctx, "parse.find_xml_files")
	var xmlFiles []string
	err := filepath.WalkDir(downloadDir, func(path string, d fs.DirEntry, err error) error {
		if ctxFind.Err() != nil {
			return ctxFind.Err()
		}
		if err != nil {
			p.Logger.Warn("Error accessing path", zap.String("path", path), zap.Error(err))
			return nil
		}
		if !d.IsDir() && strings.EqualFold(filepath.Ext(path), ".xml") {
			xmlFiles = append(xmlFiles, path)
		}
		return nil
	})
	findSpan.End()
	if err != nil {
		sessionSpan.RecordError(err)
		return fmt.Errorf("failed to walk directory: %w", err)
	}

	p.xmlFilesTotal.Add(ctx, int64(len(xmlFiles)))
	p.Logger.Info("Found XML files", zap.Int("count", len(xmlFiles)))
	sessionSpan.AddEvent(
		"xml_files_found",
		trace.WithAttributes(attribute.Int("count", len(xmlFiles))),
	)

	p.progress = progressbar.NewOptions(len(xmlFiles),
		progressbar.OptionSetWriter(os.Stdout),
		progressbar.OptionSetWidth(60),
		progressbar.OptionSetDescription("[0 processed] Parsing XML files..."),
		progressbar.OptionSpinnerType(14),
		progressbar.OptionSetElapsedTime(true),
		progressbar.OptionSetPredictTime(true),
		progressbar.OptionThrottle(50*time.Millisecond),
		progressbar.OptionSetRenderBlankState(true),
		progressbar.OptionUseANSICodes(true),
	)

	file, err := os.Create(outputCSV)
	if err != nil {
		sessionSpan.RecordError(err)
		return fmt.Errorf("failed to create CSV: %w", err)
	}
	defer file.Close()

	writer := csv.NewWriter(bufio.NewWriter(file))
	defer writer.Flush()

	if err := writer.Write([]string{"patent_id", "status", "cpc_list", "citations", "family_patents"}); err != nil {
		sessionSpan.RecordError(err)
		return fmt.Errorf("failed to write header: %w", err)
	}

	var writeMu sync.Mutex
	safeWrite := func(row []string) error {
		writeMu.Lock()
		defer writeMu.Unlock()
		return writer.Write(row)
	}

	safeFlush := func() error {
		writeMu.Lock()
		defer writeMu.Unlock()
		writer.Flush()
		return writer.Error()
	}

	sem := semaphore.NewWeighted(maxWorkers)
	var wg sync.WaitGroup
	errChan := make(chan error, 1)
	var processedFiles atomic.Int64

	for _, xmlPath := range xmlFiles {
		select {
		case <-ctx.Done():
			p.Logger.Warn("Parsing cancelled")
			return ctx.Err()
		default:
		}
		wg.Add(1)
		if err := sem.Acquire(ctx, 1); err != nil {
			return err
		}

		go func(path string) {
			defer wg.Done()
			defer sem.Release(1)

			ctxFile, fileSpan := p.Tracer.Start(ctx, "parse.xml_file", trace.WithAttributes(
				attribute.String("xml_path", path),
			))
			defer fileSpan.End()

			fileStart := time.Now()
			records := p.processSingleXML(ctxFile, path)()
			if ET.IsLeft(records) {
				_, err := ET.UnwrapError(records)
				fileSpan.RecordError(err)
				p.xmlFilesFailed.Add(
					ctxFile,
					1,
					metric.WithAttributes(attribute.String("status", "failed")),
				)
				select {
				case errChan <- fmt.Errorf("failed to process %s: %w", path, err):
				default:
				}
				p.updateProgress()
				return
			}

			res := F.Pipe3(
				records,
				ET.Chain(ET.TraverseArray(func(record [5]string) ET.Either[error, []string] {
					return ET.FromError(safeWrite)(record[:])
				})),
				ET.Map[error](func(records [][]string) uint64 {
					count := uint64(len(records))
					p.recordsTotal.Add(ctxFile, int64(count))
					p.processedRecords.Add(count)
					fileSpan.AddEvent(
						"records_processed",
						trace.WithAttributes(attribute.Int64("count", int64(count))),
					)
					return count
				}),
				ET.MapLeft[uint64](func(err error) error {
					fileSpan.RecordError(err)
					p.xmlFilesFailed.Add(
						ctxFile,
						1,
						metric.WithAttributes(attribute.String("status", "failed")),
					)
					select {
					case errChan <- err:
					default:
					}
					return err
				}),
			)

			flushErr := safeFlush()
			if flushErr != nil {
				fileSpan.RecordError(flushErr)
				p.xmlFilesFailed.Add(
					ctxFile,
					1,
					metric.WithAttributes(attribute.String("status", "failed")),
				)
				select {
				case errChan <- flushErr:
				default:
				}
				p.updateProgress()
				return
			}

			if ET.IsRight(res) {
				p.xmlFilesSuccess.Add(
					ctxFile,
					1,
					metric.WithAttributes(attribute.String("status", "success")),
				)
			}
			durationMs := time.Since(fileStart).Milliseconds()
			p.fileDuration.Record(
				ctxFile,
				durationMs,
				metric.WithAttributes(attribute.String("status", "success")),
			)
			processedFiles.Add(1)
			p.updateProgress()
			if p.processedRecords.Load()%100 == 0 {
				p.Logger.Info("Processed records", zap.Uint64("total", p.processedRecords.Load()))
			}
		}(xmlPath)
	}

	wg.Wait()
	close(errChan)
	if err, ok := <-errChan; ok {
		sessionSpan.RecordError(err)
		return err
	}

	durationMs := time.Since(startTime).Milliseconds()
	status := "success"
	if len(xmlFiles) == 0 {
		status = "empty"
	}
	p.sessionDuration.Record(
		ctx,
		durationMs,
		metric.WithAttributes(attribute.String("status", status)),
	)
	p.Logger.Info("Parsing completed", zap.Uint64("total_records", p.processedRecords.Load()))
	if p.progress != nil {
		p.progress.Describe("Parsing complete")
		_ = p.progress.Finish()
		p.progress = nil
	}
	return nil
}

func (p *Parser) updateProgress() {
	if p.progress != nil {
		_ = p.progress.Add(1)
	}
}

func (p *Parser) processSingleXML(
	ctx context.Context,
	xmlPath string,
) IOE.IOEither[error, [][5]string] {
	ctx, span := p.Tracer.Start(
		ctx,
		"parse.process_xml",
		trace.WithAttributes(attribute.String("xml_path", xmlPath)),
	)
	defer span.End()
	open := IOE.Eitherize1(os.Open)(xmlPath)
	records := F.Pipe4(
		open,
		IOE.Tap(func(f *os.File) IOE.IOEither[error, int64] {
			select {
			case <-ctx.Done():
				return IOE.Left[int64](ctx.Err())
			default:
			}
			fi, err := f.Stat()
			if err != nil {
				return IOE.Left[int64](err)
			}
			size := fi.Size()
			p.bytesTotal.Add(ctx, size)
			span.AddEvent("file_size", trace.WithAttributes(attribute.Int64("bytes", size)))
			return IOE.Right[error](size)
		}),
		IOE.Chain(func(f *os.File) IOE.IOEither[error, *xmlquery.Node] {
			select {
			case <-ctx.Done():
				return IOE.Left[*xmlquery.Node](ctx.Err())
			default:
			}
			return IOE.TryCatchError(func() (*xmlquery.Node, error) {
				return xmlquery.Parse(f)
			})
		}),
		IOE.Chain(func(doc *xmlquery.Node) IOE.IOEither[error, []*xmlquery.Node] {
			select {
			case <-ctx.Done():
				return IOE.Left[[]*xmlquery.Node](ctx.Err())
			default:
			}
			return IOE.TryCatchError(func() ([]*xmlquery.Node, error) {
				return xmlquery.QueryAll(doc, "//*[local-name()='exchange-document']")
			})
		}),
		IOE.Chain(IOE.TraverseArray(func(node *xmlquery.Node) IOE.IOEither[error, [5]string] {
			select {
			case <-ctx.Done():
				return IOE.Left[[5]string](ctx.Err())
			default:
				res, err := exchangeDocumentFromNode(node)
				if err != nil {
					return IOE.Left[[5]string](err)
				}
				return IOE.Right[error](res)
			}
		})),
	)
	return records
}

func exchangeDocumentFromNode(node *xmlquery.Node) ([5]string, error) {
	country := node.SelectAttr("country")
	docNumber := node.SelectAttr("doc-number")
	kind := node.SelectAttr("kind")
	status := node.SelectAttr("status")
	if country == "" || docNumber == "" || kind == "" || status == "" {
		return [5]string{}, fmt.Errorf("missing required attributes")
	}
	classifications := F.Pipe2(
		IOE.TryCatchError(func() ([]*xmlquery.Node, error) {
			return xmlquery.QueryAll(node, ".//*[local-name()='patent-classification']")
		}),
		IOE.Chain(
			IOE.TraverseArray(func(n *xmlquery.Node) IOE.IOEither[error, PatentClassification] {
				schemeNode := xmlquery.FindOne(n, "*[local-name()='classification-scheme']")
				if schemeNode == nil {
					return IOE.Left[PatentClassification](
						fmt.Errorf("missing classification-scheme"),
					)
				}
				scheme := schemeNode.SelectAttr("scheme")
				if scheme == "" {
					return IOE.Left[PatentClassification](fmt.Errorf("missing scheme attribute"))
				}
				symbolNode := xmlquery.FindOne(n, "*[local-name()='classification-symbol']")
				if symbolNode == nil {
					return IOE.Left[PatentClassification](
						fmt.Errorf("missing classification-symbol"),
					)
				}
				symbol := strings.TrimSpace(symbolNode.InnerText())
				return IOE.Right[error](
					PatentClassification{Scheme: scheme, ClassificationSymbol: symbol},
				)
			}),
		),
		IOE.GetOrElse(func(_ error) IO.IO[[]PatentClassification] {
			return IO.Of([]PatentClassification{})
		}),
	)()
	citations := F.Pipe2(
		IOE.TryCatchError(func() ([]*xmlquery.Node, error) {
			return xmlquery.QueryAll(node, ".//*[local-name()='references-cited']/*[local-name()='citation']")
		}),
		IOE.Chain(IOE.TraverseArray(func(n *xmlquery.Node) IOE.IOEither[error, Citation] {
			categories := F.Pipe2(
            xmlquery.Find(n, "*[local-name()='category'] | *[local-name()='rel-passage']/*[local-name()='category']"),
            array.Map(func(c *xmlquery.Node) string {
                return strings.TrimSpace(c.InnerText())
            }),
            array.Filter(func(s string) bool {
                return s != ""
            }),
        )

			citedID := F.Pipe2(option.FromNillable(xmlquery.FindOne(n, "*[local-name()='patcit']/*[local-name()='document-id']")), option.Map(func(docIDNode *xmlquery.Node) string {
				c := getText(docIDNode, "*[local-name()='country']")
				d := getText(docIDNode, "*[local-name()='doc-number']")
				k := getText(docIDNode, "*[local-name()='kind']")
				if !(c == "" && d == "" && k == "") {
					return c + d + k
				}
				return ""
			}), option.GetOrElse(func() string { return "" }))

			return IOE.Right[error](Citation{CitedID: citedID, Categories: categories})
		})),
		IOE.GetOrElse(func(_ error) IO.IO[[]Citation] {
			return IO.Of([]Citation{})
		}),
	)()
	familyMembers := F.Pipe2(
		IOE.TryCatchError(func() ([]*xmlquery.Node, error) {
			return xmlquery.QueryAll(
				node,
				".//*[local-name()='patent-family']/*[local-name()='family-member']",
			)
		}),
		IOE.Chain(
			IOE.TraverseArray(func(familyNode *xmlquery.Node) IOE.IOEither[error, FamilyMember] {
				refs := F.Pipe1(
					IOE.TryCatchError(func() ([]*xmlquery.Node, error) {
						return xmlquery.QueryAll(
							familyNode,
							"*[local-name()='publication-reference']",
						)
					}),
					IOE.Chain(
						IOE.TraverseArray(
							func(pr *xmlquery.Node) IOE.IOEither[error, PublicationReference] {
								dataFormat := pr.SelectAttr("data-format")
								if dataFormat == "" {
									return IOE.Left[PublicationReference](
										fmt.Errorf("missing data-format attribute"),
									)
								}
								docIDNode := xmlquery.FindOne(pr, "*[local-name()='document-id']")
								if docIDNode == nil {
									return IOE.Left[PublicationReference](
										fmt.Errorf("no document-id found"),
									)
								}
								c := getText(docIDNode, "*[local-name()='country']")
								d := getText(docIDNode, "*[local-name()='doc-number']")
								k := getText(docIDNode, "*[local-name()='kind']")
								return IOE.Right[error](PublicationReference{
									DataFormat: dataFormat,
									DocumentID: DocumentID{Country: c, DocNumber: d, Kind: k},
								})
							},
						),
					),
				)
				return IOE.MonadMap(refs, func(refs []PublicationReference) FamilyMember {
					return FamilyMember{PublicationReferences: refs}
				})
			}),
		),
		IOE.GetOrElse(func(_ error) IO.IO[[]FamilyMember] {
			return IO.Of([]FamilyMember{})
		}),
	)()

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

	cpcSet := make(map[string]struct{})
	for _, pc := range doc.PatentClassifications {
		if pc.Scheme == "CPCI" {
			symbol := pc.ClassificationSymbol
			cpcSet[symbol] = struct{}{}
		}
	}
	var cpcList []string
	for symbol := range cpcSet {
		cpcList = append(cpcList, symbol)
	}
	sort.Strings(cpcList)
	cpcStr := strings.Join(cpcList, ";")

	citationsStr := F.Pipe2(
		doc.Citations,
		array.Chain(func(c Citation) []string {
			if c.CitedID != "" {
				cats := strings.Join(c.Categories, ",")
				return []string{fmt.Sprintf("%s (%s)", c.CitedID, cats)}
			}
			return []string{}
		}),
		func(citations []string) string {
			return strings.Join(citations, ";")
		},
	)
	familySet := make(map[string]struct{})
	for _, fm := range doc.FamilyMembers {
		for _, pr := range fm.PublicationReferences {
			if pr.DataFormat == "docdb" {
				fid := pr.DocumentID.Country + pr.DocumentID.DocNumber + pr.DocumentID.Kind
				if fid != patentID {
					familySet[fid] = struct{}{}
				}
			}
		}
	}
	var familyList []string
	for fid := range familySet {
		familyList = append(familyList, fid)
	}
	sort.Strings(familyList)
	familyStr := strings.Join(familyList, ";")
	return [5]string{patentID, doc.Status, cpcStr, citationsStr, familyStr}, nil
}

func getText(parent *xmlquery.Node, selector string) string {
	n := xmlquery.FindOne(parent, selector)
	if n == nil {
		return ""
	}
	return strings.TrimSpace(n.InnerText())
}
