package pipeline

import (
	"context"

	"github.com/antchfx/xmlquery"
	"github.com/destel/rill"

	"github.com/Qubut/IP-Claim/packages/epo_processor/internal/parse"
)

// streamParserOptions returns lenient parser options. Strict=false lets
// the decoder pass through HTML-style entities (&alpha;, &copy;) that
// EPO XML uses but XML 1.0 does not define.
func streamParserOptions() xmlquery.ParserOptions {
	return xmlquery.ParserOptions{
		Decoder: &xmlquery.DecoderOptions{Strict: false},
	}
}

// XMLStreamExtractor streams <exchange-document> nodes from an XMLEntry,
// keeping memory bounded to a single element. It owns the entry's
// lifetime and closes it when done.
type XMLStreamExtractor struct {
	// ElementXPath selects the streamed element. Defaults to the EPO contract.
	ElementXPath string
	// ParseConcurrency is the number of goroutines that run the CPU-bound
	// per-document parse in parallel within a single entry. Because the
	// reader feeds nodes sequentially, a small value (just enough to overlap
	// parse with read) is sufficient; large per-entry values only
	// over-subscribe (N entries x ParseConcurrency goroutines). Values <= 0
	// default to defaultParseConcurrency.
	ParseConcurrency int
}

// defaultParseConcurrency is the per-entry parse pool size used when
// ParseConcurrency is unset. The reader is sequential, so a small value is
// enough to overlap parsing with reading without over-subscribing cores.
const defaultParseConcurrency = 2

// NewXMLStreamExtractor returns an extractor pre-configured for EPO documents.
func NewXMLStreamExtractor() *XMLStreamExtractor {
	return &XMLStreamExtractor{
		ElementXPath: "//*[local-name()='exchange-document']",
	}
}

func (x *XMLStreamExtractor) parseConcurrency() int {
	if x.ParseConcurrency > 0 {
		return x.ParseConcurrency
	}
	return defaultParseConcurrency
}

// Stream emits one PatentRecord per matched element. Reading the entry's XML
// is sequential (a single decoder owns the entry and its lifetime), but the
// expensive per-document parse is fanned across parseConcurrency workers so a
// single big entry still uses every core. Per-document parse errors surface as
// stream errors and are downgraded/dropped by the pipeline's Catch stage, so
// one bad document never aborts the entry.
func (x *XMLStreamExtractor) Stream(ctx context.Context, e XMLEntry) rill.Stream[PatentRecord] {
	nodes := rill.Generate(func(send func(*xmlquery.Node), sendErr func(error)) {
		defer func() { _ = e.Close() }()

		sp, err := xmlquery.CreateStreamParserWithOptions(
			e.Reader,
			streamParserOptions(),
			x.ElementXPath,
		)
		if err != nil {
			sendErr(err)
			return
		}
		for {
			if ctx.Err() != nil {
				sendErr(ctx.Err())
				return
			}
			node, err := sp.Read()
			if err != nil {
				if err.Error() != "EOF" {
					sendErr(err)
				}
				return
			}
			send(node)
		}
	})

	return rill.OrderedMap(nodes, x.parseConcurrency(), func(n *xmlquery.Node) (PatentRecord, error) {
		return parse.ExtractPatentRecord(n)
	})
}
