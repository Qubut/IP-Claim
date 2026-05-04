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
}

// NewXMLStreamExtractor returns an extractor pre-configured for EPO documents.
func NewXMLStreamExtractor() *XMLStreamExtractor {
	return &XMLStreamExtractor{
		ElementXPath: "//*[local-name()='exchange-document']",
	}
}

// Stream emits one PatentRecord per matched element. Per-element parse
// errors are reported via sendErr and do not abort the entry.
func (x *XMLStreamExtractor) Stream(ctx context.Context, e XMLEntry) rill.Stream[PatentRecord] {
	return rill.Generate(func(send func(PatentRecord), sendErr func(error)) {
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
			rec, err := parse.ExtractPatentRecord(node)
			if err != nil {
				sendErr(err)
				continue
			}
			send(rec)
		}
	})
}
