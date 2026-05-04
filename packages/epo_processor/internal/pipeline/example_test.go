package pipeline_test

import (
	"context"
	"fmt"

	"github.com/Qubut/IP-Claim/packages/epo_processor/internal/pipeline"
)

func ExampleIsXML() {
	fmt.Println(pipeline.IsXML("EP123456.xml"))
	fmt.Println(pipeline.IsXML("EP123456.XML")) // case-insensitive
	fmt.Println(pipeline.IsXML("data.tar.gz"))
	// Output:
	// true
	// true
	// false
}

func ExampleNew() {
	// Build a pipeline that discards every record (NoopExtractor + NoopSink).
	// In production, replace these with XMLStreamExtractor and ParquetSink.
	p, err := pipeline.New(
		pipeline.WithSource(&pipeline.StaticListSource{}),
		pipeline.WithOpener(pipeline.NewLocalFileOpener(pipeline.WalkConfig{})),
		pipeline.WithExtractor(&pipeline.NoopExtractor{}),
		pipeline.WithSink(&pipeline.NoopSink{}),
		pipeline.WithArchiveConcurrency(2),
		pipeline.WithBatchSize(500),
	)
	if err != nil {
		panic(err)
	}
	// Run returns when the source is exhausted, a stage errors, or ctx is cancelled.
	_ = p.Run(context.Background())
}
