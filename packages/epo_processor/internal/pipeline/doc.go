// Package pipeline implements a streaming ETL for EPO and HUPD patent
// archives, wired with [github.com/destel/rill].
//
// The five stages are:
//
//	Source -> Opener -> Extractor -> Batch -> Sink
//
// Stage contracts live in types.go; implementations sit alongside
// ([HTTPOpener], [LocalFileOpener], [XMLStreamExtractor], [ParquetSink],
// [BoltCheckpointer], ...). Cancellation and backpressure flow through
// every stage via context.Context and rill's bounded channels.
//
// # Usage
//
//	p, err := pipeline.New(
//	    pipeline.WithSource(src),
//	    pipeline.WithOpener(opener),
//	    pipeline.WithExtractor(pipeline.NewXMLStreamExtractor()),
//	    pipeline.WithSink(sink),
//	)
//	if err != nil {
//	    return err
//	}
//	return p.Run(ctx)
//
// # Concurrency
//
//   - ArchiveConcurrency: parallel archives in flight (default 4).
//   - ExtractorConcurrency: parallel XML decoders (default 4).
//   - Sink writes are serialised (concurrency 1).
//
// # Resumability
//
// Set pipeline.checkpoint_db (cmd config) to enable bbolt-backed
// resumption. Disabled by default via [NoopCheckpointer].
package pipeline
