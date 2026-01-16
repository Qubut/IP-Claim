package internal

import (
	"context"

	"github.com/IBM/fp-go/v2/ioeither"

	T "github.com/Qubut/IP-Claim/packages/epo_processor/internal/typing"
)

type DownloaderInterface interface {
	FetchFiles(ctx context.Context) ioeither.IOEither[error, []int64]
}

type ExtractorInterface interface {
	ExtractAll(ctx context.Context, dir string) ioeither.IOEither[error, T.Unit]
}

type ParserInterface interface {
	ParseAllToCSV(ctx context.Context, downloadDir, outputCSV string, maxWorkers int64) error
}
