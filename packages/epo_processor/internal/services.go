package internal

import (
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"

	"github.com/Qubut/IP-Claim/packages/epo_processor/internal/config"
	"github.com/Qubut/IP-Claim/packages/epo_processor/internal/download"
	"github.com/Qubut/IP-Claim/packages/epo_processor/internal/extract"
	"github.com/Qubut/IP-Claim/packages/epo_processor/internal/parse"
)

type Services struct {
	Downloader DownloaderInterface
	Extractor  ExtractorInterface
	Parser     ParserInterface
}

func InitServices(
	cfg config.Config,
	tracer trace.Tracer,
	logger *zap.SugaredLogger,
	meter metric.Meter,
) (*Services, error) {
	d, err := download.NewDownloader(cfg, tracer, logger, meter)
	if err != nil {
		return nil, err
	}
	e, err := extract.NewExtractor(cfg, tracer, logger, meter)
	if err != nil {
		return nil, err
	}
	p, err := parse.NewParser(cfg, tracer, logger, meter)
	if err != nil {
		return nil, err
	}
	return &Services{
		Downloader: d,
		Extractor:  e,
		Parser:     p,
	}, nil
}
