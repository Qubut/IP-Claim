package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/spf13/cobra"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/metric"
	noopmetric "go.opentelemetry.io/otel/metric/noop"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"

	"github.com/Qubut/IP-Claim/packages/epo_processor/internal/config"
	"github.com/Qubut/IP-Claim/packages/epo_processor/internal/telemetry"
)

var (
	cfgFile  string
	cfg      config.Config
	logger   *zap.SugaredLogger
	tracer   trace.Tracer
	meter    metric.Meter
	shutdown func(context.Context) error
	// Version is set at build time:
	//   go build -ldflags "-X github.com/Qubut/IP-Claim/.../cmd.Version=v1.0.0"
	Version = "dev"
)

// RootCmd is the cobra entry-point. PersistentPre/PostRunE owns the
// config and telemetry lifecycle for every subcommand.
var RootCmd = &cobra.Command{
	Use:   "epo-processor",
	Short: "Streaming EPO / HUPD patent processor",
		PersistentPreRunE: func(cmd *cobra.Command, _ []string) error {
		var err error
		cfg, err = config.Load(cfgFile, cmd.Flags())
		if err != nil {
			return fmt.Errorf("load config: %w", err)
		}
				if err := os.MkdirAll(cfg.Log.LogDir, 0o750); err != nil {
			return fmt.Errorf("create log directory: %w", err)
		}
		logFile := filepath.Join(cfg.Log.LogDir,
			fmt.Sprintf("epo-processor[%s].log", time.Now().Format("20060102-150405")))

		// Telemetry is opt-in. When disabled, fall back to a console logger
		// and no-op tracer/meter so callers stay agnostic.
		if !cfg.Telemetry.Enabled {
			zl, zerr := buildConsoleLogger(cfg.Log.LogLevel)
			if zerr != nil {
				return fmt.Errorf("init zap logger: %w", zerr)
			}
			logger = zl.Sugar()
			tracer = otel.GetTracerProvider().Tracer("epo-processor") // no-op default
			meter = noopmetric.NewMeterProvider().Meter("epo-processor")
			_ = logFile
			return nil
		}

		teleCfg := telemetry.Config{
			ServiceName: cfg.Telemetry.ServiceName,
			Exporter:    cfg.Telemetry.Exporter,
			Endpoint:    cfg.Telemetry.Endpoint,
			Protocol:    cfg.Telemetry.Protocol,
			Insecure:    cfg.Telemetry.Insecure,
			Headers:     cfg.Telemetry.Headers,
			LogFile:     logFile,
			LogLevel:    cfg.Log.LogLevel,
		}
		tracer, meter, logger, shutdown, err = telemetry.InitOTEL(teleCfg)
		_ = tracer
		_ = meter
		if err != nil {
			return fmt.Errorf("init telemetry: %w", err)
		}
		return nil
	},
		PersistentPostRunE: func(_ *cobra.Command, _ []string) error {
		if shutdown != nil {
			if err := shutdown(context.Background()); err != nil {
				if logger != nil {
					logger.Errorw("shutdown error", "err", err)
				}
				return err
			}
		}
		return nil
	},
	// No RunE: cobra prints help when no subcommand is given.
}

var versionCmd = &cobra.Command{
	Use:   "version",
	Short: "Print the version",
		Run: func(_ *cobra.Command, _ []string) { fmt.Println(Version) },
}

var configCmd = &cobra.Command{Use: "config", Short: "Config operations"}

var printConfigCmd = &cobra.Command{
	Use:   "print",
	Short: "Print the loaded configuration as JSON",
		RunE: func(_ *cobra.Command, _ []string) error {
		data, err := json.MarshalIndent(cfg, "", "  ")
		if err != nil {
			return fmt.Errorf("marshal config: %w", err)
		}
		fmt.Println(string(data))
		return nil
	},
}

// flagDef binds a viper-backed CLI flag.
type flagDef struct{ name, def, usage string }

var rootFlags = []flagDef{
	{"log.log-level", "info", "Log level (debug/info/warn/error)"},
	{"telemetry.enabled", "true", "Enable OpenTelemetry"},
	{"telemetry.exporter", "otlp", "Telemetry exporter (otlp|stdout|none)"},
	{"telemetry.endpoint", "localhost:4317", "OTLP endpoint host:port"},
	{"telemetry.protocol", "grpc", "OTLP protocol (grpc|http)"},
	{"telemetry.insecure", "true", "Allow insecure OTLP connection"},
	{"telemetry.service-name", "epo-processor", "Service name"},
	{"server.base-url", "", "EPO BDDS base URL (process only)"},
	{"server.timeout", "30s", "Request timeout"},
	{"server.max-retries", "3", "Max HTTP retries"},
	{"server.product-id", "3", "EPO product ID"},
	{"server.verify-sha1", "false", "Verify per-archive SHA-1"},
	{"hupd.url", "", "HUPD .tar URL (process-hupd only)"},
	{"hupd.filename", "", "HUPD output filename"},
	{"pipeline.output-parquet", "./data.parquet", "Parquet output path"},
	{"pipeline.archive-concurrency", "4", "Parallel archives"},
	{"pipeline.extractor-concurrency", "4", "Parallel extractors"},
	{"pipeline.batch-size", "1000", "Rows per Parquet write"},
	{"pipeline.batch-timeout", "2s", "Idle flush timeout"},
	{"pipeline.row-group-size", "50000", "Records per Parquet row group (caps RAM)"},
	{"pipeline.spool-dir", "", "Zip spool dir (defaults to OS temp)"},
	{"pipeline.use-local-dir", "", "Replay archives from this dir, no network"},
	{"pipeline.keep-archive", "false", "Tee HTTP body to archive_dir"},
	{"pipeline.archive-dir", "", "Where to keep raw archives"},
	{"pipeline.keep-extracted", "false", "Tee unwrapped entries to extracted_dir"},
	{"pipeline.extracted-dir", "", "Where to keep unwrapped entries"},
	{"pipeline.checkpoint-db", "", "bbolt path enabling resumable runs (empty disables)"},
	{"pipeline.reset-checkpoint", "false", "Delete checkpoint DB and existing parquet output(s) before running"},
}

func init() {
	RootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "Path to config file (yaml/json/toml)")
	for _, f := range rootFlags {
		RootCmd.PersistentFlags().String(f.name, f.def, f.usage)
	}
	configCmd.AddCommand(printConfigCmd)
	RootCmd.AddCommand(processCmd)
	RootCmd.AddCommand(processHupdCmd)
	RootCmd.AddCommand(analyzeCmd)
	RootCmd.AddCommand(versionCmd)
	RootCmd.AddCommand(configCmd)
}
