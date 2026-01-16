package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	ET "github.com/IBM/fp-go/v2/either"
	"github.com/IBM/fp-go/v2/function"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"

	"github.com/Qubut/IP-Claim/packages/epo_processor/internal"
	"github.com/Qubut/IP-Claim/packages/epo_processor/internal/config"
	"github.com/Qubut/IP-Claim/packages/epo_processor/internal/telemetry"
	T "github.com/Qubut/IP-Claim/packages/epo_processor/internal/typing"
)

var (
	cfgFile  string
	cfg      config.Config
	logger   *zap.SugaredLogger
	tracer   trace.Tracer
	meter    metric.Meter
	shutdown func(context.Context) error
	services *internal.Services
	Version  = "dev" // Set at build time: go build -ldflags "-X github.com/Qubut/IP-Claim/packages/epo_processor/cmd.Version=v1.0.0"
)

var RootCmd = &cobra.Command{
	Use:   "epo-processor",
	Short: "EPO Patent Processor CLI",
	PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
		var err error
		cfg, err = config.Load(cfgFile)
		if err != nil {
			return fmt.Errorf("load config: %w", err)
		}
		logDir := cfg.Log.LogDir
		if err := os.MkdirAll(logDir, 0o755); err != nil {
			return fmt.Errorf("create log directory: %w", err)
		}

		logFile := filepath.Join(logDir,
			fmt.Sprintf("epo-processor[%s].log", time.Now().Format("20060102-150405")))

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
		if err != nil {
			return fmt.Errorf("init telemetry: %w", err)
		}
		services, err = internal.InitServices(cfg, tracer, logger, meter)
		if err != nil {
			return fmt.Errorf("init services: %w", err)
		}
		return nil
	},
	PersistentPostRunE: func(cmd *cobra.Command, args []string) error {
		if shutdown != nil {
			if err := shutdown(context.Background()); err != nil {
				logger.Errorw("shutdown error", "err", err)
				return err
			}
		}
		return nil
	},
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
		defer cancel()

		if cfg.Download.Enabled {
			res := services.Downloader.FetchFiles(ctx)()
			err := function.Pipe1(
				res,
				ET.Fold(
					func(e error) error { return fmt.Errorf("download: %w", e) },
					func(_ []int64) error { return nil },
				),
			)
			if err != nil {
				return err
			}
		}
		if cfg.Extract.Enabled {
			res := services.Extractor.ExtractAll(ctx, cfg.Download.Directory)()
			err := function.Pipe1(
				res,
				ET.Fold(
					func(e error) error { return fmt.Errorf("extract: %w", e) },
					func(_ T.Unit) error { return nil },
				),
			)
			if err != nil {
				return err
			}
		}
		if cfg.Parse.Enabled {
			if err := services.Parser.ParseAllToCSV(ctx, cfg.Download.Directory, cfg.Parse.OutputCSV, int64(cfg.Parse.Workers)); err != nil {
				return fmt.Errorf("parse: %w", err)
			}
		}
		logger.Info("All steps completed")
		return nil
	},
}

var versionCmd = &cobra.Command{
	Use:   "version",
	Short: "Print the version of epo-processor",
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println(Version)
	},
}

var configCmd = &cobra.Command{
	Use:   "config",
	Short: "Config operations",
}

var printConfigCmd = &cobra.Command{
	Use:   "print",
	Short: "Print the current loaded configuration",
	RunE: func(cmd *cobra.Command, args []string) error {
		data, err := json.MarshalIndent(cfg, "", "  ")
		if err != nil {
			return fmt.Errorf("marshal config: %w", err)
		}
		fmt.Println(string(data))
		return nil
	},
}

func init() {
	RootCmd.PersistentFlags().
		StringVar(&cfgFile, "config", "", "Path to config file (yaml/json/toml)")

	// Flag map to avoid repetition
	type flagDef struct {
		name, def, usage string
	}
	flags := []flagDef{
		{"log-level", "info", "Log level (debug/info/warn/error)"},
		{"telemetry.enabled", "true", "Enable OpenTelemetry"},
		{"telemetry.exporter", "otlp", "Telemetry exporter (otlp|stdout|none)"},
		{"telemetry.endpoint", "localhost:4317", "OTLP endpoint (host:port)"},
		{"telemetry.protocol", "grpc", "OTLP protocol (grpc|http)"},
		{"telemetry.insecure", "true", "Allow insecure OTLP connection"},
		{"telemetry.service-name", "epo-processor", "Service name for telemetry"},
		{"server.base-url", "", "Server base URL"},
		{"server.timeout", "30s", "Request timeout (duration)"},
		{"server.max-retries", "3", "Max retries"},
		{"server.concurrent-downloads", "5", "Concurrent downloads"},
		{"server.product-id", "0", "Product ID"},
		{"download.directory", "./downloads", "Download directory"},
		{"download.skip-exists", "true", "Skip existing files"},
		{"download.verify-sha1", "false", "Verify SHA1"},
		{"download.enabled", "true", "Enable download"},
		{"download.hupd.url", "", "HUPD URL"},
		{"download.hupd.filename", "", "HUPD filename"},
		{"extract.enabled", "true", "Enable extract"},
		{"extract.delete-after-extract", "false", "Delete after extract"},
		{"parse.enabled", "true", "Enable parse"},
		{"parse.output-csv", "./output.csv", "Output CSV path"},
		{"parse.workers", "10", "Parse workers"},
	}
	for _, f := range flags {
		RootCmd.PersistentFlags().String(f.name, f.def, f.usage)
		viper.BindPFlag(
			strings.ReplaceAll(f.name, "-", "_"),
			RootCmd.PersistentFlags().Lookup(f.name),
		)
	}

	configCmd.AddCommand(printConfigCmd)

	RootCmd.AddCommand(downloadCmd)
	RootCmd.AddCommand(extractCmd)
	RootCmd.AddCommand(parseCmd)
	RootCmd.AddCommand(versionCmd)
	RootCmd.AddCommand(configCmd)
}
