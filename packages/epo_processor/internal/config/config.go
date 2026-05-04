package config

import (
	"fmt"
	"strings"
	"time"

	"github.com/go-playground/validator/v10"
	"github.com/spf13/pflag"
	"github.com/spf13/viper"
)

// Config is the root configuration object. Each subsection is a value
// type so it can be passed around freely; viper validates on Load().
type Config struct {
	Log       Log       `mapstructure:"log"       validate:"required"`
	Telemetry Telemetry `mapstructure:"telemetry" validate:"required"`
	Server    Server    `mapstructure:"server"`
	HUPD      HUPD      `mapstructure:"hupd"`
	Analyze   Analyze   `mapstructure:"analyze"`
	Pipeline  Pipeline  `mapstructure:"pipeline"`
}

// Log configures the zap logger (level and output directory).
type Log struct {
	// LogLevel controls the minimum severity emitted.
	// Accepted values (case-insensitive): debug, info, warn, error.
	// Default: info.
	LogLevel string `mapstructure:"log_level" validate:"required,oneof=debug info warn error"`
	// LogDir is the directory where the JSON log file is written.
	// An empty string disables file logging (console logging is unaffected).
	LogDir string `mapstructure:"log_dir"`
}

// Telemetry configures the optional OpenTelemetry exporter.
// Set Enabled=false to run without any OTel dependency.
type Telemetry struct {
	// Enabled gates all telemetry. When false the processor uses no-op
	// tracer and meter providers and skips all exporter setup.
	Enabled bool `mapstructure:"enabled"`
	// Exporter selects the wire format. Currently only "otlp" is wired.
	Exporter string `mapstructure:"exporter"`
	// Endpoint is the OTel collector address (host:port or full URL).
	// Required when Exporter="otlp".
	Endpoint string `mapstructure:"endpoint"`
	// Protocol is "grpc" (default) or "http/protobuf".
	Protocol string `mapstructure:"protocol"`
	// Insecure disables TLS for the exporter connection. Default: true.
	Insecure bool `mapstructure:"insecure"`
	// Headers are extra key/value pairs forwarded with every export call
	// (useful for auth tokens on managed collectors).
	Headers map[string]string `mapstructure:"headers"`
	// ServiceName appears as the OTel resource attribute service.name.
	// Default: "epo-processor".
	ServiceName string `mapstructure:"service_name"`
}

// Server points the EPO product source at the live BDDS API. Only
// `process` reads it; `process-hupd` is unaffected, hence optional.
type Server struct {
	// BaseURL is the root of the EPO BDDS REST API, e.g.
	// "https://ops.epo.org/3.2/rest-services".
	BaseURL string `mapstructure:"base_url" validate:"omitempty,url"`
	// Timeout is the per-request HTTP timeout. Default: 30s.
	Timeout time.Duration `mapstructure:"timeout" validate:"omitempty,gt=0"`
	// MaxRetries is the maximum number of HTTP retries per archive.
	// Range: 0–10. Default: 3.
	MaxRetries int `mapstructure:"max_retries" validate:"min=0,max=10"`
	// ProductID selects the EPO bulk-data product to process.
	// Default: 3 (EP front-text exchange data).
	ProductID int `mapstructure:"product_id"`
	// VerifySHA1 enables SHA-1 checksum verification for every download.
	// Mismatches trigger a retry. Default: false.
	VerifySHA1 bool `mapstructure:"verify_sha1"`
}

// HUPD is the HuggingFace HUPD .tar source consumed by `process-hupd`.
type HUPD struct {
	// URL is the full download URL of the HUPD .tar file.
	URL string `mapstructure:"url"`
	// Filename is used as the archive name in log messages and
	// as the key in the checkpoint database.
	Filename string `mapstructure:"filename"`
}

// Analyze configures the `analyze` subcommand.
type Analyze struct {
	// HUPDMetaURL is the URL of the HUPD metadata Feather file to download
	// when it is not present at the path given by --hupd-meta.
	// Default: the 2022-02-22 snapshot on HuggingFace.
	HUPDMetaURL string `mapstructure:"hupd_meta_url"`
}

// Pipeline configures the streaming pipeline shared by both subcommands.
type Pipeline struct {
	// ArchiveConcurrency is the number of archives fetched and walked
	// in parallel. Default: 4.
	ArchiveConcurrency int `mapstructure:"archive_concurrency"`
	// ExtractorConcurrency is the number of parallel XML decoders per
	// archive batch. Default: 4.
	ExtractorConcurrency int `mapstructure:"extractor_concurrency"`
	// BatchSize is the number of PatentRecords per sink Write call.
	// Default: 1000.
	BatchSize int `mapstructure:"batch_size"`
	// BatchTimeout is the maximum time a partial batch waits before
	// being flushed to the sink. Default: 2s.
	BatchTimeout time.Duration `mapstructure:"batch_timeout"`
	// OutputParquet is the destination Parquet file for `process`.
	// Default: "./data.parquet".
	OutputParquet string `mapstructure:"output_parquet"`
	// RowGroupSize overrides DefaultRowGroupSize for the Parquet writer.
	// 0 uses the library default.
	RowGroupSize int `mapstructure:"row_group_size"`
	// SpoolDir is where zip archives are spooled to disk (zip requires
	// random access). Empty defaults to the OS temp directory.
	SpoolDir string `mapstructure:"spool_dir"`
	// UseLocalDir, when non-empty, replays archives from this local
	// directory instead of fetching them over HTTP. Useful for re-runs
	// after a prior run with KeepArchive=true.
	UseLocalDir string `mapstructure:"use_local_dir"`

	// KeepArchive tees each raw HTTP body to a file under ArchiveDir
	// while the walker reads it. Orthogonal to KeepExtracted.
	// Default: false.
	KeepArchive bool `mapstructure:"keep_archive"`
	// ArchiveDir is the destination directory when KeepArchive is true.
	ArchiveDir string `mapstructure:"archive_dir"`
	// KeepExtracted tees each selected archive entry to a file under
	// ExtractedDir while the consumer reads it. Default: false.
	KeepExtracted bool `mapstructure:"keep_extracted"`
	// ExtractedDir is the destination directory when KeepExtracted is true.
	ExtractedDir string `mapstructure:"extracted_dir"`

	// CheckpointDB, when non-empty, enables resumable processing via a
	// bbolt database at this path. Completed archives are recorded so
	// subsequent runs skip them automatically.
	CheckpointDB string `mapstructure:"checkpoint_db"`
	// ResetCheckpoint, when true, deletes CheckpointDB and any existing
	// OutputParquet shards at startup so the next run starts from scratch.
	ResetCheckpoint bool `mapstructure:"reset_checkpoint"`
}

// Load reads config from file/env/flags/defaults and validates it.
// flags is optional; when non-nil every flag is bound to the local viper
// using its name with dashes replaced by underscores within each section
// (e.g. "pipeline.reset-checkpoint" -> "pipeline.reset_checkpoint").
func Load(cfgFile string, flags *pflag.FlagSet) (Config, error) {
	v := viper.New()
	v.AutomaticEnv()
	v.SetEnvPrefix("EPO")
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_", "-", "_"))

	if cfgFile != "" {
		v.SetConfigFile(cfgFile)
	} else {
		v.SetConfigName("config")
		v.AddConfigPath(".")
		v.AddConfigPath("$HOME/.epo-processor")
		v.AddConfigPath("/etc/epo-processor")
		v.SetConfigType("yaml")
	}

	applyDefaults(v)

	if flags != nil {
		flags.VisitAll(func(f *pflag.Flag) {
			// Only bind namespaced flags (e.g. "pipeline.reset-checkpoint");
			// top-level cobra flags like --config / --help are not config keys.
			if !strings.Contains(f.Name, ".") {
				return
			}
			key := strings.ReplaceAll(f.Name, "-", "_")
			_ = v.BindPFlag(key, f)
		})
	}

	if err := v.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return Config{}, fmt.Errorf("config read error: %w", err)
		}
	}

	var cfg Config
	if err := v.UnmarshalExact(&cfg); err != nil {
		return Config{}, fmt.Errorf("unmarshal error: %w", err)
	}
	if err := validator.New().Struct(&cfg); err != nil {
		return Config{}, fmt.Errorf("validation failed: %w", err)
	}
	if cfg.Telemetry.Enabled && cfg.Telemetry.Exporter == "otlp" && cfg.Telemetry.Endpoint == "" {
		return Config{}, fmt.Errorf("telemetry.endpoint is required when using otlp exporter")
	}
	return cfg, nil
}

// applyDefaults registers default values used when neither file nor env
// supplies them.
func applyDefaults(v *viper.Viper) {
	for k, val := range map[string]any{
		"log.log_level":                  "info",
		"log.log_dir":                    "logs",
		"telemetry.enabled":              true,
		"telemetry.exporter":             "otlp",
		"telemetry.endpoint":             "localhost:4317",
		"telemetry.protocol":             "grpc",
		"telemetry.insecure":             true,
		"telemetry.service_name":         "epo-processor",
		"server.timeout":                 30 * time.Second,
		"server.max_retries":             3,
		"server.product_id":              3,
		"pipeline.archive_concurrency":   4,
		"pipeline.extractor_concurrency": 4,
		"pipeline.batch_size":            1000,
		"pipeline.batch_timeout":         2 * time.Second,
		"pipeline.output_parquet":        "./data.parquet",
		"pipeline.keep_archive":          false,
		"pipeline.keep_extracted":        false,
		"analyze.hupd_meta_url":          "https://huggingface.co/datasets/HUPD/hupd/resolve/main/hupd_metadata_2022-02-22.feather",
	} {
		v.SetDefault(k, val)
	}
}
