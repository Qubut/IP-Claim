package config

import (
	"fmt"
	"strings"
	"time"

	"github.com/go-playground/validator/v10"
	"github.com/spf13/viper"
)

type Config struct {
	Log       Log       `mapstructure:"log"       validate:"required"`
	Telemetry Telemetry `mapstructure:"telemetry" validate:"required"`
	Server    Server    `mapstructure:"server"    validate:"required"`
	Download  Download  `mapstructure:"download"`
	Extract   Extract   `mapstructure:"extract"`
	Parse     Parse     `mapstructure:"parse"`
}

type Log struct {
	LogLevel string `mapstructure:"log_level" validate:"required,oneof=debug info warn error"`
	LogDir   string `mapstructure:"log_dir"   validate:"omitempty,dir"`
}

type Telemetry struct {
	Enabled     bool              `mapstructure:"enabled"`
	Exporter    string            `mapstructure:"exporter"`
	Endpoint    string            `mapstructure:"endpoint"`
	Protocol    string            `mapstructure:"protocol"`
	Insecure    bool              `mapstructure:"insecure"`
	Headers     map[string]string `mapstructure:"headers"`
	ServiceName string            `mapstructure:"service_name"`
}

type Server struct {
	BaseURL             string        `mapstructure:"base_url"             validate:"required,url"`
	Timeout             time.Duration `mapstructure:"timeout"              validate:"required,gt=0"`
	MaxRetries          int           `mapstructure:"max_retries"          validate:"min=0,max=10"`
	ConcurrentDownloads int           `mapstructure:"concurrent_downloads" validate:"min=1,max=30"`
	ProductID           int           `mapstructure:"product_id"           validate:"required"`
}

type Download struct {
	Directory  string `mapstructure:"directory"   validate:"required_if=Enabled true,dir"`
	SkipExists bool   `mapstructure:"skip_exists"`
	VerifySHA1 bool   `mapstructure:"verify_sha1"`
	Enabled    bool   `mapstructure:"enabled"`
	HUPD       HUPD   `mapstructure:"hupd"`
}

type HUPD struct {
	Enabled  bool   `mapstructure:"enabled"`
	URL      string `mapstructure:"url"`
	Filename string `mapstructure:"filename"`
}

type Extract struct {
	Enabled            bool `mapstructure:"enabled"`
	DeleteAfterExtract bool `mapstructure:"delete_after_extract"`
}

type Parse struct {
	Enabled   bool   `mapstructure:"enabled"`
	OutputCSV string `mapstructure:"output_csv"`
	Workers   int    `mapstructure:"workers"`
}

func Load(cfgFile string) (Config, error) {
	v := viper.New()
	v.AutomaticEnv()
	v.SetEnvPrefix("EPO")
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_", "-", "_"))

	// Flexible file loading
	if cfgFile != "" {
		v.SetConfigFile(cfgFile)
	} else {
		v.SetConfigName("config")
		v.AddConfigPath(".")
		v.AddConfigPath("$HOME/.epo-processor")
		v.AddConfigPath("/etc/epo-processor")
		v.SetConfigType("yaml")
	}

	// Defaults
	v.SetDefault("log.log_level", "info")
	v.SetDefault("log.log_dir", "logs")
	v.SetDefault("telemetry.enabled", true)
	v.SetDefault("telemetry.exporter", "otlp")
	v.SetDefault("telemetry.endpoint", "localhost:4317")
	v.SetDefault("telemetry.protocol", "grpc")
	v.SetDefault("telemetry.insecure", true)
	v.SetDefault("telemetry.service_name", "epo-processor")
	v.SetDefault("server.timeout", time.Duration(30)*time.Second)
	v.SetDefault("server.max_retries", 3)
	v.SetDefault("server.concurrent_downloads", 5)
	v.SetDefault("server.product_id", 3)
	v.SetDefault("download.directory", "data")

	err := v.ReadInConfig()
	if err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return Config{}, fmt.Errorf("config read error: %w", err)
		}
		// Not found is ok, use defaults/env
	}

	var cfg Config
	if err := v.UnmarshalExact(&cfg); err != nil {
		return Config{}, fmt.Errorf("unmarshal error: %w", err)
	}

	validate := validator.New()
	if err := validate.Struct(&cfg); err != nil {
		return Config{}, fmt.Errorf("validation failed: %w", err)
	}
	if cfg.Telemetry.Enabled && cfg.Telemetry.Exporter == "otlp" && cfg.Telemetry.Endpoint == "" {
		return Config{}, fmt.Errorf("telemetry.endpoint is required when using otlp exporter")
	}
	return cfg, nil
}
