package cmd

import (
	"os"
	"strings"

	"github.com/mattn/go-isatty"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// buildConsoleLogger returns a coloured human-friendly zap logger when
// stdout is a TTY and a JSON production logger otherwise.
func buildConsoleLogger(level string) (*zap.Logger, error) {
	lvl := zapcore.InfoLevel
	_ = lvl.UnmarshalText([]byte(strings.ToLower(level)))

	if !isatty.IsTerminal(os.Stdout.Fd()) {
		cfg := zap.NewProductionConfig()
		cfg.Level = zap.NewAtomicLevelAt(lvl)
		return cfg.Build()
	}

	encCfg := zap.NewDevelopmentEncoderConfig()
	encCfg.EncodeLevel = zapcore.CapitalColorLevelEncoder
	encCfg.EncodeTime = zapcore.TimeEncoderOfLayout("15:04:05")
	encCfg.EncodeCaller = zapcore.ShortCallerEncoder
	encCfg.ConsoleSeparator = "  "

	cfg := zap.Config{
		Level:             zap.NewAtomicLevelAt(lvl),
		Development:       false,
		Encoding:          "console",
		EncoderConfig:     encCfg,
		OutputPaths:       []string{"stdout"},
		ErrorOutputPaths:  []string{"stderr"},
		DisableStacktrace: true,
	}
	return cfg.Build()
}
