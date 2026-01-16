package logger

import (
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// NewLogger creates a production JSON logger writing only to file with configurable level.
// Levels: "debug", "info", "warn", "error" (case-insensitive).
// If logPath empty → no-op logger.
// Invalid level → error.
func NewLogger(logPath, logLevel string) (*zap.SugaredLogger, error) {
	if logPath == "" {
		return zap.NewNop().Sugar(), nil
	}

	var level zap.AtomicLevel
	if err := level.UnmarshalText([]byte(logLevel)); err != nil {
		// Default to info on invalid
		level = zap.NewAtomicLevelAt(zapcore.InfoLevel)
	}

	config := zap.NewProductionConfig()
	config.Level = level
	config.OutputPaths = []string{logPath}
	config.ErrorOutputPaths = []string{logPath}
	config.EncoderConfig = zap.NewProductionEncoderConfig()
	config.EncoderConfig.TimeKey = "timestamp"
	config.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder

	logger, err := config.Build(zap.AddCallerSkip(1))
	if err != nil {
		return nil, err
	}

	return logger.Sugar(), nil
}
