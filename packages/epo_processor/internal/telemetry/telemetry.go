package telemetry

import (
	"context"
	"fmt"
	"strings"

	"go.opentelemetry.io/contrib/bridges/otelzap"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlplog/otlploggrpc"
	"go.opentelemetry.io/otel/exporters/otlp/otlplog/otlploghttp"
	"go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetricgrpc"
	"go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetrichttp"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/exporters/stdout/stdoutlog"
	"go.opentelemetry.io/otel/exporters/stdout/stdoutmetric"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	"go.opentelemetry.io/otel/log/global"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/sdk/log"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"gopkg.in/natefinch/lumberjack.v2"
)

// Config for OTEL setup (unchanged)
type Config struct {
	ServiceName string            // e.g., "epo-processor"
	Exporter    string            // "stdout" or "otlp"
	Endpoint    string            // OTLP endpoint, e.g., "localhost:4317" (required for "otlp")
	Protocol    string            // "grpc" or "http" (default "grpc" for "otlp")
	Insecure    bool              // Disable TLS for OTLP (development only)
	Headers     map[string]string // Custom headers for OTLP, e.g., for auth
	LogFile     string            // Path for JSON logs
	LogLevel    string            // "debug", "info", "warn", "error" (default "info")
}

// InitOTEL sets up providers, tracer, meter, and returns them + bridged logger.
func InitOTEL(
	cfg Config,
) (trace.Tracer, metric.Meter, *zap.SugaredLogger, func(context.Context) error, error) {
	ctx := context.Background()

	// Resource with service name (unchanged)
	res, err := resource.Merge(
		resource.Default(),
		resource.NewSchemaless(
			semconv.ServiceNameKey.String(cfg.ServiceName),
		),
	)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	var traceExp sdktrace.SpanExporter
	var logExp log.Exporter
	switch cfg.Exporter {
	case "stdout":
		traceExp, err = stdouttrace.New(stdouttrace.WithPrettyPrint())
		if err != nil {
			return nil, nil, nil, nil, err
		}
		logExp, err = stdoutlog.New()
		if err != nil {
			return nil, nil, nil, nil, err
		}
	case "otlp":
		if cfg.Endpoint == "" {
			return nil, nil, nil, nil, fmt.Errorf("OTLP endpoint required")
		}
		if cfg.Protocol == "" {
			cfg.Protocol = "grpc"
		}

		var traceClient otlptrace.Client
		switch cfg.Protocol {
		case "grpc":
			opts := []otlptracegrpc.Option{otlptracegrpc.WithEndpoint(cfg.Endpoint)}
			if cfg.Insecure {
				opts = append(opts, otlptracegrpc.WithInsecure())
			}
			if len(cfg.Headers) > 0 {
				opts = append(opts, otlptracegrpc.WithHeaders(cfg.Headers))
			}
			traceClient = otlptracegrpc.NewClient(opts...)
		case "http":
			opts := []otlptracehttp.Option{otlptracehttp.WithEndpoint(cfg.Endpoint)}
			if cfg.Insecure {
				opts = append(opts, otlptracehttp.WithInsecure())
			}
			if len(cfg.Headers) > 0 {
				opts = append(opts, otlptracehttp.WithHeaders(cfg.Headers))
			}
			traceClient = otlptracehttp.NewClient(opts...)
		default:
			return nil, nil, nil, nil, fmt.Errorf("invalid protocol: %s", cfg.Protocol)
		}
		traceExp, err = otlptrace.New(ctx, traceClient)
		if err != nil {
			return nil, nil, nil, nil, err
		}

		switch cfg.Protocol {
		case "grpc":
			opts := []otlploggrpc.Option{otlploggrpc.WithEndpoint(cfg.Endpoint)}
			if cfg.Insecure {
				opts = append(opts, otlploggrpc.WithInsecure())
			}
			if len(cfg.Headers) > 0 {
				opts = append(opts, otlploggrpc.WithHeaders(cfg.Headers))
			}
			logExp, err = otlploggrpc.New(ctx, opts...)
		case "http":
			opts := []otlploghttp.Option{otlploghttp.WithEndpoint(cfg.Endpoint)}
			if cfg.Insecure {
				opts = append(opts, otlploghttp.WithInsecure())
			}
			if len(cfg.Headers) > 0 {
				opts = append(opts, otlploghttp.WithHeaders(cfg.Headers))
			}
			logExp, err = otlploghttp.New(ctx, opts...)
		}
		if err != nil {
			return nil, nil, nil, nil, err
		}
	default:
		return nil, nil, nil, nil, fmt.Errorf("unsupported exporter: %s", cfg.Exporter)
	}

	// Added: Set up metric exporter
	var metricExp sdkmetric.Exporter // Note: sdkmetric.Exporter interface
	switch cfg.Exporter {
	case "stdout":
		metricExp, err = stdoutmetric.New(stdoutmetric.WithPrettyPrint())
		if err != nil {
			return nil, nil, nil, nil, err
		}
	case "otlp":
		// Metric exporter uses direct New() like logs
		switch cfg.Protocol {
		case "grpc":
			opts := []otlpmetricgrpc.Option{otlpmetricgrpc.WithEndpoint(cfg.Endpoint)}
			if cfg.Insecure {
				opts = append(opts, otlpmetricgrpc.WithInsecure())
			}
			if len(cfg.Headers) > 0 {
				opts = append(opts, otlpmetricgrpc.WithHeaders(cfg.Headers))
			}
			metricExp, err = otlpmetricgrpc.New(ctx, opts...)
			if err != nil {
				return nil, nil, nil, nil, err
			}
		case "http":
			opts := []otlpmetrichttp.Option{otlpmetrichttp.WithEndpoint(cfg.Endpoint)}
			if cfg.Insecure {
				opts = append(opts, otlpmetrichttp.WithInsecure())
			}
			if len(cfg.Headers) > 0 {
				opts = append(opts, otlpmetrichttp.WithHeaders(cfg.Headers))
			}
			metricExp, err = otlpmetrichttp.New(ctx, opts...)
			if err != nil {
				return nil, nil, nil, nil, err
			}
		default:
			return nil, nil, nil, nil, fmt.Errorf("invalid protocol: %s", cfg.Protocol)
		}
	}

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(traceExp),
		sdktrace.WithResource(res),
	)
	otel.SetTracerProvider(tp)

	// Added: Metric provider
	mp := sdkmetric.NewMeterProvider(
		sdkmetric.WithResource(res),
		sdkmetric.WithReader(sdkmetric.NewPeriodicReader(metricExp)),
	)
	otel.SetMeterProvider(mp)

	lp := log.NewLoggerProvider(
		log.WithProcessor(log.NewBatchProcessor(logExp)),
		log.WithResource(res),
	)
	global.SetLoggerProvider(lp)

	tracer := otel.Tracer(cfg.ServiceName)
	meter := otel.Meter(cfg.ServiceName) // Added: Get scoped meter

	level := zap.NewAtomicLevelAt(zap.InfoLevel)
	if cfg.LogLevel != "" {
		l := strings.ToLower(cfg.LogLevel)
		if err := level.UnmarshalText([]byte(l)); err != nil {
			// Fallback to info on invalid
			level = zap.NewAtomicLevelAt(zap.InfoLevel)
		}
	}
	var cores []zapcore.Core
	if cfg.LogFile != "" {
		jsonConfig := zap.NewProductionEncoderConfig()
		jsonConfig.TimeKey = "timestamp"
		jsonEncoder := zapcore.NewJSONEncoder(jsonConfig)
		jsonWriter := zapcore.AddSync(
			zapcore.NewMultiWriteSyncer(zapcore.AddSync(&lumberjack.Logger{
				Filename:   cfg.LogFile,
				MaxSize:    100, // MB
				MaxBackups: 5,
			})),
		)
		jsonCore := zapcore.NewCore(jsonEncoder, jsonWriter, level)
		cores = append(cores, jsonCore)
	}

	otelCore := otelzap.NewCore(
		cfg.ServiceName,
		otelzap.WithLoggerProvider(global.GetLoggerProvider()),
		otelzap.WithVersion("1.0.0"),
	)
	cores = append(cores, otelCore)

	zapLogger := zap.New(zapcore.NewTee(cores...))
	logger := zapLogger.Sugar()

	// Shutdown now includes metric provider
	shutdown := func(ctx context.Context) error {
		var shutdownErr error
		if err := tp.Shutdown(ctx); err != nil {
			shutdownErr = err
		}
		if err := lp.Shutdown(ctx); err != nil {
			shutdownErr = err
		}
		if err := mp.Shutdown(ctx); err != nil { // Added
			shutdownErr = err
		}
		_ = zapLogger.Sync()
		return shutdownErr
	}

	return tracer, meter, logger, shutdown, nil
}
