package download

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/IBM/fp-go/v2/array"
	ET "github.com/IBM/fp-go/v2/either"
	"github.com/IBM/fp-go/v2/function"
	IOE "github.com/IBM/fp-go/v2/ioeither"
	"github.com/IBM/fp-go/v2/ioeither/file"
	Http "github.com/IBM/fp-go/v2/ioeither/http"
	"github.com/IBM/fp-go/v2/option"
	"github.com/IBM/fp-go/v2/retry"
	"github.com/IBM/fp-go/v2/tuple"
	"github.com/schollz/progressbar/v3"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"

	"github.com/Qubut/IP-Claim/packages/epo_processor/internal/config"
	"github.com/Qubut/IP-Claim/packages/epo_processor/internal/models"
	T "github.com/Qubut/IP-Claim/packages/epo_processor/internal/typing"
)

type Downloader struct {
	Cfg                     config.Config
	progress                *progressbar.ProgressBar
	total                   int
	Logger                  *zap.SugaredLogger
	Tracer                  trace.Tracer
	Meter                   metric.Meter
	downloadSessionDuration metric.Int64Histogram
	downloadFilesTotal      metric.Int64Counter
	downloadFilesSuccess    metric.Int64Counter
	downloadFilesFailed     metric.Int64Counter
	downloadBytesTotal      metric.Int64Counter
	downloadFileDuration    metric.Int64Histogram
}

type DownloadFile struct {
	filename     string
	filePath     string
	expectedSize int64
	checksum     string
	url          string
}

func NewDownloader(
	cfg config.Config,
	tracer trace.Tracer,
	logger *zap.SugaredLogger,
	meter metric.Meter,
) (*Downloader, error) {
	d := &Downloader{
		Cfg:    cfg,
		Tracer: tracer,
		Logger: logger,
		Meter:  meter,
	}

	var err error
	d.downloadSessionDuration, err = d.Meter.Int64Histogram(
		"download.session.duration",
		metric.WithDescription("Duration of bulk download session"),
		metric.WithUnit("ms"),
	)
	if err != nil {
		return nil, err
	}

	d.downloadFilesTotal, err = d.Meter.Int64Counter(
		"download.files.total",
		metric.WithDescription("Total number of files processed"),
	)
	if err != nil {
		return nil, err
	}

	d.downloadFilesSuccess, err = d.Meter.Int64Counter(
		"download.files.success",
		metric.WithDescription("Number of successfully downloaded or skipped files"),
	)
	if err != nil {
		return nil, err
	}

	d.downloadFilesFailed, err = d.Meter.Int64Counter(
		"download.files.failed",
		metric.WithDescription("Number of failed downloads after retries"),
	)
	if err != nil {
		return nil, err
	}

	d.downloadBytesTotal, err = d.Meter.Int64Counter(
		"download.bytes.total",
		metric.WithDescription("Total bytes actually downloaded"),
		metric.WithUnit("By"),
	)
	if err != nil {
		return nil, err
	}

	d.downloadFileDuration, err = d.Meter.Int64Histogram(
		"download.file.duration",
		metric.WithDescription("Duration of individual file download"),
		metric.WithUnit("ms"),
	)
	if err != nil {
		return nil, err
	}

	return d, nil
}

func (downloader *Downloader) FetchFiles(ctx context.Context) IOE.IOEither[error, []int64] {
	ctx, span := downloader.Tracer.Start(ctx, "download.session", trace.WithAttributes(
		attribute.Int("product_id", downloader.Cfg.Server.ProductID),
		attribute.String("base_url", downloader.Cfg.Server.BaseURL),
		attribute.Int("max_concurrent", downloader.Cfg.Server.ConcurrentDownloads),
		attribute.Int("max_retries", downloader.Cfg.Server.MaxRetries),
	))
	defer span.End()
	startTime := time.Now()
	downloader.Logger.Infow("Starting bulk download session",
		"product_id", downloader.Cfg.Server.ProductID,
		"concurrent", downloader.Cfg.Server.ConcurrentDownloads)
	addProgressBar := function.Flow2(
		array.Reduce(
			func(acc tuple.Tuple2[int64, int], item DownloadFile) tuple.Tuple2[int64, int] {
				return tuple.Tuple2[int64, int]{F1: acc.F1 + item.expectedSize, F2: acc.F2 + 1}
			},
			tuple.Tuple2[int64, int]{F1: 0, F2: 0},
		),
		func(total tuple.Tuple2[int64, int]) IOE.IOEither[error, T.Unit] {
			downloader.progress = progressbar.NewOptions64(
				total.F1,
				progressbar.OptionSetWriter(os.Stdout),
				progressbar.OptionSetWidth(60),
				progressbar.OptionSetDescription(
					"[0/"+strconv.Itoa(total.F2)+"] Downloading files...",
				),
				progressbar.OptionShowBytes(true),
				progressbar.OptionShowIts(),
				progressbar.OptionSetElapsedTime(true),
				progressbar.OptionSetPredictTime(true),
				progressbar.OptionThrottle(50*time.Millisecond),
				progressbar.OptionSetRenderBlankState(true),
				progressbar.OptionUseANSICodes(true),
			)
			downloader.total = total.F2
			return IOE.Of[error](T.Unit{})
		},
	)
	timeout := function.Ternary(
		func(t time.Duration) bool { return t > 0 },
		function.Constant1[time.Duration, time.Duration](
			time.Duration(downloader.Cfg.Server.Timeout)*time.Second,
		),
		function.Constant1[time.Duration, time.Duration](30*time.Second),
	)(
		downloader.Cfg.Server.Timeout,
	)
	var completed atomic.Int64
	client := Http.MakeClient(&http.Client{Timeout: timeout})
	url := fmt.Sprintf(
		"%s/products/%d",
		downloader.Cfg.Server.BaseURL,
		downloader.Cfg.Server.ProductID,
	)
	request := Http.MakeGetRequest(url)
	semaphore := make(chan struct{}, downloader.Cfg.Server.ConcurrentDownloads)
	download := func(downloadFile DownloadFile) IOE.IOEither[error, int64] {
		select {
		case <-ctx.Done():
			return IOE.Left[int64](ctx.Err())
		default:
			acquire := IOE.FromIO[error](
				func() DownloadFile { semaphore <- T.Unit{}; return downloadFile },
			)
			use := function.Flow2(
				function.Curry3(downloader.DownloadFile)(ctx)(client),
				IOE.Chain(func(size int64) IOE.IOEither[error, int64] {
					completed.Add(1)
					desc := fmt.Sprintf(
						"[%d/%d completed] Downloading files...",
						completed.Load(),
						downloader.total,
					)
					downloader.progress.Describe(desc)
					return IOE.Of[error](size)
				}),
			)
			release := func(_ DownloadFile, _ ET.Either[error, int64]) IOE.IOEither[error, T.Unit] {
				<-semaphore
				return IOE.Of[error](T.Unit{})
			}
			return IOE.Bracket(acquire, use, release)
		}
	}
	cleanUp := func(_ []int64) IOE.IOEither[error, T.Unit] {
		if downloader.progress != nil {
			downloader.progress.Describe("Download complete")
			err := downloader.progress.Finish()
			if err != nil {
				return IOE.Left[T.Unit](fmt.Errorf("Error upon progress bar finishing, %s", err))
			}

			exit_err := downloader.progress.Exit()
			if exit_err != nil {
				return IOE.Left[T.Unit](fmt.Errorf("Error upon progress bar cleanup, %s", exit_err))
			}
		}
		fmt.Fprintln(os.Stderr)
		return IOE.Of[error](T.Unit{})
	}
	program := function.Pipe6(
		request,
		Http.ReadJSON[models.Product](client),
		IOE.Chain(func(p models.Product) IOE.IOEither[error, []DownloadFile] {
			select {
			case <-ctx.Done():
				return IOE.Left[[]DownloadFile](ctx.Err())
			default:
				items := array.MonadChain(
					p.Deliveries,
					func(delivery models.Delivery) []DownloadFile {
						return array.MonadMap(delivery.Items, func(item models.Item) DownloadFile {
							size := parseFileSize(item.FileSize)
							return DownloadFile{
								filename: item.ItemName,
								filePath: filepath.Join(
									downloader.Cfg.Download.Directory,
									item.ItemName,
								),
								expectedSize: size,
								checksum:     item.FileChecksum,
								url: fmt.Sprintf(
									"%s/products/%d/delivery/%d/item/%d/download",
									downloader.Cfg.Server.BaseURL,
									p.Id,
									delivery.DeliveryID,
									item.ItemId,
								),
							}
						})
					},
				)
				downloader.downloadFilesTotal.Add(ctx, int64(len(items)),
					metric.WithAttributes(
						attribute.Int("product_id", downloader.Cfg.Server.ProductID),
					),
				)
				return IOE.Of[error](items)
			}
		}),
		IOE.Tap(addProgressBar),
		IOE.Chain(IOE.TraverseArrayPar(download)),
		IOE.Tap(cleanUp),
		IOE.Tap(func(sizes []int64) IOE.IOEither[error, T.Unit] {
			durationMs := time.Since(startTime).Milliseconds()
			status := "success"
			if len(sizes) == 0 {
				status = "empty"
			}
			downloader.downloadSessionDuration.Record(ctx, durationMs,
				metric.WithAttributes(
					attribute.Int("product_id", downloader.Cfg.Server.ProductID),
					attribute.String("status", status),
					attribute.Int("concurrent", downloader.Cfg.Server.ConcurrentDownloads),
				),
			)
			return IOE.Of[error](T.Unit{})
		}),
	)
	select {
	case <-ctx.Done():
		downloader.Logger.Warn("Download session cancelled")
		return IOE.Left[[]int64](ctx.Err())
	default:
		return program
	}
}

func parseFileSize(s string) int64 {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0
	}
	re := regexp.MustCompile(`^(\d+)(?:[.,](\d+))?\s*([A-Za-z]*)$`)
	matches := re.FindStringSubmatch(s)
	if matches == nil {
		return 0
	}
	integerPart := matches[1]
	decimalPart := matches[2]
	unit := strings.ToUpper(matches[3])
	parseInt := func(str string) option.Option[int64] {
		if str == "" {
			return option.None[int64]()
		}
		return option.TryCatch(func() (int64, error) {
			return strconv.ParseInt(str, 10, 64)
		})
	}
	multiplier := getUnitMultiplier(unit)
	whole := function.Pipe2(
		integerPart,
		parseInt,
		option.Map(func(whole int64) int64 { return whole * multiplier }),
	)
	add := func(a, b int64) option.Option[int64] { return option.Some(a + b) }
	decimal := function.Pipe2(decimalPart, parseInt, option.Map(func(decimal int64) int64 {
		scale := int64(math.Pow10(len(decimalPart)))
		return (decimal * multiplier) / scale
	}))
	total := option.Sequence2(add)(whole, decimal)
	return option.MonadGetOrElse(total, func() int64 { return 0 })
}

func getUnitMultiplier(unit string) int64 {
	u := strings.ToUpper(strings.TrimSpace(unit))
	switch u {
	case "TB", "TIB", "T":
		return 1 << 40
	case "GB", "GIB", "G":
		return 1 << 30
	case "MB", "MIB", "M":
		return 1 << 20
	case "KB", "KIB", "K":
		return 1 << 10
	case "B", "BYTES", "BYTE", "":
		return 1
	default:
		return 0
	}
}

func (downloader *Downloader) DownloadFile(
	ctx context.Context,
	client Http.Client,
	f DownloadFile,
) IOE.IOEither[error, int64] {
	startTime := time.Now()
	ctx, span := downloader.Tracer.Start(ctx, "download.file", trace.WithAttributes(
		attribute.String("file.name", f.filename),
		attribute.String("file.url", f.url),
		attribute.Int64("file.expected_size_bytes", f.expectedSize),
		attribute.String("file.checksum", f.checksum[:12]+"..."),
	))
	defer span.End()
	select {
	case <-ctx.Done():
		return IOE.Left[int64](ctx.Err())
	default:
	}
	if downloader.Cfg.Download.SkipExists {
		verify := verifyChecksum(f.checksum, f.filePath)
		if ET.IsRight(verify()) {
			span.SetAttributes(attribute.Bool("skipped", true))
			span.AddEvent("file_already_exists_and_valid")
			if downloader.progress != nil {
				_ = downloader.progress.Add64(f.expectedSize)
			}
			downloader.downloadFilesSuccess.Add(ctx, 1,
				metric.WithAttributes(
					attribute.Int("product_id", downloader.Cfg.Server.ProductID),
					attribute.String("method", "skip"),
					attribute.Bool("skipped", true),
				),
			)
			return IOE.Of[error](f.expectedSize)
		}
		span.AddEvent("existing_file_invalid_or_missing")
		_ = os.Remove(f.filePath)
	}
	policy := retry.Monoid.Concat(
		retry.LimitRetries(uint(downloader.Cfg.Server.MaxRetries)),
		retry.ExponentialBackoff(5*time.Millisecond),
	)
	action := func(_ retry.RetryStatus) IOE.IOEither[error, int64] {
		select {
		case <-ctx.Done():
			return IOE.Left[int64](ctx.Err())
		default:
			return IOE.Bracket(
				client.Do(Http.MakeGetRequest(f.url)),
				func(resp *http.Response) IOE.IOEither[error, int64] {
					if resp.StatusCode != http.StatusOK {
						return IOE.Left[int64](fmt.Errorf("bad status: %d", resp.StatusCode))
					}
					return IOE.Bracket(
						file.Create(f.filePath),
						func(f *os.File) IOE.IOEither[error, int64] {
							var writer io.Writer = f
							if downloader.progress != nil {
								writer = io.MultiWriter(f, downloader.progress)
							}
							return IOE.TryCatchError(func() (int64, error) {
								return io.Copy(writer, resp.Body)
							})
						},
						func(f *os.File, _ ET.Either[error, int64]) IOE.IOEither[error, any] {
							return IOE.TryCatchError(func() (any, error) { return nil, f.Close() })
						},
					)
				},
				func(resp *http.Response, _ ET.Either[error, int64]) IOE.IOEither[error, any] {
					return IOE.TryCatchError(func() (any, error) { return nil, resp.Body.Close() })
				},
			)
		}
	}
	result := function.Pipe2(IOE.Retrying(policy, action, ET.Fold(
		func(err error) bool {
			fmt.Println(err)
			return true
		},
		function.Constant1[int64](false),
	),
	), IOE.Tap(func(size int64) IOE.IOEither[error, T.Unit] {
		durationMs := time.Since(startTime).Milliseconds()
		attrs := []attribute.KeyValue{
			attribute.String("file.name", f.filename),
			attribute.Int("product_id", downloader.Cfg.Server.ProductID),
		}
		downloader.downloadFilesSuccess.Add(ctx, 1, metric.WithAttributes(attrs...))
		downloader.downloadBytesTotal.Add(ctx, size, metric.WithAttributes(attrs...))
		downloader.downloadFileDuration.Record(ctx, durationMs, metric.WithAttributes(
			attribute.String("status", "success"),
			attribute.Bool("skipped", false),
		))
		return IOE.Of[error](T.Unit{})
	}), IOE.TapLeft[int64](func(result error) IOE.IOEither[error, T.Unit] {
		durationMs := time.Since(startTime).Milliseconds()
		downloader.downloadFilesFailed.Add(ctx, 1, metric.WithAttributes(
			attribute.String("error", fmt.Sprintf("%v", result)),
		))
		downloader.downloadFileDuration.Record(ctx, durationMs, metric.WithAttributes(
			attribute.String("status", "failed"),
		))
		return IOE.Of[error](T.Unit{})
	}))
	return result
}

func verifyChecksum(expectedChecksum, filePath string) IOE.IOEither[error, string] {
	h := sha1.New()
	acquire := file.Open(filePath)
	use := func(f *os.File) IOE.IOEither[error, string] {
		if _, err := io.Copy(h, f); err != nil {
			return IOE.Left[string](err)
		}
		actual := hex.EncodeToString(h.Sum(nil))
		if actual == expectedChecksum {
			return IOE.Right[error](filePath)
		}
		return IOE.Left[string](
			fmt.Errorf("checksum mismatch: expected %s, got %s", expectedChecksum, actual),
		)
	}
	release := func(f *os.File, _ ET.Either[error, string]) IOE.IOEither[error, any] {
		return IOE.TryCatchError(func() (any, error) {
			return nil, f.Close()
		})
	}
	return IOE.Bracket(acquire, use, release)
}
