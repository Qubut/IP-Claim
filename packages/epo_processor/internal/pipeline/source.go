package pipeline

import (
	"context"
	"fmt"
	"net/http"

	"github.com/IBM/fp-go/v2/array"
	F "github.com/IBM/fp-go/v2/function"
	IOR "github.com/IBM/fp-go/v2/idiomatic/ioresult"
	IORH "github.com/IBM/fp-go/v2/idiomatic/ioresult/http"
	"github.com/destel/rill"

	"github.com/Qubut/IP-Claim/packages/epo_processor/internal/models"
)

// EPOProductSource lists items from an EPO product manifest and emits
// one [ArchiveJob] per item. Network failure aborts the whole stream.
type EPOProductSource struct {
	BaseURL   string
	ProductID int
	Client    *http.Client
}

// NewEPOProductSource constructs a source. A nil client uses
// [http.DefaultClient].
func NewEPOProductSource(baseURL string, productID int, client *http.Client) *EPOProductSource {
	if client == nil {
		client = http.DefaultClient
	}
	return &EPOProductSource{BaseURL: baseURL, ProductID: productID, Client: client}
}

// Stream fetches the manifest, then emits one ArchiveJob per item until
// ctx is cancelled.
func (s *EPOProductSource) Stream(ctx context.Context) rill.Stream[ArchiveJob] {
	return rill.Generate(func(send func(ArchiveJob), sendErr func(error)) {
		jobs, err := s.list(ctx)
		if err != nil {
			sendErr(err)
			return
		}
		for _, j := range jobs {
			if ctx.Err() != nil {
				return
			}
			send(j)
		}
	})
}

// list fetches the product catalogue and projects it to [ArchiveJob]s.
func (s *EPOProductSource) list(ctx context.Context) ([]ArchiveJob, error) {
	url := fmt.Sprintf("%s/products/%d", s.BaseURL, s.ProductID)
	client := IORH.MakeClient(s.Client)
	requester := F.Pipe1(
		IORH.MakeGetRequest(url),
		IOR.Map(func(req *http.Request) *http.Request {
			return req.WithContext(ctx)
		}),
	)
	return F.Pipe2(
		requester,
		IORH.ReadJSON[models.Product](client),
		IOR.Map(s.productToJobs),
	)()
}

// productToJobs flattens Product → Delivery → Item → [ArchiveJob].
func (s *EPOProductSource) productToJobs(p models.Product) []ArchiveJob {
	return F.Pipe1(
		p.Deliveries,
		array.Chain(func(d models.Delivery) []ArchiveJob {
			return array.Map(func(it models.Item) ArchiveJob {
				return ArchiveJob{
					Name: it.ItemName,
					URL: fmt.Sprintf("%s/products/%d/delivery/%d/item/%d/download",
						s.BaseURL, p.ID, d.DeliveryID, it.ItemID),
					Checksum: it.FileChecksum,
				}
			})(d.Items)
		}),
	)
}

// StaticListSource emits a fixed slice of [ArchiveJob]s. Used by tests
// and CLI overrides.
type StaticListSource struct{ Jobs []ArchiveJob }

// Stream emits all configured jobs on a channel and then closes it.
func (s *StaticListSource) Stream(_ context.Context) rill.Stream[ArchiveJob] {
	return rill.FromSlice(s.Jobs, nil)
}
