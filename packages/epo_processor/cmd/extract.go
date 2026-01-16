package cmd

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	ET "github.com/IBM/fp-go/v2/either"
	"github.com/spf13/cobra"
)

var extractCmd = &cobra.Command{
	Use:   "extract",
	Short: "Extract downloaded files",
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
		defer cancel()
		res := services.Extractor.ExtractAll(ctx, cfg.Download.Directory)()
		if ET.IsLeft(res) {
			_, err := ET.UnwrapError(res)
			return fmt.Errorf("extract failed: %w", err)
		}
		logger.Info("Extract completed")
		return nil
	},
}
