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

var downloadCmd = &cobra.Command{
	Use:   "download",
	Short: "Download EPO files",
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
		defer cancel()
		res := services.Downloader.FetchFiles(ctx)()
		if ET.IsLeft(res) {
			_, err := ET.UnwrapError(res)
			return fmt.Errorf("download failed: %w", err)
		}
		logger.Info("Download completed")
		return nil
	},
}
