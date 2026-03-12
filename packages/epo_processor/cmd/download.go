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

var downloadEpoCmd = &cobra.Command{
	Use:   "download-epo",
	Short: "Download EPO files",
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
		defer cancel()
		res := services.Downloader.FetchEPOFiles(ctx)()
		if ET.IsLeft(res) {
			_, err := ET.UnwrapError(res)
			return fmt.Errorf("download EPO Files failed: %w", err)
		}
		logger.Info("Download completed")
		return nil
	},
}

var downloadHupdCmd = &cobra.Command{
	Use:   "download-hupd",
	Short: "Download HUPD files",
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
		defer cancel()
		res := services.Downloader.DownloadHupd(ctx)()
		if ET.IsLeft(res) {
			_, err := ET.UnwrapError(res)
			return fmt.Errorf("download HUPD dataset failed: %w", err)
		}
		logger.Info("Download completed")
		return nil
	},
}
