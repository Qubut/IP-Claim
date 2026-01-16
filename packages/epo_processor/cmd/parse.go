package cmd

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"
)

var parseCmd = &cobra.Command{
	Use:   "parse",
	Short: "Parse extracted files to CSV",
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
		defer cancel()
		err := services.Parser.ParseAllToCSV(
			ctx,
			cfg.Download.Directory,
			cfg.Parse.OutputCSV,
			int64(cfg.Parse.Workers),
		)
		if err != nil {
			return fmt.Errorf("parse failed: %w", err)
		}
		logger.Info("Parse completed")
		return nil
	},
}
