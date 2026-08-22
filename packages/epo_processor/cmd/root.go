package cmd

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"maps"
	"os"
	"time"

	"github.com/IBM/fp-go/v2/logging"
	"github.com/spf13/cobra"

	"github.com/Qubut/IP-Claim/packages/epo_processor/internal/config"
	applog "github.com/Qubut/IP-Claim/packages/epo_processor/internal/logger"
)

var (
	cfgFile string
	cfg     config.Config
	logger  *slog.Logger
	logCtl  *applog.Controls
	logDone func() error
	// Version is set at build time:
	//   go build -ldflags "-X github.com/Qubut/IP-Claim/.../cmd.Version=v1.0.0"
	Version = "dev"
)

// flagBinds records, per command.
var flagBinds = map[*cobra.Command]map[string]string{}

// bindKey records that cmd's flag name feeds viper key.
func bindKey(cmd *cobra.Command, name, key string) {
	m := flagBinds[cmd]
	if m == nil {
		m = map[string]string{}
		flagBinds[cmd] = m
	}
	m[name] = key
}

// effectiveBinds merges the root (persistent) binds with the running
// command's own binds; the command flags win on collision. It resolves the
// root via cmd.Root() (not the RootCmd package var) to avoid an init cycle.
func effectiveBinds(cmd *cobra.Command) map[string]string {
	out := map[string]string{}
	maps.Copy(out, flagBinds[cmd.Root()])
	maps.Copy(out, flagBinds[cmd])
	return out
}

// Typed flag+bind helpers: register a flag on cmd and record its viper key in
// a single call so the two never drift apart.

func flagStr(cmd *cobra.Command, key, name, short, def, usage string) {
	cmd.Flags().StringP(name, short, def, usage)
	bindKey(cmd, name, key)
}

func flagInt(cmd *cobra.Command, key, name, short string, def int, usage string) {
	cmd.Flags().IntP(name, short, def, usage)
	bindKey(cmd, name, key)
}

func flagBool(cmd *cobra.Command, key, name, short string, def bool, usage string) {
	cmd.Flags().BoolP(name, short, def, usage)
	bindKey(cmd, name, key)
}

func flagDur(cmd *cobra.Command, key, name, short string, def time.Duration, usage string) {
	cmd.Flags().DurationP(name, short, def, usage)
	bindKey(cmd, name, key)
}

var RootCmd = &cobra.Command{
	Use:   "epo-processor",
	Short: "Streaming EPO / HUPD patent processor",
	PersistentPreRunE: func(cmd *cobra.Command, _ []string) error {
		var err error
		cfg, err = config.Load(cfgFile, cmd.Flags(), effectiveBinds(cmd))
		if err != nil {
			return fmt.Errorf("load config: %w", err)
		}

		if cfg.Log.LogDir != "" {
			if err := os.MkdirAll(cfg.Log.LogDir, 0o750); err != nil {
				return fmt.Errorf("create log directory: %w", err)
			}
		}

		logger, logCtl, logDone, err = applog.New(cfg.Log.LogLevel, cfg.Log.LogDir)
		if err != nil {
			return fmt.Errorf("init logger: %w", err)
		}
		// Thread the logger through every logging path:
		//   - slog.SetDefault makes it the stdlib slog default AND redirects
		//     the stdlib log package's output through its handler, so fp-go's
		//     value-tap combinators (which log via the stdlib log package)
		//     land in our console+file handlers.
		//   - logging.SetLogger points fp-go's own slog-based helpers
		//     (GetLogger / context fallback) at the same logger.
		slog.SetDefault(logger)
		logging.SetLogger(logger)
		return nil
	},
	PersistentPostRunE: func(_ *cobra.Command, _ []string) error {
		if logDone != nil {
			return logDone()
		}
		return nil
	},
	// No RunE: cobra prints help when no subcommand is given.
}

var versionCmd = &cobra.Command{
	Use:   "version",
	Short: "Print the version",
	Run:   func(_ *cobra.Command, _ []string) { fmt.Println(Version) },
}

var configCmd = &cobra.Command{Use: "config", Short: "Config operations"}

var printConfigCmd = &cobra.Command{
	Use:   "print",
	Short: "Print the loaded configuration as JSON",
	RunE: func(_ *cobra.Command, _ []string) error {
		data, err := json.MarshalIndent(cfg, "", "  ")
		if err != nil {
			return fmt.Errorf("marshal config: %w", err)
		}
		fmt.Println(string(data))
		return nil
	},
}

func init() {
	RootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "Path to config file (yaml/json/toml)")
	RootCmd.PersistentFlags().StringP("log-level", "l", "info", "Log level (debug/info/warn/error)")
	RootCmd.PersistentFlags().String("log-dir", "logs", "Directory for log + errors files (empty disables file logging)")
	bindKey(RootCmd, "log-level", "log.log_level")
	bindKey(RootCmd, "log-dir", "log.log_dir")

	configCmd.AddCommand(printConfigCmd)
	RootCmd.AddCommand(processCmd)
	RootCmd.AddCommand(processHupdCmd)
	RootCmd.AddCommand(analyzeCmd)
	RootCmd.AddCommand(versionCmd)
	RootCmd.AddCommand(configCmd)
}
