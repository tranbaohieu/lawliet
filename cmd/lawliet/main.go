// Command lawliet runs the Lawliet in-memory vector cache server.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/tranbaohieu/lawliet/internal/command"
	"github.com/tranbaohieu/lawliet/internal/config"
	"github.com/tranbaohieu/lawliet/internal/server"
	"github.com/tranbaohieu/lawliet/internal/snapshot"
	"github.com/tranbaohieu/lawliet/internal/store"
	"github.com/tranbaohieu/lawliet/internal/version"
)

func main() {
	os.Exit(run(os.Args[1:]))
}

func run(args []string) int {
	cfg, err := config.Parse(args, os.Stderr)
	if errors.Is(err, flag.ErrHelp) {
		return 0
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "lawliet:", err)
		return 2
	}
	if cfg.ShowVersion {
		fmt.Println("lawliet", version.Version)
		return 0
	}

	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if cfg.ExitOnStdinEOF {
		var cancel context.CancelFunc
		ctx, cancel = context.WithCancel(ctx)
		defer cancel()
		go cancelOnEOF(os.Stdin, cancel, logger)
	}

	policy, err := store.ParsePolicy(cfg.MaxMemoryPol)
	if err != nil {
		fmt.Fprintln(os.Stderr, "lawliet: invalid --maxmemory-policy:", err)
		return 2
	}
	st := store.New()
	st.SetLimits(store.Limits{
		MaxMemory:      cfg.MaxMemory,
		MaxCollections: cfg.MaxCollections,
		DefaultPolicy:  policy,
	})

	// Persistence (DESIGN.md §10): load before serving, save after the server stopped.
	var snap *snapshot.Snapshotter
	if path := cfg.SnapshotPath(); path != "" {
		rules, err := saveRules(cfg.SaveRules)
		if err != nil {
			fmt.Fprintln(os.Stderr, "lawliet: invalid --save:", err)
			return 2
		}
		snap = snapshot.New(st, path, rules, logger)
		start := time.Now()
		stats, found, err := snap.Load()
		if err != nil {
			logger.Error("cannot load snapshot; refusing to start (move or delete the file to start empty)",
				"path", path, "err", err)
			return 1
		}
		if found {
			logger.Info("snapshot loaded", "path", path, "collections", stats.Collections,
				"records", stats.Records, "expired_skipped", stats.Skipped,
				"duration", time.Since(start).Round(time.Millisecond))
		}
		go snap.Run(ctx)
	}
	go st.RunExpirer(ctx, 100*time.Millisecond)

	srv := server.New(cfg.Addr(), logger)
	command.Register(srv, st)
	command.RegisterInfo(srv, st)
	command.RegisterPersistence(srv, snap)
	serveErr := srv.ListenAndServe(ctx)

	// All connections are closed now, so no write can follow the final snapshot.
	if snap != nil {
		if _, err := snap.Save(); err != nil {
			return 1
		}
	}
	if serveErr != nil {
		logger.Error("server error", "err", serveErr)
		return 1
	}
	return 0
}

// cancelOnEOF calls cancel once r reaches EOF or fails. With --exit-on-stdin-eof,
// stdin is a pipe held by the parent process: it is closed when the parent exits
// for any reason (including SIGKILL), so the server does not outlive it.
func cancelOnEOF(r io.Reader, cancel context.CancelFunc, logger *slog.Logger) {
	_, err := io.Copy(io.Discard, r)
	logger.Info("stdin closed, shutting down", "err", err)
	cancel()
}

// saveRules returns the snapshot rules from --save values: defaults when none
// were given, no rules for a single empty value.
func saveRules(raw []string) ([]snapshot.Rule, error) {
	if raw == nil {
		return snapshot.DefaultRules, nil
	}
	if len(raw) == 1 && strings.TrimSpace(raw[0]) == "" {
		return nil, nil
	}
	rules := make([]snapshot.Rule, 0, len(raw))
	for _, r := range raw {
		rule, err := snapshot.ParseRule(r)
		if err != nil {
			return nil, err
		}
		rules = append(rules, rule)
	}
	return rules, nil
}
