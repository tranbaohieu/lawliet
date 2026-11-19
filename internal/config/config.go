// Package config parses server command-line flags.
package config

import (
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Config holds server settings. More fields are added per milestone (see DESIGN.md §12).
type Config struct {
	Bind           string
	Port           int
	MaxMemory      int64  // bytes, 0 = unlimited
	MaxMemoryRaw   string // as given, e.g. "2gb"
	MaxMemoryPol   string // default eviction policy for new collections
	MaxCollections int    // 0 = unlimited
	Dir            string // snapshot directory; empty = persistence off
	DBFilename     string
	SaveRules      []string // raw --save values; nil = not given (use defaults)
	ShowVersion    bool
	ExitOnStdinEOF bool // stop when stdin is closed (parent process gone)
}

// SnapshotPath returns the snapshot file path, or "" if persistence is off.
func (c Config) SnapshotPath() string {
	if c.Dir == "" {
		return ""
	}
	return filepath.Join(c.Dir, c.DBFilename)
}

// stringList is a repeatable string flag.
type stringList struct{ vals *[]string }

func (s stringList) String() string {
	if s.vals == nil {
		return ""
	}
	return strings.Join(*s.vals, ", ")
}

func (s stringList) Set(v string) error {
	*s.vals = append(*s.vals, v)
	return nil
}

// Addr returns the listen address "bind:port".
func (c Config) Addr() string { return net.JoinHostPort(c.Bind, strconv.Itoa(c.Port)) }

// Parse parses args (without the program name). Returns flag.ErrHelp for -h/--help.
func Parse(args []string, out io.Writer) (Config, error) {
	var c Config
	fs := flag.NewFlagSet("lawliet", flag.ContinueOnError)
	fs.SetOutput(out)
	fs.StringVar(&c.Bind, "bind", "127.0.0.1", "interface to listen on")
	fs.IntVar(&c.Port, "port", 6380, "TCP port")
	fs.StringVar(&c.MaxMemoryRaw, "maxmemory", "0", "memory limit for records, e.g. 512mb, 2gb (0 = unlimited)")
	fs.StringVar(&c.MaxMemoryPol, "maxmemory-policy", "noeviction",
		"default eviction policy for new collections: noeviction, allkeys-lru, volatile-ttl")
	fs.IntVar(&c.MaxCollections, "max-collections", 0, "maximum number of collections (0 = unlimited)")
	fs.StringVar(&c.Dir, "dir", "", "snapshot directory (empty = in-memory only, no persistence)")
	fs.StringVar(&c.DBFilename, "dbfilename", "dump.lwdb", "snapshot file name inside --dir")
	fs.Var(stringList{&c.SaveRules}, "save",
		`snapshot rule "<seconds> <changes>", repeatable; "" disables automatic snapshots (default "3600 1", "300 100", "60 10000")`)
	fs.BoolVar(&c.ShowVersion, "version", false, "print version and exit")
	fs.BoolVar(&c.ExitOnStdinEOF, "exit-on-stdin-eof", false,
		"shut down gracefully when stdin is closed; a parent process that keeps a pipe to stdin\n"+
			"(the Python package) then stops the server by exiting, even when it is killed")

	if err := fs.Parse(args); err != nil {
		return Config{}, err
	}
	if fs.NArg() > 0 {
		return Config{}, fmt.Errorf("unexpected argument: %q", fs.Arg(0))
	}
	if c.Port < 0 || c.Port > 65535 {
		return Config{}, fmt.Errorf("invalid --port %d: must be 0-65535", c.Port)
	}
	n, err := ParseSize(c.MaxMemoryRaw)
	if err != nil {
		return Config{}, fmt.Errorf("invalid --maxmemory: %w", err)
	}
	c.MaxMemory = n
	if c.MaxCollections < 0 {
		return Config{}, fmt.Errorf("invalid --max-collections %d: must be >= 0", c.MaxCollections)
	}
	if c.Dir != "" {
		fi, err := os.Stat(c.Dir)
		if err != nil || !fi.IsDir() {
			return Config{}, fmt.Errorf("invalid --dir %q: not an existing directory", c.Dir)
		}
		if c.DBFilename == "" || strings.ContainsRune(c.DBFilename, filepath.Separator) {
			return Config{}, fmt.Errorf("invalid --dbfilename %q: must be a plain file name", c.DBFilename)
		}
	}
	return c, nil
}

// ParseSize parses a memory size like Redis: a plain number of bytes, or a
// number with a unit. k/m/g are powers of 1000, kb/mb/gb powers of 1024.
func ParseSize(s string) (int64, error) {
	t := strings.ToLower(strings.TrimSpace(s))
	units := []struct {
		suffix string
		mult   int64
	}{
		{"kb", 1 << 10}, {"mb", 1 << 20}, {"gb", 1 << 30},
		{"k", 1e3}, {"m", 1e6}, {"g", 1e9}, {"b", 1},
	}
	mult := int64(1)
	for _, u := range units {
		if strings.HasSuffix(t, u.suffix) {
			t, mult = strings.TrimSuffix(t, u.suffix), u.mult
			break
		}
	}
	n, err := strconv.ParseInt(t, 10, 64)
	if err != nil || n < 0 || (n > 0 && mult > (1<<62)/n) {
		return 0, fmt.Errorf("%q is not a valid size (examples: 1048576, 512mb, 2gb)", s)
	}
	return n * mult, nil
}
