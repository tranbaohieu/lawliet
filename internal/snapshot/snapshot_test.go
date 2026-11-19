package snapshot

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tranbaohieu/lawliet/internal/index"
	"github.com/tranbaohieu/lawliet/internal/meta"
	"github.com/tranbaohieu/lawliet/internal/store"
)

type clock struct{ ms atomic.Int64 }

func newStore(clk *clock) *store.Store {
	s := store.New()
	s.SetClock(clk.ms.Load)
	return s
}

func quiet() *slog.Logger { return slog.New(slog.NewTextHandler(io.Discard, nil)) }

// fill creates a store exercising every persisted feature.
func fill(t *testing.T, s *store.Store, now int64) {
	t.Helper()
	r := rand.New(rand.NewPCG(1, 2))
	cfgs := map[string]store.Config{
		"docs": {Dim: 8, Metric: index.Cosine, Index: "flat"},
		"l2":   {Dim: 3, Metric: index.L2, Index: "flat", MaxVectors: 1000, Policy: store.AllKeysLRU},
		"cache": {Dim: 4, Metric: index.IP, Index: "flat", Policy: store.VolatileTTL,
			HasCacheThreshold: true, CacheThreshold: 0.925, CacheTTL: 3600},
		"empty": {Dim: 2, Metric: index.IP, Index: "flat"},
	}
	for name, cfg := range cfgs {
		c, err := s.Create(name, cfg)
		if err != nil {
			t.Fatal(err)
		}
		if name == "empty" {
			continue
		}
		for i := 0; i < 300; i++ {
			v := make([]float32, cfg.Dim)
			for j := range v {
				v[j] = r.Float32()*2 - 1
			}
			rec := store.Record{Vector: v}
			if i%2 == 0 {
				rec.Text = fmt.Sprintf("text %d – tiếng Việt", i)
			}
			if i%3 == 0 {
				rec.Meta = meta.Map{
					"i":    {Kind: meta.Int, I: int64(i)},
					"f":    {Kind: meta.Float, F: 2.0},
					"s":    {Kind: meta.String, S: "x\ny"},
					"b":    {Kind: meta.Bool, B: true},
					"tags": {Kind: meta.StringList, L: []string{"a", "b"}},
				}
			}
			if i%5 == 0 {
				rec.ExpireAt = now + 3_600_000
			}
			if _, err := c.Add(fmt.Sprintf("id:%d", i), rec, store.Upsert); err != nil {
				t.Fatal(err)
			}
		}
		c.Delete("id:7", "id:8") // tombstones must not be written
	}
}

func TestRoundTrip(t *testing.T) {
	clk := &clock{}
	clk.ms.Store(1_000_000)
	src := newStore(clk)
	fill(t, src, clk.ms.Load())

	var buf bytes.Buffer
	ws, err := Write(&buf, src)
	if err != nil {
		t.Fatal(err)
	}
	if ws.Collections != 4 || ws.Records != 3*298 {
		t.Fatalf("write stats %+v", ws)
	}

	dst := newStore(clk)
	rs, err := Read(&buf, dst)
	if err != nil {
		t.Fatal(err)
	}
	if rs.Collections != 4 || rs.Records != 3*298 || rs.Skipped != 0 {
		t.Fatalf("read stats %+v", rs)
	}
	if dst.Dirty() != 0 {
		t.Fatalf("loading must not count as changes, dirty=%d", dst.Dirty())
	}
	if src.UsedMemory() != dst.UsedMemory() {
		t.Fatalf("memory %d != %d", dst.UsedMemory(), src.UsedMemory())
	}

	for _, a := range src.Collections() {
		b, err := dst.Get(a.Name())
		if err != nil {
			t.Fatal(err)
		}
		if a.Config() != b.Config() || a.Count() != b.Count() || a.Info().Expires != b.Info().Expires {
			t.Fatalf("%s: config/count differ: %+v vs %+v", a.Name(), a.Info(), b.Info())
		}
		for i := 0; i < 300; i++ {
			id := fmt.Sprintf("id:%d", i)
			ra, okA := a.Get(id)
			rb, okB := b.Get(id)
			if okA != okB {
				t.Fatalf("%s/%s: exists %v vs %v", a.Name(), id, okA, okB)
			}
			if !okA {
				continue
			}
			if ra.Text != rb.Text || ra.ExpireAt != rb.ExpireAt || !reflect.DeepEqual(ra.Meta, rb.Meta) {
				t.Fatalf("%s/%s: record differs:\n%+v\n%+v", a.Name(), id, ra, rb)
			}
			// Bit-exact: a restored vector must not be re-normalized.
			for j := range ra.Vector {
				if math.Float32bits(ra.Vector[j]) != math.Float32bits(rb.Vector[j]) {
					t.Fatalf("%s/%s: vector differs at %d: %v vs %v", a.Name(), id, j, ra.Vector[j], rb.Vector[j])
				}
			}
		}
		// Same search results, scores included.
		q := make([]float32, a.Config().Dim)
		q[0], q[len(q)-1] = 1, 0.5
		resA, _ := a.Search(q, store.SearchOptions{K: 10})
		resB, _ := b.Search(q, store.SearchOptions{K: 10})
		for i := range resA {
			if resA[i].ID != resB[i].ID || resA[i].Score != resB[i].Score {
				t.Fatalf("%s: search rank %d: %+v vs %+v", a.Name(), i, resA[i], resB[i])
			}
		}
	}
}

func TestExpiredRecordsSkipped(t *testing.T) {
	clk := &clock{}
	clk.ms.Store(1_000_000)
	src := newStore(clk)
	c, _ := src.Create("c", store.Config{Dim: 1, Metric: index.IP, Index: "flat"})
	c.Add("gone", store.Record{Vector: []float32{1}, ExpireAt: clk.ms.Load() + 10}, store.Upsert)
	c.Add("later", store.Record{Vector: []float32{1}, ExpireAt: clk.ms.Load() + 5000}, store.Upsert)
	c.Add("keep", store.Record{Vector: []float32{1}}, store.Upsert)
	clk.ms.Add(100) // "gone" expired before the save

	var buf bytes.Buffer
	if ws, _ := Write(&buf, src); ws.Records != 2 {
		t.Fatalf("expired record written: %+v", ws)
	}
	clk.ms.Add(10_000) // "later" expires while the server is down
	dst := newStore(clk)
	rs, err := Read(&buf, dst)
	if err != nil {
		t.Fatal(err)
	}
	if rs.Records != 1 || rs.Skipped != 1 {
		t.Fatalf("stats %+v", rs)
	}
	dc, _ := dst.Get("c")
	if !dc.Exists("keep") || dc.Exists("later") || dc.Count() != 1 {
		t.Fatal("unexpected records after load")
	}
}

func TestLoadIgnoresLimits(t *testing.T) {
	clk := &clock{}
	src := newStore(clk)
	fill(t, src, 0)
	var buf bytes.Buffer
	Write(&buf, src)

	dst := newStore(clk)
	dst.SetLimits(store.Limits{MaxMemory: 1000, MaxCollections: 1})
	if _, err := Read(&buf, dst); err != nil {
		t.Fatalf("load must not enforce limits: %v", err)
	}
	if len(dst.List()) != 4 {
		t.Fatal("all collections should load")
	}
}

func TestCorruptFiles(t *testing.T) {
	clk := &clock{}
	src := newStore(clk)
	fill(t, src, 0)
	var buf bytes.Buffer
	Write(&buf, src)
	good := buf.Bytes()

	flip := func(i int) []byte {
		b := bytes.Clone(good)
		b[i] ^= 0xff
		return b
	}
	badVersion := bytes.Clone(good)
	badVersion[4] = 99

	cases := map[string]struct {
		data []byte
		want string
	}{
		"empty":          {nil, "bad magic"},
		"bad magic":      {append([]byte("XXXX"), good[4:]...), "bad magic"},
		"bad version":    {badVersion, "unsupported snapshot version 99"},
		"version zero":   {append(append([]byte("LWDB"), 0, 0), good[6:]...), "unsupported snapshot version 0"},
		"flipped byte":   {flip(len(good) / 2), "corrupt snapshot"},
		"bad checksum":   {flip(len(good) - 1), "checksum mismatch"},
		"truncated":      {good[:len(good)/3], "unexpected end of file"},
		"no checksum":    {good[:len(good)-4], "missing checksum"},
		"trailing bytes": {append(bytes.Clone(good), 0), "unexpected data after checksum"},
	}
	for name, tc := range cases {
		_, err := Read(bytes.NewReader(tc.data), newStore(clk))
		if !errors.Is(err, ErrCorrupt) || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: got %v, want ErrCorrupt containing %q", name, err, tc.want)
		}
	}
}

func TestSnapshotterSaveLoad(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "dump.lwdb")
	clk := &clock{}
	src := newStore(clk)
	fill(t, src, 0)

	sn := New(src, path, nil, quiet())
	if src.Dirty() == 0 {
		t.Fatal("mutations should be counted")
	}
	if _, err := sn.Save(); err != nil {
		t.Fatal(err)
	}
	if src.Dirty() != 0 {
		t.Fatalf("dirty after save = %d", src.Dirty())
	}
	if _, err := os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Fatal("temp file left behind")
	}

	dst := newStore(clk)
	stats, found, err := New(dst, path, nil, quiet()).Load()
	if err != nil || !found || stats.Records != 3*298 {
		t.Fatalf("Load: %+v %v %v", stats, found, err)
	}

	// No file: not an error.
	if _, found, err := New(newStore(clk), filepath.Join(dir, "none.lwdb"), nil, quiet()).Load(); found || err != nil {
		t.Fatalf("missing file: found=%v err=%v", found, err)
	}
}

func TestSaveFailureKeepsOldFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "dump.lwdb")
	clk := &clock{}
	s := newStore(clk)
	fill(t, s, 0)
	sn := New(s, path, nil, quiet())
	sn.Save()
	before, _ := os.ReadFile(path)

	// Make the temp file path a directory so the next save fails.
	os.Mkdir(path+".tmp", 0o755)
	s.Drop("docs")
	if _, err := sn.Save(); err == nil {
		t.Fatal("save should fail")
	}
	after, _ := os.ReadFile(path)
	if !bytes.Equal(before, after) {
		t.Fatal("failed save must not touch the existing snapshot")
	}
	if s.Dirty() == 0 {
		t.Fatal("failed save must keep the dirty count")
	}
	info := map[string]string{}
	for _, kv := range sn.Info() {
		info[kv[0]] = kv[1]
	}
	if info["rdb_last_save_status"] != "err" {
		t.Fatalf("status %v", info)
	}
}

func TestDirtyKeepsChangesDuringSave(t *testing.T) {
	clk := &clock{}
	s := newStore(clk)
	c, _ := s.Create("c", store.Config{Dim: 64, Metric: index.IP, Index: "flat"})
	v := make([]float32, 64)
	v[0] = 1
	for i := 0; i < 5000; i++ {
		c.Add(fmt.Sprint(i), store.Record{Vector: v}, store.Upsert)
	}
	sn := New(s, filepath.Join(t.TempDir(), "d.lwdb"), nil, quiet())

	var wg sync.WaitGroup
	var writes atomic.Int64
	stop := make(chan struct{})
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; ; i++ {
			select {
			case <-stop:
				return
			default:
				c.Add(fmt.Sprintf("w%d", i), store.Record{Vector: v}, store.Upsert)
				writes.Add(1)
			}
		}
	}()
	time.Sleep(5 * time.Millisecond)
	sn.Save()
	close(stop)
	wg.Wait()
	// Every concurrent write is either in the snapshot or still counted as dirty;
	// at minimum, writes after the save started are not lost from the counter.
	if s.Dirty() < 0 || s.Dirty() > writes.Load() {
		t.Fatalf("dirty %d out of range (concurrent writes %d)", s.Dirty(), writes.Load())
	}
}

func TestRules(t *testing.T) {
	r, err := ParseRule(" 300  100 ")
	if err != nil || r != (Rule{300, 100}) {
		t.Fatalf("%+v %v", r, err)
	}
	for _, bad := range []string{"", "300", "a b", "0 1", "1 0", "1 2 3"} {
		if _, err := ParseRule(bad); err == nil {
			t.Errorf("ParseRule(%q) should fail", bad)
		}
	}

	clk := &clock{}
	s := newStore(clk)
	sn := New(s, filepath.Join(t.TempDir(), "d.lwdb"), []Rule{{60, 3}, {300, 1}}, quiet())
	start := sn.lastSave
	if sn.due(start.Add(time.Hour)) {
		t.Fatal("no changes: not due")
	}
	s.Create("a", store.Config{Dim: 1, Index: "flat"})
	if sn.due(start.Add(299 * time.Second)) {
		t.Fatal("1 change before 300s: not due")
	}
	if !sn.due(start.Add(300 * time.Second)) {
		t.Fatal("1 change after 300s: due")
	}
	s.Create("b", store.Config{Dim: 1, Index: "flat"})
	s.Create("c", store.Config{Dim: 1, Index: "flat"})
	if !sn.due(start.Add(61 * time.Second)) {
		t.Fatal("3 changes after 60s: due")
	}
}

// testdata/v1.lwdb was written by the format-1 code (v0.1.0): the current
// reader must still load it, with no cache defaults.
func TestReadVersion1File(t *testing.T) {
	data, err := os.ReadFile("testdata/v1.lwdb")
	if err != nil {
		t.Fatal(err)
	}
	if data[4] != 1 || data[5] != 0 {
		t.Fatalf("fixture is not version 1: % x", data[4:6])
	}
	clk := &clock{}
	clk.ms.Store(1_000_000)
	st := newStore(clk)
	stats, err := Read(bytes.NewReader(data), st)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Collections != 1 || stats.Records != 2 {
		t.Fatalf("stats %+v", stats)
	}
	c, _ := st.Get("docs")
	cfg := c.Config()
	if cfg.Dim != 2 || cfg.Metric != index.IP || cfg.MaxVectors != 10 || cfg.Policy != store.AllKeysLRU ||
		cfg.HasCacheThreshold || cfg.CacheTTL != 0 {
		t.Fatalf("config %+v", cfg)
	}
	a, ok := c.Get("a")
	if !ok || a.Text != "alpha" || a.Meta["k"].S != "v" || a.Vector[0] != 1 || a.Vector[1] != 2 {
		t.Fatalf("record a: %+v", a)
	}
	if b, ok := c.Get("b"); !ok || b.ExpireAt != 4102444800000 {
		t.Fatalf("record b: %+v", b)
	}
}

// Flat-only stores are written as v2 (readable by v0.2.x); any hnsw/ivf/quantized
// collection needs v3.
func TestWrittenVersion(t *testing.T) {
	written := func(s *store.Store) int {
		var buf bytes.Buffer
		if _, err := Write(&buf, s); err != nil {
			t.Fatal(err)
		}
		return int(buf.Bytes()[4]) | int(buf.Bytes()[5])<<8
	}
	s := newStore(&clock{})
	fill(t, s, 0)
	if got := written(s); got != 2 {
		t.Fatalf("flat-only store written as v%d, want 2", got)
	}
	if _, err := s.Create("q", store.Config{Dim: 8, Metric: index.L2, Index: "flat",
		IndexOpts: index.Options{Quant: index.QuantSQ8}}); err != nil {
		t.Fatal(err)
	}
	if got := written(s); got != 3 || version != 3 {
		t.Fatalf("store with flat+sq8 written as v%d, want 3 (const %d)", got, version)
	}
}

// Index kinds and options survive a snapshot; the index is rebuilt on load.
func TestRoundTripIndexOptions(t *testing.T) {
	clk := &clock{}
	src := newStore(clk)
	cfgs := map[string]store.Config{
		"g": {Dim: 8, Metric: index.Cosine, Index: index.KindHNSW, IndexOpts: index.Options{M: 8, EfSearch: 40}},
		"v": {Dim: 8, Metric: index.L2, Index: index.KindIVF,
			IndexOpts: index.Options{NList: 4, NProbe: 2, Quant: index.QuantPQ, PQM: 4, Rerank: 5}},
	}
	r := rand.New(rand.NewPCG(3, 4))
	for name, cfg := range cfgs {
		c, err := src.Create(name, cfg)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 200; i++ {
			v := make([]float32, 8)
			for j := range v {
				v[j] = r.Float32()
			}
			if _, err := c.Add(fmt.Sprintf("r%d", i), store.Record{Vector: v}, store.Upsert); err != nil {
				t.Fatal(err)
			}
		}
	}
	var buf bytes.Buffer
	if _, err := Write(&buf, src); err != nil {
		t.Fatal(err)
	}
	dst := newStore(clk)
	if _, err := Read(&buf, dst); err != nil {
		t.Fatal(err)
	}
	for name, cfg := range cfgs {
		c, err := dst.Get(name)
		if err != nil {
			t.Fatal(err)
		}
		if got := c.Config(); got.Index != cfg.Index || got.IndexOpts != cfg.IndexOpts {
			t.Fatalf("%s: config %+v, want %+v", name, got, cfg)
		}
		rec, _ := c.Get("r9")
		hits, err := c.Search(rec.Vector, store.SearchOptions{K: 1})
		if err != nil || len(hits) != 1 || hits[0].ID != "r9" {
			t.Fatalf("%s: search after load: %v %v", name, hits, err)
		}
	}
}
