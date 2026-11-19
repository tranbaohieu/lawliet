package store

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
	"sync"
	"testing"

	"github.com/tranbaohieu/lawliet/internal/filter"
	"github.com/tranbaohieu/lawliet/internal/index"
	"github.com/tranbaohieu/lawliet/internal/meta"
)

func mustCreate(t *testing.T, s *Store, name string, dim int, m index.Metric) *Collection {
	t.Helper()
	c, err := s.Create(name, Config{Dim: dim, Metric: m, Index: "flat"})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestStoreCreateDropList(t *testing.T) {
	s := New()
	mustCreate(t, s, "b", 2, index.IP)
	mustCreate(t, s, "a", 2, index.IP)
	if _, err := s.Create("a", Config{Dim: 2, Index: "flat"}); !errors.Is(err, ErrCollectionExists) {
		t.Fatalf("duplicate: %v", err)
	}
	if got := s.List(); fmt.Sprint(got) != "[a b]" {
		t.Fatalf("List = %v", got)
	}
	if !s.Drop("a") || s.Drop("a") {
		t.Fatal("Drop should succeed once")
	}
	if _, err := s.Get("a"); !errors.Is(err, ErrCollectionNotFound) {
		t.Fatalf("Get dropped: %v", err)
	}
}

func TestCreateValidation(t *testing.T) {
	s := New()
	bad := []struct {
		name string
		cfg  Config
	}{
		{"", Config{Dim: 2, Index: "flat"}},
		{"x", Config{Dim: 0, Index: "flat"}},
		{"x", Config{Dim: MaxDim + 1, Index: "flat"}},
		{"x", Config{Dim: 2, Index: "annoy"}},
		{"x", Config{Dim: 2, Index: "hnsw", IndexOpts: index.Options{NList: 4}}},
		{"x", Config{Dim: 2, Metric: 99, Index: "flat"}},
	}
	for _, b := range bad {
		var ce *ConfigError
		if _, err := s.Create(b.name, b.cfg); !errors.As(err, &ce) {
			t.Errorf("Create(%q, %+v): want ConfigError, got %v", b.name, b.cfg, err)
		}
	}
}

func TestAddModesAndGet(t *testing.T) {
	c := mustCreate(t, New(), "c", 2, index.IP)

	if r, err := c.Add("a", Record{Vector: []float32{1, 2}}, Upsert); err != nil || r != Inserted {
		t.Fatalf("insert: %v %v", r, err)
	}
	if r, _ := c.Add("a", Record{Vector: []float32{3, 4}}, OnlyNew); r != Skipped {
		t.Fatalf("NX on existing: %v", r)
	}
	if r, _ := c.Add("b", Record{Vector: []float32{1, 1}}, OnlyExisting); r != Skipped {
		t.Fatalf("XX on missing: %v", r)
	}
	if r, _ := c.Add("a", Record{Vector: []float32{5, 6}}, OnlyExisting); r != Updated {
		t.Fatalf("XX on existing: %v", r)
	}
	if r, ok := c.Get("a"); !ok || r.Vector[0] != 5 || r.Vector[1] != 6 {
		t.Fatalf("Get after update: %v %v", r, ok)
	}
	if c.Exists("b") || c.Count() != 1 {
		t.Fatalf("Exists(b)=%v Count=%d", c.Exists("b"), c.Count())
	}
}

func TestAddValidation(t *testing.T) {
	cos := mustCreate(t, New(), "c", 3, index.Cosine)
	var dm *DimMismatchError
	if _, err := cos.Add("a", Record{Vector: []float32{1, 2}}, Upsert); !errors.As(err, &dm) || dm.Expected != 3 || dm.Got != 2 {
		t.Fatalf("dim mismatch: %v", err)
	}
	if _, err := cos.Add("a", Record{Vector: []float32{0, 0, 0}}, Upsert); !errors.Is(err, ErrZeroVector) {
		t.Fatalf("zero vector: %v", err)
	}
	if _, err := cos.Add("a", Record{Vector: []float32{1, float32(math.NaN()), 0}}, Upsert); !errors.Is(err, ErrNonFinite) {
		t.Fatalf("NaN: %v", err)
	}
	if cos.Count() != 0 {
		t.Fatal("invalid vectors must not be stored")
	}
}

func TestCosineStoresNormalizedCopy(t *testing.T) {
	c := mustCreate(t, New(), "c", 2, index.Cosine)
	in := []float32{3, 4}
	if _, err := c.Add("a", Record{Vector: in}, Upsert); err != nil {
		t.Fatal(err)
	}
	if in[0] != 3 || in[1] != 4 {
		t.Fatal("caller's slice was modified")
	}
	rec, _ := c.Get("a")
	v := rec.Vector
	if math.Abs(float64(v[0])-0.6) > 1e-6 || math.Abs(float64(v[1])-0.8) > 1e-6 {
		t.Fatalf("stored %v, want [0.6 0.8]", v)
	}
}

func TestDeleteReusesSlots(t *testing.T) {
	c := mustCreate(t, New(), "c", 1, index.IP)
	for i := 0; i < 5; i++ {
		c.Add(fmt.Sprint(i), Record{Vector: []float32{float32(i)}}, Upsert)
	}
	if n := c.Delete("1", "3", "missing"); n != 2 {
		t.Fatalf("Delete = %d", n)
	}
	if c.Exists("1") || c.Count() != 3 {
		t.Fatal("deleted record still present")
	}
	c.Add("x", Record{Vector: []float32{9}}, Upsert)
	c.Add("y", Record{Vector: []float32{8}}, Upsert)
	if info := c.Info(); info.Slots != 5 || info.Count != 5 {
		t.Fatalf("slots should be reused: %+v", info)
	}
	// Deleted records never appear in search results.
	res, _ := c.Search([]float32{1}, SearchOptions{K: 10})
	ids := make([]string, len(res))
	for i, r := range res {
		ids[i] = r.ID
	}
	if fmt.Sprint(ids) != "[x y 4 2 0]" {
		t.Fatalf("search ids = %v", ids)
	}
}

func TestSlabCrossesChunks(t *testing.T) {
	c := mustCreate(t, New(), "c", 3, index.IP)
	n := ChunkSize*2 + 7
	for i := 0; i < n; i++ {
		c.Add(fmt.Sprint(i), Record{Vector: []float32{float32(i), 1, -1}}, Upsert)
	}
	for _, i := range []int{0, ChunkSize - 1, ChunkSize, 2 * ChunkSize, n - 1} {
		rec, ok := c.Get(fmt.Sprint(i))
		v := rec.Vector
		if !ok || v[0] != float32(i) || v[1] != 1 || v[2] != -1 {
			t.Fatalf("slot %d: %v %v", i, v, ok)
		}
	}
	if info := c.Info(); info.VectorBytes != 3*ChunkSize*3*4 {
		t.Fatalf("VectorBytes = %d", info.VectorBytes)
	}
}

// bruteForce computes expected top-k IDs in float64 for comparison.
func bruteForce(m index.Metric, data map[string][]float32, q []float32, k int) []string {
	norm := func(v []float32) []float64 {
		out := make([]float64, len(v))
		var n float64
		for i, x := range v {
			out[i] = float64(x)
			n += out[i] * out[i]
		}
		if m == index.Cosine {
			n = math.Sqrt(n)
			for i := range out {
				out[i] /= n
			}
		}
		return out
	}
	qq := norm(q)
	type scored struct {
		id    string
		score float64
	}
	var all []scored
	for id, v := range data {
		vv := norm(v)
		var s float64
		for i := range vv {
			if m == index.L2 {
				d := vv[i] - qq[i]
				s -= d * d
			} else {
				s += vv[i] * qq[i]
			}
		}
		all = append(all, scored{id, s})
	}
	sort.Slice(all, func(i, j int) bool { return all[i].score > all[j].score })
	ids := make([]string, k)
	for i := range ids {
		ids[i] = all[i].id
	}
	return ids
}

func TestSearchMatchesBruteForce(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 8))
	for _, m := range []index.Metric{index.Cosine, index.L2, index.IP} {
		t.Run(m.String(), func(t *testing.T) {
			const dim, n, k = 64, 3000, 10
			c := mustCreate(t, New(), "c", dim, m)
			data := make(map[string][]float32, n)
			for i := 0; i < n; i++ {
				v := make([]float32, dim)
				for j := range v {
					v[j] = r.Float32()*2 - 1
				}
				id := fmt.Sprintf("doc:%d", i)
				data[id] = v
				if _, err := c.Add(id, Record{Vector: v}, Upsert); err != nil {
					t.Fatal(err)
				}
			}
			for trial := 0; trial < 5; trial++ {
				q := make([]float32, dim)
				for j := range q {
					q[j] = r.Float32()*2 - 1
				}
				res, err := c.Search(q, SearchOptions{K: k})
				if err != nil {
					t.Fatal(err)
				}
				want := bruteForce(m, data, q, k)
				for i := range want {
					if res[i].ID != want[i] {
						t.Fatalf("trial %d rank %d: got %s want %s", trial, i, res[i].ID, want[i])
					}
				}
				for i := 1; i < len(res); i++ {
					if res[i].Score > res[i-1].Score {
						t.Fatalf("results not sorted: %v", res)
					}
				}
			}
		})
	}
}

func TestConcurrentAccess(t *testing.T) {
	c := mustCreate(t, New(), "c", 8, index.Cosine)
	var wg sync.WaitGroup
	for w := 0; w < 4; w++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			for i := 0; i < 500; i++ {
				id := fmt.Sprintf("%d-%d", w, i%50)
				c.Add(id, Record{Vector: []float32{1, float32(i), 0, 0, 0, 0, 0, 1}}, Upsert)
				if i%7 == 0 {
					c.Delete(id)
				}
			}
		}()
		go func() {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				if _, err := c.Search([]float32{1, 0, 0, 0, 0, 0, 0, 0}, SearchOptions{K: 5, WithVectors: true, WithText: true, WithMeta: true}); err != nil {
					t.Error(err)
					return
				}
				c.Info()
			}
		}()
	}
	wg.Wait()
	if info := c.Info(); info.Count != len(c.idToSlot) {
		t.Fatalf("count %d != map size %d", info.Count, len(c.idToSlot))
	}
}

func TestTextAndMeta(t *testing.T) {
	c := mustCreate(t, New(), "c", 2, index.IP)
	m := meta.Map{"source": {Kind: meta.String, S: "hr.md"}}
	c.Add("a", Record{Vector: []float32{1, 0}, Text: "hello", Meta: m}, Upsert)

	rec, ok := c.Get("a")
	if !ok || rec.Text != "hello" || rec.Meta["source"].S != "hr.md" {
		t.Fatalf("Get: %+v %v", rec, ok)
	}

	// Upsert replaces the whole record: text and meta not given are cleared.
	c.Add("a", Record{Vector: []float32{0, 1}}, Upsert)
	if rec, _ := c.Get("a"); rec.Text != "" || rec.Meta != nil {
		t.Fatalf("upsert should clear text/meta: %+v", rec)
	}

	// A reused slot must not leak the previous record's text/meta.
	c.Add("b", Record{Vector: []float32{1, 1}, Text: "old", Meta: m}, Upsert)
	c.Delete("b")
	c.Add("c", Record{Vector: []float32{1, 1}}, Upsert)
	if rec, _ := c.Get("c"); rec.Text != "" || rec.Meta != nil {
		t.Fatalf("reused slot leaked data: %+v", rec)
	}
}

func TestSearchWithFilterAndFields(t *testing.T) {
	c := mustCreate(t, New(), "c", 1, index.IP)
	for i := 0; i < 10; i++ {
		m := meta.Map{"even": {Kind: meta.Bool, B: i%2 == 0}}
		c.Add(fmt.Sprint(i), Record{Vector: []float32{float32(i)}, Text: fmt.Sprintf("t%d", i), Meta: m}, Upsert)
	}
	f, err := filter.Parse("even == true")
	if err != nil {
		t.Fatal(err)
	}

	// The filter is applied before ranking: top 3 of the even IDs, not evens among the top 3.
	res, err := c.Search([]float32{1}, SearchOptions{K: 3, Filter: f.Match, WithText: true, WithMeta: true})
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, r := range res {
		got = append(got, r.ID+"/"+r.Text)
		if !r.Meta["even"].B || r.Vector != nil {
			t.Fatalf("unexpected result fields: %+v", r)
		}
	}
	if fmt.Sprint(got) != "[8/t8 6/t6 4/t4]" {
		t.Fatalf("got %v", got)
	}

	// Without With* options, optional fields stay empty.
	res, _ = c.Search([]float32{1}, SearchOptions{K: 1})
	if res[0].Text != "" || res[0].Meta != nil {
		t.Fatalf("fields should not be set: %+v", res[0])
	}

	// A filter that matches nothing returns no results.
	none, _ := filter.Parse("even == 1")
	if res, _ := c.Search([]float32{1}, SearchOptions{K: 3, Filter: none.Match}); len(res) != 0 {
		t.Fatalf("want no results, got %+v", res)
	}
}

func TestRestoreKeepsVectorBits(t *testing.T) {
	s := New()
	src := mustCreate(t, s, "src", 3, index.Cosine)
	src.Add("a", Record{Vector: []float32{0.3, -1.7, 2.9}}, Upsert)
	stored, _ := src.Get("a")

	dst := mustCreate(t, s, "dst", 3, index.Cosine)
	if ok, err := dst.Restore("a", Record{Vector: stored.Vector}); !ok || err != nil {
		t.Fatalf("Restore: %v %v", ok, err)
	}
	got, _ := dst.Get("a")
	for i := range got.Vector {
		if math.Float32bits(got.Vector[i]) != math.Float32bits(stored.Vector[i]) {
			t.Fatalf("vector changed at %d: %v vs %v", i, got.Vector, stored.Vector)
		}
	}
	if _, err := dst.Restore("a", Record{Vector: stored.Vector}); !errors.Is(err, ErrDuplicateID) {
		t.Fatalf("duplicate: %v", err)
	}
	var dm *DimMismatchError
	if _, err := dst.Restore("b", Record{Vector: []float32{1}}); !errors.As(err, &dm) {
		t.Fatalf("dim: %v", err)
	}
}
