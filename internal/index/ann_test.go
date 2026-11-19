package index

import (
	"math/rand"
	"testing"

	"github.com/tranbaohieu/lawliet/internal/vecmath"
)

// memSource is an index.Source backed by a slice, with deletions.
type memSource struct {
	dim  int
	vecs [][]float32
	dead map[uint32]bool
}

func (m *memSource) NumSlots() int             { return len(m.vecs) }
func (m *memSource) Vector(s uint32) []float32 { return m.vecs[s] }
func (m *memSource) accept(s uint32) bool      { return !m.dead[s] }
func (m *memSource) set(s uint32, v []float32) { m.vecs[s] = v; delete(m.dead, s) }

// clustered returns n vectors around nc random centers (realistic for embeddings).
// Each call draws new centers: take data and queries from the same call
// (split) so queries are in-distribution, as in RAG.
func clustered(rng *rand.Rand, n, dim, nc int, normalize bool) [][]float32 {
	centers := make([][]float32, nc)
	for i := range centers {
		centers[i] = make([]float32, dim)
		for j := range centers[i] {
			centers[i][j] = float32(rng.NormFloat64())
		}
	}
	out := make([][]float32, n)
	for i := range out {
		c := centers[rng.Intn(nc)]
		v := make([]float32, dim)
		for j := range v {
			v[j] = c[j] + 0.5*float32(rng.NormFloat64())
		}
		if normalize {
			vecmath.Normalize(v)
		}
		out[i] = v
	}
	return out
}

func recallAt(t *testing.T, idx Index, src *memSource, metric Metric, queries [][]float32, k int, p SearchParams) float64 {
	t.Helper()
	exact := NewFlat(src, metric)
	found, total := 0, 0
	for _, q := range queries {
		want := map[uint32]bool{}
		for _, h := range exact.Search(q, k, src.accept, SearchParams{}) {
			want[h.Slot] = true
		}
		got := idx.Search(q, k, src.accept, p)
		if len(got) != len(want) {
			t.Fatalf("got %d hits, want %d", len(got), len(want))
		}
		for i, h := range got {
			if i > 0 && got[i-1].Score < h.Score {
				t.Fatalf("hits not sorted: %v", got)
			}
			if want[h.Slot] {
				found++
			}
		}
		total += len(want)
	}
	return float64(found) / float64(total)
}

func build(t *testing.T, kind string, o Options, metric Metric, vecs [][]float32) (Index, *memSource) {
	t.Helper()
	src := &memSource{dim: len(vecs[0]), dead: map[uint32]bool{}}
	idx, err := New(src, metric, src.dim, kind, o)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range vecs {
		src.vecs = append(src.vecs, v)
		idx.Add(uint32(i), v)
	}
	return idx, src
}

func TestANNRecall(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const n, dim, k = 12000, 32, 10
	cases := []struct {
		name string
		kind string
		o    Options
		min  float64
	}{
		{"hnsw", KindHNSW, Options{}, 0.95},
		{"ivf", KindIVF, Options{NProbe: 16}, 0.90},
		{"ivf-sq8", KindIVF, Options{NProbe: 16, Quant: QuantSQ8, Rerank: 2}, 0.88},
		{"ivf-pq", KindIVF, Options{NProbe: 16, Quant: QuantPQ, PQM: 8, Rerank: 10}, 0.85},
		{"flat-sq8", KindFlat, Options{Quant: QuantSQ8, Rerank: 2}, 0.97},
		{"flat-pq", KindFlat, Options{Quant: QuantPQ, PQM: 8, Rerank: 10}, 0.90},
	}
	for _, metric := range []Metric{Cosine, L2, IP} {
		all := clustered(rng, n+100, dim, 50, metric == Cosine) // ip: raw, unnormalized vectors
		vecs, queries := all[:n], all[n:]
		for _, tc := range cases {
			t.Run(metric.String()+"/"+tc.name, func(t *testing.T) {
				idx, src := build(t, tc.kind, tc.o, metric, vecs)
				r := recallAt(t, idx, src, metric, queries, k, SearchParams{})
				t.Logf("recall@%d = %.3f  (%s)", k, r, idx.Stats().State)
				if r < tc.min {
					t.Errorf("recall %.3f < %.2f", r, tc.min)
				}
			})
		}
	}
}

func TestQuantizedScoresAreExact(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	vecs := clustered(rng, 12000, 16, 20, true)
	for _, o := range []Options{{Quant: QuantSQ8}, {Quant: QuantPQ, PQM: 4}} {
		q := o.Quant
		idx, src := build(t, KindIVF, o, Cosine, vecs)
		for _, h := range idx.Search(vecs[5], 5, src.accept, SearchParams{}) {
			if want := vecmath.Dot(vecs[5], vecs[h.Slot]); h.Score != want {
				t.Fatalf("%s: score %v, exact %v", q, h.Score, want)
			}
		}
	}
}

func TestSearchParamsOverride(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	data := clustered(rng, 12050, 32, 50, true)
	vecs, queries := data[:12000], data[12000:]
	ivf, src := build(t, KindIVF, Options{NProbe: 1}, Cosine, vecs)
	low := recallAt(t, ivf, src, Cosine, queries, 10, SearchParams{})
	full := recallAt(t, ivf, src, Cosine, queries, 10, SearchParams{NProbe: MaxNList})
	if full != 1 || low >= full {
		t.Errorf("nprobe=1 recall %.3f, nprobe=all recall %.3f (want < and == 1)", low, full)
	}
	hnsw, src := build(t, KindHNSW, Options{EfSearch: 10}, Cosine, vecs)
	if r := recallAt(t, hnsw, src, Cosine, queries, 10, SearchParams{EfSearch: 400}); r < 0.98 {
		t.Errorf("hnsw ef=400 recall %.3f", r)
	}
}

// Deletes, updates and slot reuse keep results correct (tombstones never returned).
func TestANNMutations(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	for _, kind := range []struct {
		name string
		kind string
		o    Options
	}{
		{"hnsw", KindHNSW, Options{}},
		{"ivf", KindIVF, Options{}},
		{"ivf-pq", KindIVF, Options{Quant: QuantPQ, PQM: 8, Rerank: 4}},
	} {
		t.Run(kind.name, func(t *testing.T) {
			all := clustered(rng, 16000, 32, 50, true)
			vecs, fresh := all[:12000], all[12000:]
			idx, src := build(t, kind.kind, kind.o, Cosine, vecs)
			// delete a third, re-use half of those slots with new vectors, update some live ones
			for s := uint32(0); s < 4000; s++ {
				idx.Remove(s)
				src.dead[s] = true
			}
			for s := uint32(0); s < 2000; s++ {
				src.set(s, fresh[s])
				idx.Add(s, fresh[s])
			}
			for s := uint32(5000); s < 5500; s++ {
				idx.Remove(s)
				src.set(s, fresh[s-3000])
				idx.Add(s, fresh[s-3000])
			}
			for _, q := range fresh[:200] {
				for _, h := range idx.Search(q, 10, src.accept, SearchParams{}) {
					if src.dead[h.Slot] {
						t.Fatalf("deleted slot %d returned", h.Slot)
					}
				}
			}
			if r := recallAt(t, idx, src, Cosine, fresh[:100], 10, SearchParams{}); r < 0.8 {
				t.Errorf("recall after mutations %.3f", r)
			}
		})
	}
}

// A selective filter still returns k hits (fallback to exact).
func TestANNSelectiveFilter(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	vecs := clustered(rng, 12000, 32, 50, true)
	for _, kind := range []string{KindHNSW, KindIVF} {
		idx, src := build(t, kind, Options{}, Cosine, vecs)
		accept := func(s uint32) bool { return s%997 == 0 } // 13 matches
		hits := idx.Search(vecs[1], 10, accept, SearchParams{})
		want := NewFlat(src, Cosine).Search(vecs[1], 10, accept, SearchParams{})
		if len(hits) != 10 {
			t.Fatalf("%s: %d hits", kind, len(hits))
		}
		for i := range hits {
			if hits[i] != want[i] {
				t.Fatalf("%s: hit %d = %v, want %v", kind, i, hits[i], want[i])
			}
		}
	}
}

// IVF is exact until trained, trains at the threshold and retrains as it grows.
func TestIVFTraining(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	vecs := clustered(rng, 4100, 8, 10, true)
	src := &memSource{dim: 8, dead: map[uint32]bool{}}
	ivf := NewIVF(src, Cosine, 8, Options{}, 0)
	for i, v := range vecs {
		src.vecs = append(src.vecs, v)
		ivf.Add(uint32(i), v)
		switch i + 1 {
		case ivfMinTrain - 1:
			if ivf.trained {
				t.Fatal("trained too early")
			}
		case ivfMinTrain:
			if !ivf.trained || ivf.nlist != 31 {
				t.Fatalf("trained=%v nlist=%d at %d", ivf.trained, ivf.nlist, i+1)
			}
		case 4000:
			if ivf.trainedAt != 4000 || ivf.nlist != 63 {
				t.Fatalf("trainedAt=%d nlist=%d", ivf.trainedAt, ivf.nlist)
			}
		}
	}
	total := 0
	for _, l := range ivf.lists {
		total += len(l.slots)
	}
	if total != len(vecs) {
		t.Fatalf("%d records in lists, want %d", total, len(vecs))
	}
}

func TestValidate(t *testing.T) {
	bad := []struct {
		kind string
		o    Options
	}{
		{"annoy", Options{}},
		{KindHNSW, Options{NList: 10}},
		{KindHNSW, Options{Quant: QuantSQ8}},
		{KindIVF, Options{M: 8}},
		{KindFlat, Options{NProbe: 2}},
		{KindIVF, Options{Quant: QuantPQ, PQM: 7}}, // does not divide 32
		{KindIVF, Options{PQM: 8}},
		{KindIVF, Options{Rerank: 2}},
		{KindFlat, Options{Quant: 7}}, // unknown value (e.g. from a snapshot)
		{KindIVF, Options{Quant: QuantPQ + 1}},
	}
	for _, b := range bad {
		if err := Validate(b.kind, 32, b.o); err == nil {
			t.Errorf("Validate(%s, %+v) accepted", b.kind, b.o)
		}
	}
	if err := Validate(KindIVF, 1024, Options{Quant: QuantPQ}); err != nil {
		t.Error(err)
	}
	if got := autoPQM(1024); got != 64 {
		t.Errorf("autoPQM(1024) = %d", got)
	}
}

func TestSQ8SIMDMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	for _, dim := range []int{8, 16, 17, 100, 1024} {
		data := clustered(rng, 300, dim, 5, true)
		flat := make([]float32, 0, 300*dim)
		for _, v := range data {
			flat = append(flat, v...)
		}
		q := trainSQ(flat, dim)
		code := make([]byte, dim)
		q.encode(data[1], code)
		for _, m := range []Metric{Cosine, L2} {
			p := q.prepare(data[0], m)
			got, want := p.score(code), p.scoreScalar(code)
			if d := got - want; d > 1e-4 || d < -1e-4 {
				t.Errorf("dim %d %s: %v vs scalar %v", dim, m, got, want)
			}
		}
	}
}

// Repeated delete + re-insert (cache churn) keeps HNSW recall close to a fresh build.
func TestHNSWChurnKeepsRecall(t *testing.T) {
	// Reused slots get new vectors, as with TTL expiry or eviction followed by new
	// records. (Re-inserting the same vector into the same slot would flatter a graph
	// that keeps links to removed nodes: they come back valid.)
	for _, seed := range []int64{10, 11} {
		rng := rand.New(rand.NewSource(seed))
		const n, rounds = 12000, 3
		churn := n * 3 / 10
		pool := clustered(rng, n+rounds*churn+1000, 32, 50, true)
		vecs, extra, queries := pool[:n], pool[n:n+rounds*churn], pool[n+rounds*churn:]
		idx, src := build(t, KindHNSW, Options{EfSearch: 32}, Cosine, vecs)
		for round := 0; round < rounds; round++ {
			perm := rng.Perm(n)[:churn]
			for _, i := range perm {
				idx.Remove(uint32(i))
				src.dead[uint32(i)] = true
			}
			for j, i := range perm {
				v := extra[round*churn+j]
				src.set(uint32(i), v)
				idx.Add(uint32(i), v)
			}
		}
		after := recallAt(t, idx, src, Cosine, queries, 10, SearchParams{})
		fresh, fsrc := build(t, KindHNSW, Options{EfSearch: 32}, Cosine, src.vecs)
		want := recallAt(t, fresh, fsrc, Cosine, queries, 10, SearchParams{})
		t.Logf("seed %d: recall@10 after 3 × 30%% churn %.3f, fresh build on the same data %.3f (%s)",
			seed, after, want, idx.Stats().State)
		if after < want-0.01 {
			t.Errorf("seed %d: recall after churn %.3f < fresh %.3f - 0.01", seed, after, want)
		}
	}
}

// Removing every node empties the graph; it can be filled again.
func TestHNSWRemoveAll(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	vecs := clustered(rng, 500, 8, 5, true)
	idx, src := build(t, KindHNSW, Options{}, Cosine, vecs)
	h := idx.(*HNSW)
	for i := range vecs {
		h.Remove(uint32(i))
	}
	if h.entry != -1 || h.inGraph != 0 || len(h.Search(vecs[0], 5, nil, SearchParams{})) != 0 {
		t.Fatalf("entry %d, %d nodes after removing all", h.entry, h.inGraph)
	}
	for i, v := range vecs {
		h.Add(uint32(i), v)
	}
	if r := recallAt(t, idx, src, Cosine, vecs[:50], 5, SearchParams{}); r < 0.95 {
		t.Errorf("recall after refill %.3f", r)
	}
}

// Links to removed nodes are dropped when a list is touched again (connect),
// not kept as candidates that block live neighbours in the heuristic.
func TestHNSWDropsLinksToRemovedNodes(t *testing.T) {
	rng := rand.New(rand.NewSource(12))
	vecs := clustered(rng, 6000, 16, 20, true)
	idx, src := build(t, KindHNSW, Options{}, Cosine, vecs[:3000])
	h := idx.(*HNSW)
	dead := func() (n int) {
		for _, node := range h.nodes {
			for l, links := range node.links {
				for _, x := range links {
					if !h.inLevel(x, l) {
						n++
					}
				}
			}
		}
		return n
	}
	for i := 0; i < 1000; i++ {
		h.Remove(uint32(i))
		src.dead[uint32(i)] = true
	}
	afterRemove := dead()
	for i := 3000; i < 6000; i++ {
		src.vecs = append(src.vecs, vecs[i])
		h.Add(uint32(i), vecs[i])
	}
	afterInserts := dead()
	t.Logf("links to removed nodes: %d after removing 1/3, %d after 3000 inserts", afterRemove, afterInserts)
	if afterInserts*3 > afterRemove {
		t.Errorf("dead links %d → %d: not dropped by later inserts", afterRemove, afterInserts)
	}
}
