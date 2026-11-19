package index

import (
	"math"
	"math/rand/v2"
	"sort"
	"testing"
)

type sliceSource [][]float32

func (s sliceSource) NumSlots() int                { return len(s) }
func (s sliceSource) Vector(slot uint32) []float32 { return s[slot] }

func TestParseMetric(t *testing.T) {
	for _, s := range []string{"cosine", "COSINE", "l2", "ip"} {
		m, err := ParseMetric(s)
		if err != nil {
			t.Fatalf("%s: %v", s, err)
		}
		if m.String() != map[string]string{"cosine": "cosine", "COSINE": "cosine", "l2": "l2", "ip": "ip"}[s] {
			t.Errorf("%s -> %s", s, m)
		}
	}
	if _, err := ParseMetric("hamming"); err == nil {
		t.Error("expected error for unknown metric")
	}
}

func TestTopKKeepsBest(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	for _, k := range []int{1, 5, 100} {
		top := newTopK(k)
		var all []Hit
		for i := 0; i < 1000; i++ {
			h := Hit{Slot: uint32(i), Score: r.Float32()}
			all = append(all, h)
			top.push(h)
		}
		sort.Slice(all, func(i, j int) bool { return worse(all[j], all[i]) })
		got := top.sorted()
		if len(got) != k {
			t.Fatalf("k=%d: got %d hits", k, len(got))
		}
		for i := range got {
			if got[i] != all[i] {
				t.Fatalf("k=%d rank %d: got %+v want %+v", k, i, got[i], all[i])
			}
		}
	}
}

func TestFlatSearchMetrics(t *testing.T) {
	src := sliceSource{
		{1, 0},  // slot 0
		{0, 1},  // slot 1
		{2, 2},  // slot 2
		{-1, 0}, // slot 3
	}
	q := []float32{1, 0}

	// IP: dot products 1, 0, 2, -1
	hits := NewFlat(src, IP).Search(q, 2, nil, SearchParams{})
	if len(hits) != 2 || hits[0].Slot != 2 || hits[1].Slot != 0 || hits[0].Score != 2 {
		t.Fatalf("IP: %+v", hits)
	}

	// L2: distances 0, sqrt2, sqrt5, 2 -> scores are negated distances
	hits = NewFlat(src, L2).Search(q, 3, nil, SearchParams{})
	wantSlots := []uint32{0, 1, 3}
	wantScores := []float64{0, -math.Sqrt2, -2}
	for i := range hits {
		if hits[i].Slot != wantSlots[i] || math.Abs(float64(hits[i].Score)-wantScores[i]) > 1e-6 {
			t.Fatalf("L2 rank %d: %+v", i, hits[i])
		}
	}
}

func TestFlatSearchAcceptAndK(t *testing.T) {
	src := sliceSource{{1}, {2}, {3}, {4}}
	f := NewFlat(src, IP)

	hits := f.Search([]float32{1}, 10, func(s uint32) bool { return s != 3 }, SearchParams{})
	if len(hits) != 3 || hits[0].Slot != 2 {
		t.Fatalf("accept filter: %+v", hits)
	}
	if hits := f.Search([]float32{1}, 0, nil, SearchParams{}); hits != nil {
		t.Fatalf("k=0: %+v", hits)
	}
}
