package index

import (
	"math"
	"runtime"
	"sort"
	"sync"

	"github.com/tranbaohieu/lawliet/internal/vecmath"
)

// rawScore ranks vectors for metric m: higher is better. It is the dot product
// for cosine/ip and the negated squared distance for l2 (no sqrt per vector).
func rawScore(m Metric) func(a, b []float32) float32 {
	if m == L2 {
		return func(a, b []float32) float32 { return -vecmath.L2Sq(a, b) }
	}
	return vecmath.Dot
}

// finalize converts raw scores to reply scores (l2: negated distance).
func finalize(m Metric, hits []Hit) []Hit {
	if m == L2 {
		for i := range hits {
			hits[i].Score = -float32(math.Sqrt(float64(-hits[i].Score)))
		}
	}
	return hits
}

// exactSearch scores every accepted slot of src (the flat algorithm), raw scores.
func exactSearch(src Source, score func(a, b []float32) float32, query []float32, k int, accept func(uint32) bool) []Hit {
	top := newTopK(k)
	n := src.NumSlots()
	for i := 0; i < n; i++ {
		slot := uint32(i)
		if accept != nil && !accept(slot) {
			continue
		}
		top.push(Hit{Slot: slot, Score: score(query, src.Vector(slot))})
	}
	return top.sorted()
}

// rescore replaces approximate scores by exact raw scores and keeps the best k.
func rescore(src Source, score func(a, b []float32) float32, query []float32, hits []Hit, k int) []Hit {
	for i := range hits {
		hits[i].Score = score(query, src.Vector(hits[i].Slot))
	}
	sort.Slice(hits, func(i, j int) bool { return worse(hits[j], hits[i]) })
	if len(hits) > k {
		hits = hits[:k]
	}
	return hits
}

// parallelFor runs fn over [0, n) split into contiguous chunks, one goroutine per CPU.
func parallelFor(n int, fn func(lo, hi int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers > n {
		workers = n
	}
	if workers <= 1 {
		if n > 0 {
			fn(0, n)
		}
		return
	}
	var wg sync.WaitGroup
	chunk := (n + workers - 1) / workers
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			fn(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}
