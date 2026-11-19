package index

import (
	"math/rand"

	"github.com/tranbaohieu/lawliet/internal/vecmath"
)

// kmeans clusters n vectors of length dim (row-major in data) into k centroids
// with Lloyd iterations, like Faiss Clustering: random distinct initial points,
// empty clusters re-seeded by splitting the largest one.
//
// spherical (cosine) assigns by dot product and keeps centroids unit length;
// otherwise assignment is by squared L2 distance.
func kmeans(data []float32, dim, k, iters int, spherical bool, seed int64) []float32 {
	n := len(data) / dim
	if k > n {
		k = n
	}
	rng := rand.New(rand.NewSource(seed))
	cent := make([]float32, k*dim)
	for i, p := range rng.Perm(n)[:k] {
		copy(cent[i*dim:(i+1)*dim], data[p*dim:(p+1)*dim])
	}
	assign := make([]int32, n)
	for it := 0; it < iters; it++ {
		assignNearest(data, dim, cent, spherical, assign)

		sums := make([]float64, k*dim)
		counts := make([]int, k)
		for i := 0; i < n; i++ {
			c := int(assign[i])
			counts[c]++
			row, s := data[i*dim:(i+1)*dim], sums[c*dim:(c+1)*dim]
			for j, x := range row {
				s[j] += float64(x)
			}
		}
		for c := 0; c < k; c++ {
			if counts[c] == 0 {
				continue
			}
			inv := 1 / float64(counts[c])
			for j := 0; j < dim; j++ {
				cent[c*dim+j] = float32(sums[c*dim+j] * inv)
			}
		}
		splitEmpty(cent, dim, counts, rng)
		if spherical {
			for c := 0; c < k; c++ {
				vecmath.Normalize(cent[c*dim : (c+1)*dim])
			}
		}
	}
	return cent
}

// splitEmpty gives each empty cluster half of the largest one: a slightly
// perturbed copy of its centroid (Faiss uses the same trick).
func splitEmpty(cent []float32, dim int, counts []int, rng *rand.Rand) {
	const eps = 1.0 / 1024
	for c := range counts {
		if counts[c] != 0 {
			continue
		}
		big := 0
		for j := range counts {
			if counts[j] > counts[big] {
				big = j
			}
		}
		if counts[big] < 2 {
			return
		}
		src, dst := cent[big*dim:(big+1)*dim], cent[c*dim:(c+1)*dim]
		for j := range dst {
			if rng.Intn(2) == 0 {
				dst[j], src[j] = src[j]*(1+eps), src[j]*(1-eps)
			} else {
				dst[j], src[j] = src[j]*(1-eps), src[j]*(1+eps)
			}
		}
		counts[c], counts[big] = counts[big]/2, counts[big]-counts[big]/2
	}
}

// assignNearest writes the nearest centroid of each row into out, in parallel.
func assignNearest(data []float32, dim int, cent []float32, spherical bool, out []int32) {
	k := len(cent) / dim
	var norms []float32 // ||c||² for argmin ||x-c||² = argmin ||c||² - 2 x·c
	if !spherical {
		norms = make([]float32, k)
		for c := 0; c < k; c++ {
			v := cent[c*dim : (c+1)*dim]
			norms[c] = vecmath.Dot(v, v)
		}
	}
	parallelFor(len(data)/dim, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			out[i] = int32(nearest(data[i*dim:(i+1)*dim], cent, dim, norms))
		}
	})
}

// nearest returns the centroid closest to x: max dot if norms is nil, else min L2.
func nearest(x, cent []float32, dim int, norms []float32) int {
	best, bestScore := 0, float32(0)
	for c := 0; c*dim < len(cent); c++ {
		d := vecmath.Dot(x, cent[c*dim:(c+1)*dim])
		s := d
		if norms != nil {
			s = 2*d - norms[c]
		}
		if c == 0 || s > bestScore {
			best, bestScore = c, s
		}
	}
	return best
}
