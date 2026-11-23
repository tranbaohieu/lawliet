package index

import (
	"math"
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

// trainFunc clusters n vectors of length dim (row-major in data) into k centroids.
type trainFunc func(data []float32, dim, k int, spherical bool, seed int64) []float32

// lloyd is the IVF trainer: kmeans with kmeansIters iterations.
func lloyd(data []float32, dim, k int, spherical bool, seed int64) []float32 {
	return kmeans(data, dim, k, kmeansIters, spherical, seed)
}

// emeansTrain is the IVF trainer variant using emeans with default epochs.
func emeansTrain(data []float32, dim, k int, spherical bool, seed int64) []float32 {
	return emeans(data, dim, k, emeansEpochs, spherical, seed)
}

// emeansEpochs is the default number of passes over the data (reference code).
const emeansEpochs = 5

// splitEps is the relative perturbation used to split a cluster in two when
// reviving it (BF16 machine epsilon, e-means paper Appendix H).
const splitEps = 1.0 / 128

// emeansBatch returns the mini-batch size and initial step size from the
// scaling laws of the reference implementation (Faiss contrib/e_means):
//
//	B  = 0.035134 · k^0.134749 · n^0.876473 · T^0.813344, rounded to a power of two, at most n/2
//	α0 = 0.8244 · k^-0.4617 · B^0.9003 · n^-0.4498 · T^-0.1057, clamped to [1e-4, 1]
func emeansBatch(n, k, epochs int) (batch int, alpha0 float64) {
	fk, fn, ft := float64(k), float64(n), float64(epochs)
	b := 0.035134 * math.Pow(fk, 0.134749) * math.Pow(fn, 0.876473) * math.Pow(ft, 0.813344)
	batch = 1
	if b > 1 {
		batch = 1 << int(math.Round(math.Log2(b)))
	}
	batch = max(1, min(batch, n/2))
	alpha0 = 0.8244 * math.Pow(fk, -0.4617) * math.Pow(float64(batch), 0.9003) *
		math.Pow(fn, -0.4498) * math.Pow(ft, -0.1057)
	return batch, math.Min(1, math.Max(1e-4, alpha0))
}

// emeans clusters like kmeans but with e-means (FAIR at Meta, arXiv 2611.09501),
// a mini-batch k-means that keeps per cluster a decayed count N_j and sum S_j.
// For each mini-batch with per-cluster counts n_j and sums s_j:
//
//	N_j ← γN_j + αn_j,  S_j ← γS_j + αs_j,  centroid c_j = S_j / N_j
//
// with α = 1−γ annealed by α(t) = α0·(0.1 + 0.9·½(1+cos(πt/M))) over M batches.
// Every five batches, clusters whose count fell below 10% of the mean are
// revived by splitting a donor drawn with probability proportional to its count.
// The result is the mean of the points assigned during the last epoch.
//
// spherical (cosine) assigns by dot product and returns unit-length centroids.
func emeans(data []float32, dim, k, epochs int, spherical bool, seed int64) []float32 {
	n := len(data) / dim
	if k > n {
		k = n
	}
	rng := rand.New(rand.NewSource(seed))
	cent := make([]float32, k*dim)
	for i, p := range rng.Perm(n)[:k] {
		copy(cent[i*dim:(i+1)*dim], data[p*dim:(p+1)*dim])
	}
	if n < 2 || k < 2 {
		return cent
	}
	batch, alpha0 := emeansBatch(n, k, epochs)

	N := make([]float64, k)
	S := make([]float64, k*dim)
	for c := range N {
		N[c] = 1e-10
		for j, v := range cent[c*dim : (c+1)*dim] {
			S[c*dim+j] = N[c] * float64(v)
		}
	}

	perBatch := max(1, n/batch) // batches per epoch
	total := epochs * perBatch
	lastEpoch := total - perBatch

	perm := rng.Perm(n)
	pos := 0
	xb := make([]float32, batch*dim)
	assign := make([]int32, batch)
	bn := make([]float64, k)     // batch counts
	bs := make([]float64, k*dim) // batch sums
	fn := make([]float64, k)     // last-epoch counts
	fs := make([]float64, k*dim) // last-epoch sums

	for t := 0; t < total; t++ {
		for i := 0; i < batch; i++ {
			if pos == n {
				rng.Shuffle(n, func(a, b int) { perm[a], perm[b] = perm[b], perm[a] })
				pos = 0
			}
			p := perm[pos]
			copy(xb[i*dim:(i+1)*dim], data[p*dim:(p+1)*dim])
			pos++
		}
		assignNearest(xb, dim, cent, spherical, assign)

		clear(bn)
		clear(bs)
		for i := 0; i < batch; i++ {
			c := int(assign[i])
			bn[c]++
			s := bs[c*dim : (c+1)*dim]
			for j, x := range xb[i*dim : (i+1)*dim] {
				s[j] += float64(x)
			}
		}
		if t >= lastEpoch {
			for c := range fn {
				fn[c] += bn[c]
			}
			for i, v := range bs {
				fs[i] += v
			}
		}

		alpha := alpha0 * (0.1 + 0.9*0.5*(1+math.Cos(math.Pi*float64(t)/float64(total))))
		gamma := 1 - alpha
		for c := range N {
			N[c] = gamma*N[c] + alpha*bn[c]
		}
		for i := range S {
			S[i] = gamma*S[i] + alpha*bs[i]
		}
		if (t+1)%5 == 0 {
			reviveClusters(N, S, dim, rng)
		}
		meansInto(cent, S, N, dim, spherical)
	}
	meansInto(cent, fs, fn, dim, spherical) // clusters empty in the last epoch keep their centroid
	return cent
}

// meansInto sets cent[c] = sums[c]/counts[c] for every non-empty cluster
// (unit length if spherical).
func meansInto(cent []float32, sums, counts []float64, dim int, spherical bool) {
	for c, cnt := range counts {
		if cnt <= 0 {
			continue
		}
		inv, row := 1/cnt, cent[c*dim:(c+1)*dim]
		for j := range row {
			row[j] = float32(sums[c*dim+j] * inv)
		}
		if spherical {
			vecmath.Normalize(row)
		}
	}
}

// reviveClusters re-seeds clusters whose decayed count is below 10% of the mean
// by splitting a donor (drawn without replacement, probability ∝ count): the
// donor's count and sum are halved between the two, the sums pushed apart by splitEps.
func reviveClusters(N, S []float64, dim int, rng *rand.Rand) {
	var total float64
	for _, x := range N {
		total += x
	}
	thr := 0.1 * total / float64(len(N))
	used := make([]bool, len(N))
	for c := range N {
		if N[c] >= thr {
			continue
		}
		used[c] = true
		var avail float64
		for i, x := range N {
			if !used[i] && x >= thr {
				avail += x
			}
		}
		if avail <= 0 {
			return
		}
		donor, u := -1, rng.Float64()*avail
		for i, x := range N {
			if used[i] || x < thr {
				continue
			}
			donor = i
			if u -= x; u <= 0 {
				break
			}
		}
		used[donor] = true
		half := N[donor] / 2
		N[donor], N[c] = half, half
		sd, sc := S[donor*dim:(donor+1)*dim], S[c*dim:(c+1)*dim]
		for j := range sd {
			v := sd[j] / 2
			sc[j], sd[j] = v*(1+splitEps), v*(1-splitEps)
		}
	}
}
