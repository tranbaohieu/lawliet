package index

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/tranbaohieu/lawliet/internal/vecmath"
)

func flatten(vecs [][]float32) []float32 {
	out := make([]float32, 0, len(vecs)*len(vecs[0]))
	for _, v := range vecs {
		out = append(out, v...)
	}
	return out
}

// clusterLoss is the mean distance of each row to its nearest centroid:
// squared L2, or 1 − cosine if spherical.
func clusterLoss(data []float32, dim int, cent []float32, spherical bool) float64 {
	n := len(data) / dim
	assign := make([]int32, n)
	assignNearest(data, dim, cent, spherical, assign)
	var sum float64
	for i := 0; i < n; i++ {
		x, c := data[i*dim:(i+1)*dim], cent[int(assign[i])*dim:(int(assign[i])+1)*dim]
		if spherical {
			sum += 1 - float64(vecmath.Dot(x, c))
		} else {
			sum += float64(vecmath.L2Sq(x, c))
		}
	}
	return sum / float64(n)
}

func TestEMeansBatch(t *testing.T) {
	for _, tc := range []struct{ n, k int }{{2, 2}, {1000, 31}, {64 * 1024, 1024}, {1e6, 4096}} {
		b, a := emeansBatch(tc.n, tc.k, emeansEpochs)
		if b < 1 || b > max(1, tc.n/2) || b&(b-1) != 0 {
			t.Errorf("n=%d k=%d: batch %d", tc.n, tc.k, b)
		}
		if a < 1e-4 || a > 1 {
			t.Errorf("n=%d k=%d: alpha0 %g", tc.n, tc.k, a)
		}
	}
}

func TestEMeansBasics(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const n, dim, k = 4000, 16, 40
	for _, spherical := range []bool{false, true} {
		data := flatten(clustered(rng, n, dim, 20, spherical))
		cent := emeans(data, dim, k, emeansEpochs, spherical, 7)
		if len(cent) != k*dim {
			t.Fatalf("got %d centroids, want %d", len(cent)/dim, k)
		}
		for _, x := range cent {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				t.Fatalf("spherical=%v: non-finite centroid", spherical)
			}
		}
		counts := make([]int, k)
		assign := make([]int32, n)
		assignNearest(data, dim, cent, spherical, assign)
		for _, c := range assign {
			counts[c]++
		}
		for c, cnt := range counts {
			if cnt == 0 {
				t.Errorf("spherical=%v: cluster %d is empty", spherical, c)
			}
			if spherical {
				v := cent[c*dim : (c+1)*dim]
				if norm := vecmath.Dot(v, v); math.Abs(float64(norm)-1) > 1e-4 {
					t.Errorf("cluster %d: |c|² = %g, want 1", c, norm)
				}
			}
		}
		again := emeans(data, dim, k, emeansEpochs, spherical, 7)
		for i := range cent {
			if cent[i] != again[i] {
				t.Fatalf("spherical=%v: not deterministic for a fixed seed", spherical)
			}
		}
	}
	// k ≥ n and tiny inputs must not panic.
	data := flatten(clustered(rng, 3, dim, 2, false))
	if got := len(emeans(data, dim, 5, emeansEpochs, false, 1)) / dim; got != 3 {
		t.Fatalf("k > n: got %d centroids, want 3", got)
	}
	emeans(data[:dim], dim, 1, emeansEpochs, false, 1)
}

// TestTrainersCompare compares Lloyd (current default) and e-means at the
// IVF training scale: held-out clustering loss, training time, and IVF
// recall@10. e-means is also run on a 4× larger sample, the case where its
// mini-batches should pay off. Numbers are logged (go test -v -run Trainers);
// the asserts only guard against e-means being clearly worse.
func TestTrainersCompare(t *testing.T) {
	if testing.Short() {
		t.Skip("slow")
	}
	const dim, nlist, k = 32, 64, 10
	rng := rand.New(rand.NewSource(3))
	for _, metric := range []Metric{L2, Cosine} {
		spherical := metric != L2
		all := clusteredNoise(rng, 4*maxTrainPerCell*nlist+12000+100, dim, 200, 1.0, spherical)
		held := flatten(all[:10000])
		pool := all[10000:]

		type run struct {
			name    string
			train   trainFunc
			perCell int
		}
		runs := []run{
			{"lloyd", lloyd, maxTrainPerCell},
			{"emeans", emeansTrain, maxTrainPerCell},
			{"emeans-4x", emeansTrain, 4 * maxTrainPerCell},
		}
		loss := map[string]float64{}
		for _, r := range runs {
			sample := flatten(pool[:r.perCell*nlist])
			start := time.Now()
			cent := r.train(sample, dim, nlist, spherical, 1)
			el := time.Since(start)
			loss[r.name] = clusterLoss(held, dim, cent, spherical)
			t.Logf("%-6s %-9s sample=%6d train=%8v held-out loss=%.4f",
				metric, r.name, r.perCell*nlist, el.Round(time.Millisecond), loss[r.name])
		}
		if loss["emeans"] > 1.05*loss["lloyd"] {
			t.Errorf("%s: e-means loss %.4f > 1.05 × Lloyd %.4f", metric, loss["emeans"], loss["lloyd"])
		}

		// IVF recall with each trainer (nlist fixed, same data and queries).
		vecs, queries := pool[:len(pool)-100], pool[len(pool)-100:]
		recall := map[string]float64{}
		for _, r := range runs {
			func() {
				defer func(f trainFunc, p int) { trainCentroids, trainPerCell = f, p }(trainCentroids, trainPerCell)
				trainCentroids, trainPerCell = r.train, r.perCell
				idx, src := build(t, KindIVF, Options{NList: nlist, NProbe: 2}, metric, vecs)
				recall[r.name] = recallAt(t, idx, src, metric, queries, k, SearchParams{})
				t.Logf("%-6s %-9s IVF nprobe=2 recall@%d=%.3f (%s)", metric, r.name, k, recall[r.name], idx.Stats().State)
			}()
		}
		if recall["emeans"] < recall["lloyd"]-0.03 {
			t.Errorf("%s: e-means recall %.3f < Lloyd %.3f − 0.03", metric, recall["emeans"], recall["lloyd"])
		}
	}
}

// BenchmarkTrain times IVF centroid training (nlist 256, 64 points per cell,
// dim 128) with each trainer.
func BenchmarkTrain(b *testing.B) {
	const dim, nlist = 128, 256
	rng := rand.New(rand.NewSource(1))
	for _, perCell := range []int{maxTrainPerCell, 4 * maxTrainPerCell} {
		data := flatten(clustered(rng, perCell*nlist, dim, 500, false))
		for _, tr := range []struct {
			name string
			f    trainFunc
		}{{"lloyd", lloyd}, {"emeans", emeansTrain}} {
			b.Run(fmt.Sprintf("%s/n=%d", tr.name, perCell*nlist), func(b *testing.B) {
				var cent []float32
				for i := 0; i < b.N; i++ {
					cent = tr.f(data, dim, nlist, false, 1)
				}
				b.ReportMetric(clusterLoss(data, dim, cent, false), "loss")
			})
		}
	}
}
