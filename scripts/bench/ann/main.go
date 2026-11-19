// Command ann benchmarks Lawliet's indexes in-process on a prepared dataset
// (see prepare_data.py and faiss_bench.py, which writes the ground truth).
//
// Records go through store.Collection (the same path as VADD / VSEARCH, minus
// RESP), inserted one by one in dataset order. For each index configuration it
// reports build time (including training), the slowest single insert (training
// pause), index memory, and per search setting: recall@10, single-thread latency
// and multi-thread throughput. A filtered run (10% of records match) checks
// recall against exact filtered search.
//
//	go run ./scripts/bench/ann -data ~/lawliet-bench/data -out results-lawliet.json
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/tranbaohieu/lawliet/internal/filter"
	"github.com/tranbaohieu/lawliet/internal/index"
	"github.com/tranbaohieu/lawliet/internal/meta"
	"github.com/tranbaohieu/lawliet/internal/store"
)

type config struct {
	Name  string
	Kind  string
	Opts  index.Options
	Sweep []index.SearchParams
}

func configs(nlist int) []config {
	efs := func(vs ...int) (out []index.SearchParams) {
		for _, v := range vs {
			out = append(out, index.SearchParams{EfSearch: v})
		}
		return
	}
	probes := func(rerank int, vs ...int) (out []index.SearchParams) {
		for _, v := range vs {
			out = append(out, index.SearchParams{NProbe: v, Rerank: rerank})
		}
		return
	}
	reranks := func(vs ...int) (out []index.SearchParams) {
		for _, v := range vs {
			out = append(out, index.SearchParams{Rerank: v})
		}
		return
	}
	pq := index.Options{Quant: index.QuantPQ, PQM: 64}
	ivf := func(o index.Options) index.Options { o.NList = nlist; return o }
	return []config{
		{"flat", index.KindFlat, index.Options{}, []index.SearchParams{{}}},
		{"flat-sq8", index.KindFlat, index.Options{Quant: index.QuantSQ8}, reranks(1, 4)},
		{"flat-pq64", index.KindFlat, pq, reranks(1, 4, 10)},
		{"hnsw16", index.KindHNSW, index.Options{M: 16, EfConstruction: 100}, efs(16, 32, 64, 128, 256)},
		{"hnsw32", index.KindHNSW, index.Options{M: 32, EfConstruction: 200}, efs(16, 32, 64, 128, 256)},
		{"ivf", index.KindIVF, ivf(index.Options{}), probes(0, 1, 4, 8, 16, 32, 64)},
		{"ivf-sq8", index.KindIVF, ivf(index.Options{Quant: index.QuantSQ8}), probes(1, 8, 16, 32)},
		{"ivf-pq64", index.KindIVF, ivf(pq), append(probes(1, 16, 32), probes(10, 16, 32)...)},
	}
}

type searchResult struct {
	Params    index.SearchParams `json:"params"`
	Recall    float64            `json:"recall_at_10"`
	P50ms     float64            `json:"p50_ms"`
	P99ms     float64            `json:"p99_ms"`
	QPS1      float64            `json:"qps_1_thread"`
	QPSAll    float64            `json:"qps_all_threads"`
	Threads   int                `json:"threads"`
	FilterRec *float64           `json:"filtered_recall_at_10,omitempty"`
	FilterP50 *float64           `json:"filtered_p50_ms,omitempty"`
}

type result struct {
	Engine      string         `json:"engine"`
	Name        string         `json:"name"`
	Describe    string         `json:"describe"`
	BuildSec    float64        `json:"build_s"`
	MaxInsertMs float64        `json:"max_insert_ms"`
	IndexBytes  int64          `json:"index_bytes"`
	HeapBytes   uint64         `json:"heap_bytes"`
	State       string         `json:"state"`
	Searches    []searchResult `json:"searches"`
	Churn       *churnResult   `json:"churn,omitempty"`
}

// churnResult: recall before and after deleting a fraction of the records and
// inserting them again (same ids and vectors, new slots), like TTL expiry and
// eviction in a semantic cache. The ground truth stays valid.
type churnResult struct {
	Fraction     float64            `json:"fraction"`
	Rounds       int                `json:"rounds"`
	Params       index.SearchParams `json:"params"`
	RecallBefore float64            `json:"recall_before"`
	RecallAfter  float64            `json:"recall_after"`
	Seconds      float64            `json:"seconds"`
	StateAfter   string             `json:"state_after"`
}

func main() {
	dataDir := flag.String("data", "data", "dataset directory")
	only := flag.String("configs", "", "comma-separated config names (default: all)")
	out := flag.String("out", "results-lawliet.json", "output file (JSON lines, appended)")
	threads := flag.Int("threads", runtime.GOMAXPROCS(0), "threads for the throughput run")
	nlist := flag.Int("nlist", 316, "NLIST for ivf configs")
	churn := flag.Float64("churn", 0, "fraction of records to delete and re-insert per round (0 = skip)")
	churnRounds := flag.Int("churn-rounds", 3, "churn rounds")
	cpuprofile := flag.String("cpuprofile", "", "write a CPU profile of the whole run to this file")
	flag.Parse()
	if *cpuprofile != "" {
		pf, err := os.Create(*cpuprofile)
		must(err)
		must(pprof.StartCPUProfile(pf))
		defer pprof.StopCPUProfile()
	}

	var info struct {
		N, Queries, Dim int
	}
	must(json.Unmarshal(mustRead(filepath.Join(*dataDir, "info.json")), &info))
	corpus := readF32(filepath.Join(*dataDir, "corpus.f32"), info.Dim)
	queries := readF32(filepath.Join(*dataDir, "queries.f32"), info.Dim)
	gt := readI32(filepath.Join(*dataDir, "gt.i32"), len(queries))
	fmt.Printf("dataset: %d × %d, %d queries, ground truth top-%d\n", len(corpus), info.Dim, len(queries), len(gt[0]))

	f, err := os.OpenFile(*out, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	must(err)
	defer f.Close()

	var filtGT [][]int // exact filtered top-10, computed once with the flat index
	for _, cfg := range configs(*nlist) {
		if *only != "" && !slices.Contains(strings.Split(*only, ","), cfg.Name) {
			continue
		}
		r, ft := run(cfg, corpus, queries, gt, filtGT, *threads, *churn, *churnRounds)
		if filtGT == nil {
			filtGT = ft
		}
		line, _ := json.Marshal(r)
		f.Write(append(line, '\n'))
	}
}

// filterExpr matches 10% of the records (meta b = i mod 10).
const filterExpr = "b == 3"

func run(cfg config, corpus, queries [][]float32, gt [][]int, filtGT [][]int, threads int, churn float64, rounds int) (result, [][]int) {
	runtime.GC()
	var m0 runtime.MemStats
	runtime.ReadMemStats(&m0)

	st := store.New()
	col, err := st.Create("bench", store.Config{
		Dim: len(corpus[0]), Metric: index.Cosine, Index: cfg.Kind, IndexOpts: cfg.Opts,
	})
	must(err)
	metas := make([]meta.Map, 10)
	for b := range metas {
		metas[b], err = meta.ParseJSON([]byte(fmt.Sprintf(`{"b":%d}`, b)))
		must(err)
	}

	start := time.Now()
	var maxInsert time.Duration
	for i, v := range corpus {
		t := time.Now()
		_, err := col.Add(strconv.Itoa(i), store.Record{Vector: v, Meta: metas[i%10]}, store.Upsert)
		must(err)
		maxInsert = max(maxInsert, time.Since(t))
		if (i+1)%20000 == 0 {
			fmt.Printf("  %s: %d inserted, %.0fs\n", cfg.Name, i+1, time.Since(start).Seconds())
		}
	}
	build := time.Since(start)
	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)
	info := col.Info()

	res := result{
		Engine: "lawliet", Name: cfg.Name, Describe: cfg.Kind + " " + index.Describe(cfg.Kind, len(corpus[0]), cfg.Opts),
		BuildSec: build.Seconds(), MaxInsertMs: ms(maxInsert),
		IndexBytes: info.IndexStats.Bytes, HeapBytes: m1.HeapAlloc - m0.HeapAlloc, State: info.IndexStats.State,
	}
	fmt.Printf("%s: built in %.1fs (max insert %.0f ms), index %.1f MiB, %s\n",
		cfg.Name, res.BuildSec, res.MaxInsertMs, float64(res.IndexBytes)/(1<<20), res.State)

	flt, err := filter.Parse(filterExpr)
	must(err)
	if filtGT == nil && cfg.Kind == index.KindFlat && cfg.Opts.Quant == index.QuantNone {
		filtGT = make([][]int, len(queries))
		for qi, q := range queries {
			filtGT[qi] = ids(search(col, q, index.SearchParams{}, flt.Match))
		}
	}

	for _, p := range cfg.Sweep {
		sr := searchResult{Params: p, Threads: threads}
		lat := make([]float64, len(queries))
		hits := 0
		for qi, q := range queries {
			t := time.Now()
			got := search(col, q, p, nil)
			lat[qi] = ms(time.Since(t))
			hits += overlap(ids(got), gt[qi][:10])
		}
		sr.Recall = float64(hits) / float64(10*len(queries))
		sr.P50ms, sr.P99ms = pct(lat, 50), pct(lat, 99)
		sr.QPS1 = 1000 / mean(lat)
		sr.QPSAll = throughput(col, queries, p, threads)

		if filtGT != nil {
			fh, flat := 0, make([]float64, len(queries))
			for qi, q := range queries {
				t := time.Now()
				got := search(col, q, p, flt.Match)
				flat[qi] = ms(time.Since(t))
				fh += overlap(ids(got), filtGT[qi])
			}
			fr, fp := float64(fh)/float64(10*len(queries)), pct(flat, 50)
			sr.FilterRec, sr.FilterP50 = &fr, &fp
		}
		fmt.Printf("  %-28s recall@10 %.4f  p50 %6.2f ms  p99 %6.2f ms  qps(1) %7.0f  qps(%d) %7.0f\n",
			fmt.Sprintf("%+v", p), sr.Recall, sr.P50ms, sr.P99ms, sr.QPS1, threads, sr.QPSAll)
		res.Searches = append(res.Searches, sr)
	}
	if churn > 0 {
		p := cfg.Sweep[len(cfg.Sweep)/2]
		c := &churnResult{Fraction: churn, Rounds: rounds, Params: p, RecallBefore: recallOf(col, queries, gt, p)}
		rng := rand.New(rand.NewPCG(9, 9))
		t := time.Now()
		for r := 0; r < rounds; r++ {
			perm := rng.Perm(len(corpus))[:int(churn*float64(len(corpus)))]
			for _, i := range perm {
				col.Delete(strconv.Itoa(i))
			}
			for _, i := range perm {
				_, err := col.Add(strconv.Itoa(i), store.Record{Vector: corpus[i], Meta: metas[i%10]}, store.Upsert)
				must(err)
			}
		}
		c.Seconds = time.Since(t).Seconds()
		c.RecallAfter = recallOf(col, queries, gt, p)
		c.StateAfter = col.Info().IndexStats.State
		res.Churn = c
		fmt.Printf("  churn %d × %.0f%% (%.0fs): recall@10 %.4f -> %.4f at %+v (%s)\n",
			rounds, 100*churn, c.Seconds, c.RecallBefore, c.RecallAfter, p, c.StateAfter)
	}
	st.Drop("bench")
	return res, filtGT
}

func recallOf(col *store.Collection, queries [][]float32, gt [][]int, p index.SearchParams) float64 {
	hits := 0
	for qi, q := range queries {
		hits += overlap(ids(search(col, q, p, nil)), gt[qi][:10])
	}
	return float64(hits) / float64(10*len(queries))
}

func search(col *store.Collection, q []float32, p index.SearchParams, f func(meta.Map) bool) []store.Result {
	r, err := col.Search(q, store.SearchOptions{K: 10, NoTouch: true, Params: p, Filter: f})
	must(err)
	return r
}

func throughput(col *store.Collection, queries [][]float32, p index.SearchParams, threads int) float64 {
	const rounds = 3
	var next atomic.Int64
	total := int64(rounds * len(queries))
	start := time.Now()
	var wg sync.WaitGroup
	for w := 0; w < threads; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := next.Add(1) - 1; i < total; i = next.Add(1) - 1 {
				search(col, queries[i%int64(len(queries))], p, nil)
			}
		}()
	}
	wg.Wait()
	return float64(total) / time.Since(start).Seconds()
}

func ids(rs []store.Result) []int {
	out := make([]int, len(rs))
	for i, r := range rs {
		out[i], _ = strconv.Atoi(r.ID)
	}
	return out
}

func overlap(a, b []int) int {
	n := 0
	for _, x := range a {
		if slices.Contains(b, x) {
			n++
		}
	}
	return n
}

func pct(v []float64, p float64) float64 {
	s := slices.Clone(v)
	slices.Sort(s)
	return s[min(len(s)-1, int(math.Ceil(p/100*float64(len(s))))-1)]
}

func mean(v []float64) float64 {
	t := 0.0
	for _, x := range v {
		t += x
	}
	return t / float64(len(v))
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

func readF32(path string, dim int) [][]float32 {
	b := mustRead(path)
	n := len(b) / (4 * dim)
	flat := make([]float32, n*dim)
	for i := range flat {
		flat[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	out := make([][]float32, n)
	for i := range out {
		out[i] = flat[i*dim : (i+1)*dim : (i+1)*dim]
	}
	return out
}

func readI32(path string, rows int) [][]int {
	b := mustRead(path)
	cols := len(b) / 4 / rows
	out := make([][]int, rows)
	for i := range out {
		out[i] = make([]int, cols)
		for j := range out[i] {
			out[i][j] = int(int32(binary.LittleEndian.Uint32(b[4*(i*cols+j):])))
		}
	}
	return out
}

func mustRead(path string) []byte {
	b, err := os.ReadFile(path)
	must(err)
	return b
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "ann:", err)
		os.Exit(1)
	}
}
