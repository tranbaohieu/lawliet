package index

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/tranbaohieu/lawliet/internal/vecmath"
)

// IVF is an inverted-file index (Faiss IndexIVFFlat / IVFScalarQuantizer / IVFPQ):
// k-means centroids split the space into NLIST cells; a query scans the NPROBE
// cells whose centroids score best.
//
// Codes per cell are stored contiguously: none = slots only (vectors are read
// from the collection), sq8 = 1 byte per dimension of the vector, pq = PQ_M
// bytes encoding the residual (vector − centroid).
//
// Training needs data, so the index is built lazily: until the collection
// holds trainMin records every search is exact; then it trains on a sample and
// assigns all records. It retrains whenever the record count doubles, while the
// sample is not yet saturated (centroids follow the data as it grows). Training
// runs inside the mutation that crosses the threshold (under the collection
// write lock).
//
// With a fixed nlist of 1 (flat + QUANT) it is a flat scan over codes (Faiss
// IndexScalarQuantizer / IndexPQ).
type IVF struct {
	src    Source
	metric Metric
	dim    int
	opts   Options
	fixed  int // fixed nlist (1 for flat+quant); 0 = opts.NList or auto
	score  func(a, b []float32) float32

	present []bool // slot is a live record known to the index
	listOf  []int32
	posOf   []int32
	n       int

	trained   bool
	trainedAt int
	nlist     int
	cent      []float32
	centNorms []float32 // l2 only
	lists     []ivfList
	sq        *scalarQuantizer
	pq        *productQuantizer
	codeSize  int
}

type ivfList struct {
	slots []uint32
	codes []byte
}

const (
	ivfMinTrain     = 1000
	pqMinTrain      = 10000 // Faiss warns below 39 points per PQ centroid (39 × 256)
	maxTrainPerCell = 64    // k-means points per cell (Faiss: 256 with BLAS; lower bounds pure-Go training time)
	pqMaxTrain      = 32768
	sqMaxTrain      = 65536
	kmeansIters     = 10
)

// NewIVF returns an IVF index. fixedNList > 0 overrides opts.NList.
func NewIVF(src Source, metric Metric, dim int, opts Options, fixedNList int) *IVF {
	return &IVF{src: src, metric: metric, dim: dim, opts: opts, fixed: fixedNList, score: rawScore(metric)}
}

func (x *IVF) wantNList(n int) int {
	switch {
	case x.fixed > 0:
		return x.fixed
	case x.opts.NList > 0:
		return x.opts.NList
	}
	return max(1, min(MaxNList, int(math.Sqrt(float64(n)))))
}

func (x *IVF) trainMin() int {
	m := ivfMinTrain
	if x.opts.Quant == QuantPQ {
		m = pqMinTrain
	}
	if x.fixed == 0 && x.opts.NList > 0 {
		m = max(m, 39*x.opts.NList)
	}
	return m
}

func (x *IVF) grow(slot uint32) {
	for int(slot) >= len(x.present) {
		x.present = append(x.present, false)
		x.listOf = append(x.listOf, -1)
		x.posOf = append(x.posOf, -1)
	}
}

// Add indexes slot (an update first removes the old entry).
func (x *IVF) Add(slot uint32, vec []float32) {
	x.grow(slot)
	if x.present[slot] {
		x.Remove(slot)
	}
	x.present[slot] = true
	x.n++
	switch {
	case !x.trained && x.n >= x.trainMin():
		x.train()
	case x.trained && x.shouldRetrain():
		x.train()
	case x.trained:
		code := make([]byte, x.codeSize)
		l := x.encode(vec, code)
		x.appendTo(l, slot, code)
	}
}

// sampleCap is the largest useful training sample for nlist cells.
func (x *IVF) sampleCap(nlist int) int {
	c := maxTrainPerCell * nlist
	switch x.opts.Quant {
	case QuantPQ:
		c = max(c, pqMaxTrain)
	case QuantSQ8:
		c = max(c, sqMaxTrain)
	}
	return c
}

// shouldRetrain: the count doubled since the last training, and either nlist is
// auto (it grows with n) or the last sample was below the useful maximum.
func (x *IVF) shouldRetrain() bool {
	if x.n < 2*x.trainedAt {
		return false
	}
	auto := x.fixed == 0 && x.opts.NList == 0
	return auto || x.trainedAt < x.sampleCap(x.wantNList(x.trainedAt))
}

// Remove drops slot from its cell.
func (x *IVF) Remove(slot uint32) {
	if int(slot) >= len(x.present) || !x.present[slot] {
		return
	}
	x.present[slot] = false
	x.n--
	l := x.listOf[slot]
	if l < 0 {
		return
	}
	lst := &x.lists[l]
	p := int(x.posOf[slot])
	last := len(lst.slots) - 1
	if p != last {
		moved := lst.slots[last]
		lst.slots[p] = moved
		copy(lst.codes[p*x.codeSize:(p+1)*x.codeSize], lst.codes[last*x.codeSize:(last+1)*x.codeSize])
		x.posOf[moved] = int32(p)
	}
	lst.slots = lst.slots[:last]
	lst.codes = lst.codes[:last*x.codeSize]
	x.listOf[slot], x.posOf[slot] = -1, -1
}

func (x *IVF) appendTo(l int, slot uint32, code []byte) {
	lst := &x.lists[l]
	x.listOf[slot], x.posOf[slot] = int32(l), int32(len(lst.slots))
	lst.slots = append(lst.slots, slot)
	lst.codes = append(lst.codes, code...)
}

// coarse returns the best cell for v.
func (x *IVF) coarse(v []float32) int {
	if x.nlist == 1 {
		return 0
	}
	return nearest(v, x.cent, x.dim, x.centNorms)
}

// encode assigns v to a cell and writes its code; returns the cell.
func (x *IVF) encode(v []float32, code []byte) int {
	l := x.coarse(v)
	switch x.opts.Quant {
	case QuantSQ8:
		x.sq.encode(v, code)
	case QuantPQ:
		r := make([]float32, x.dim)
		c := x.cent[l*x.dim : (l+1)*x.dim]
		for j := range r {
			r[j] = v[j] - c[j]
		}
		x.pq.encode(r, code)
	}
	return l
}

// train fits centroids (and the quantizer) on a sample of the live records and
// re-assigns all of them.
func (x *IVF) train() {
	slots := make([]uint32, 0, x.n)
	for s, ok := range x.present {
		if ok {
			slots = append(slots, uint32(s))
		}
	}
	nlist := min(x.wantNList(len(slots)), len(slots))
	rng := rand.New(rand.NewSource(int64(len(slots))))
	sampleSize := min(len(slots), x.sampleCap(nlist))
	perm := rng.Perm(len(slots))[:sampleSize]
	sample := make([]float32, sampleSize*x.dim)
	for i, p := range perm {
		copy(sample[i*x.dim:], x.src.Vector(slots[p]))
	}

	x.nlist = nlist
	if nlist == 1 {
		x.cent = make([]float32, x.dim)
		if x.opts.Quant == QuantPQ { // residuals from the mean
			for i := 0; i < sampleSize; i++ {
				for j := 0; j < x.dim; j++ {
					x.cent[j] += sample[i*x.dim+j] / float32(sampleSize)
				}
			}
		}
	} else {
		km := min(sampleSize, maxTrainPerCell*nlist) // the sample is a random permutation: any prefix is random
		x.cent = kmeans(sample[:km*x.dim], x.dim, nlist, kmeansIters, x.metric != L2, 1)
	}
	x.centNorms = nil
	if x.metric == L2 && nlist > 1 {
		x.centNorms = make([]float32, nlist)
		for c := 0; c < nlist; c++ {
			v := x.cent[c*x.dim : (c+1)*x.dim]
			x.centNorms[c] = vecmath.Dot(v, v)
		}
	}

	x.codeSize = 0
	switch x.opts.Quant {
	case QuantSQ8:
		x.sq = trainSQ(sample, x.dim)
		x.codeSize = x.dim
	case QuantPQ:
		m := orDefault(x.opts.PQM, autoPQM(x.dim))
		assign := make([]int32, sampleSize)
		if nlist > 1 {
			assignNearest(sample, x.dim, x.cent, x.metric != L2, assign)
		}
		res := make([]float32, len(sample))
		for i := 0; i < sampleSize; i++ {
			c := x.cent[int(assign[i])*x.dim:]
			for j := 0; j < x.dim; j++ {
				res[i*x.dim+j] = sample[i*x.dim+j] - c[j]
			}
		}
		x.pq = trainPQ(res, x.dim, m, 7)
		x.codeSize = m
	}

	// Assign and encode all records in parallel, then fill the cells in slot order.
	cells := make([]int32, len(slots))
	codes := make([]byte, len(slots)*x.codeSize)
	parallelFor(len(slots), func(lo, hi int) {
		for i := lo; i < hi; i++ {
			cells[i] = int32(x.encode(x.src.Vector(slots[i]), codes[i*x.codeSize:(i+1)*x.codeSize]))
		}
	})
	x.lists = make([]ivfList, nlist)
	for i, s := range slots {
		x.appendTo(int(cells[i]), s, codes[i*x.codeSize:(i+1)*x.codeSize])
	}
	x.trained, x.trainedAt = true, len(slots)
}

func (x *IVF) nprobe(p SearchParams) int {
	np := p.NProbe
	if np == 0 {
		np = x.opts.NProbe
	}
	if np == 0 {
		np = (x.nlist + 15) / 16 // auto: 1/16 of the cells
	}
	return max(1, min(np, x.nlist))
}

// Search scans the NPROBE best cells. With a quantizer the best K×RERANK
// candidates are re-scored with full vectors, so returned scores are exact.
// If fewer than k accepted records are found (selective filter, small cells),
// it falls back to an exact scan.
func (x *IVF) Search(query []float32, k int, accept func(uint32) bool, p SearchParams) []Hit {
	if k <= 0 {
		return nil
	}
	if !x.trained {
		return finalize(x.metric, exactSearch(x.src, x.score, query, k, accept))
	}

	nprobe := x.nprobe(p)
	probes := make([]int, 0, nprobe)
	if x.nlist == 1 {
		probes = append(probes, 0)
	} else {
		top := newTopK(nprobe)
		for c := 0; c < x.nlist; c++ {
			s := vecmath.Dot(query, x.cent[c*x.dim:(c+1)*x.dim])
			if x.centNorms != nil {
				s = 2*s - x.centNorms[c]
			}
			top.push(Hit{Slot: uint32(c), Score: s})
		}
		for _, h := range top.sorted() {
			probes = append(probes, int(h.Slot))
		}
	}

	cand := k
	if x.opts.Quant != QuantNone {
		cand = k * max(1, orDefault(p.Rerank, orDefault(x.opts.Rerank, DefaultRerank)))
	}
	top := newTopK(cand)
	switch x.opts.Quant {
	case QuantNone:
		for _, l := range probes {
			for _, s := range x.lists[l].slots {
				if accept == nil || accept(s) {
					top.push(Hit{Slot: s, Score: x.score(query, x.src.Vector(s))})
				}
			}
		}
	case QuantSQ8:
		q := x.sq.prepare(query, x.metric)
		for _, l := range probes {
			lst := &x.lists[l]
			for i, s := range lst.slots {
				if accept == nil || accept(s) {
					top.push(Hit{Slot: s, Score: q.score(lst.codes[i*x.codeSize:])})
				}
			}
		}
	case QuantPQ:
		x.scanPQ(query, probes, accept, top)
	}
	hits := top.sorted()
	if x.opts.Quant != QuantNone {
		hits = rescore(x.src, x.score, query, hits, k)
	}
	if len(hits) < k && len(hits) < x.n {
		return finalize(x.metric, exactSearch(x.src, x.score, query, k, accept))
	}
	return finalize(x.metric, hits)
}

// scanPQ scores residual PQ codes. For dot metrics q·x = q·c + q·r̂, so one
// table serves all cells; for l2 the table is rebuilt per cell on q − c.
func (x *IVF) scanPQ(query []float32, probes []int, accept func(uint32) bool, top *topK) {
	m := x.codeSize
	t := make([]float32, m*pqK)
	l2 := x.metric == L2
	if !l2 {
		x.pq.table(query, false, t)
	}
	r := make([]float32, x.dim)
	for _, l := range probes {
		lst := &x.lists[l]
		if len(lst.slots) == 0 {
			continue
		}
		c := x.cent[l*x.dim : (l+1)*x.dim]
		var base float32
		if l2 {
			for j := range r {
				r[j] = query[j] - c[j]
			}
			x.pq.table(r, true, t)
		} else {
			base = vecmath.Dot(query, c)
		}
		for i, s := range lst.slots {
			if accept != nil && !accept(s) {
				continue
			}
			d := adc(t, lst.codes[i*m:(i+1)*m])
			if l2 {
				d = -d
			} else {
				d += base
			}
			top.push(Hit{Slot: s, Score: d})
		}
	}
}

// Stats implements Index.
func (x *IVF) Stats() Stats {
	b := int64(len(x.cent)+len(x.centNorms))*4 + int64(len(x.present))*9
	for _, l := range x.lists {
		b += int64(cap(l.slots))*4 + int64(cap(l.codes))
	}
	if x.pq != nil {
		b += int64(len(x.pq.cent)) * 4
	}
	if x.sq != nil {
		b += int64(x.dim) * 8
	}
	state := fmt.Sprintf("untrained (exact until %d records)", x.trainMin())
	if x.trained {
		state = fmt.Sprintf("trained on %d records, nlist=%d", x.trainedAt, x.nlist)
	}
	return Stats{State: state, Bytes: b}
}
