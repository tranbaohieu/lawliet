package index

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"sort"
	"sync"
)

// HNSW is a Hierarchical Navigable Small World graph (Malkov & Yashunin 2016),
// like Faiss IndexHNSWFlat: full vectors are read from the collection, the
// graph stores neighbor lists only.
//
// Each node gets a random top level (geometric, mL = 1/ln M). Upper levels keep
// up to M neighbors, level 0 up to 2M; neighbors are chosen with the
// diversity heuristic (keep a candidate only if it is closer to the new node
// than to every neighbor already kept).
//
// Remove takes the node out of the graph and repairs it: every neighbor that
// linked to it re-selects its links (heuristic) among its other links plus the
// removed node's links, so the graph keeps its connectivity under churn (TTL,
// eviction, updates). Links to the node from nodes it did not link to are left
// dangling and skipped while searching (and lead to the new vector if the slot
// is reused). Measured: without repair, recall@10 fell 0.979 → 0.954 after
// replacing 30% of 100k records three times (docs/benchmarks/ann-2026-11-16).
//
// Filters are applied while searching level 0 (non-matching nodes are traversed
// but not returned). If fewer than k matches are found, the search falls back
// to an exact scan, so selective filters stay correct.
type HNSW struct {
	src    Source
	metric Metric
	score  func(a, b []float32) float32
	m, m0  int
	efC    int
	efS    int
	mL     float64
	rng    *rand.Rand

	nodes    []hnswNode // indexed by slot
	entry    int64      // -1 = empty
	maxLevel int
	inGraph  int
	visited  sync.Pool
}

type hnswNode struct {
	level int8 // -1 = not in the graph
	links [][]uint32
}

// NewHNSW returns an empty graph.
func NewHNSW(src Source, metric Metric, o Options) *HNSW {
	m := orDefault(o.M, DefaultM)
	return &HNSW{
		src: src, metric: metric, score: rawScore(metric),
		m: m, m0: 2 * m,
		efC:   orDefault(o.EfConstruction, DefaultEfConstruction),
		efS:   orDefault(o.EfSearch, DefaultEfSearch),
		mL:    1 / math.Log(float64(m)),
		rng:   rand.New(rand.NewSource(42)),
		entry: -1,
	}
}

// cand is a node and its distance to the current query (lower is closer).
type cand struct {
	d float32
	s uint32
}

func (h *HNSW) dist(q []float32, s uint32) float32 { return -h.score(q, h.src.Vector(s)) }

func (h *HNSW) maxLinks(level int) int {
	if level == 0 {
		return h.m0
	}
	return h.m
}

func (h *HNSW) randomLevel() int {
	l := int(-math.Log(1-h.rng.Float64()) * h.mL)
	return min(l, 16)
}

// Add inserts slot into the graph (re-inserts it if present).
func (h *HNSW) Add(slot uint32, vec []float32) {
	for int(slot) >= len(h.nodes) {
		h.nodes = append(h.nodes, hnswNode{level: -1})
	}
	if h.nodes[slot].level >= 0 {
		h.Remove(slot)
	}
	level := h.randomLevel()
	node := &h.nodes[slot]
	node.level = int8(level)
	node.links = make([][]uint32, level+1)
	h.inGraph++

	if h.entry < 0 {
		h.entry, h.maxLevel = int64(slot), level
		return
	}
	vis := h.getVisited()
	defer h.visited.Put(vis)

	ep := cand{h.dist(vec, uint32(h.entry)), uint32(h.entry)}
	for l := h.maxLevel; l > level; l-- {
		ep = h.greedy(vec, ep, l)
	}
	for l := min(level, h.maxLevel); l >= 0; l-- {
		w := h.searchLayer(vec, ep, h.efC, l, nil, vis, slot)
		nbrs := h.selectNeighbors(w, h.maxLinks(l))
		links := make([]uint32, len(nbrs), h.maxLinks(l))
		for i, c := range nbrs {
			links[i] = c.s
		}
		h.nodes[slot].links[l] = links
		for _, c := range nbrs {
			h.connect(c.s, slot, c.d, l)
		}
		if len(w) > 0 {
			ep = w[0]
		}
	}
	if level > h.maxLevel {
		h.entry, h.maxLevel = int64(slot), level
	}
}

// Remove takes slot out of the graph and repairs its neighbors (see type doc).
func (h *HNSW) Remove(slot uint32) {
	if int(slot) >= len(h.nodes) || h.nodes[slot].level < 0 {
		return
	}
	old := h.nodes[slot]
	h.nodes[slot] = hnswNode{level: -1}
	h.inGraph--
	for l, links := range old.links {
		for _, n := range links {
			h.repair(n, slot, links, l)
		}
	}
	if h.entry == int64(slot) {
		h.pickEntry()
	}
}

// repair replaces removed in n's links at level l, choosing among n's other
// links and extra (the removed node's links) with the heuristic.
func (h *HNSW) repair(n, removed uint32, extra []uint32, l int) {
	node := &h.nodes[n]
	if int(node.level) < l {
		return
	}
	links := node.links[l]
	if !slices.Contains(links, removed) {
		return
	}
	vn := h.src.Vector(n)
	cs := make([]cand, 0, len(links)+len(extra))
	add := func(x uint32) {
		if x == n || x == removed || !h.inLevel(x, l) {
			return
		}
		for _, c := range cs {
			if c.s == x {
				return
			}
		}
		cs = append(cs, cand{h.dist(vn, x), x})
	}
	for _, x := range links {
		add(x)
	}
	for _, x := range extra {
		add(x)
	}
	sort.Slice(cs, func(i, j int) bool { return cs[i].d < cs[j].d })
	kept := h.selectNeighbors(cs, h.maxLinks(l))
	links = links[:0]
	for _, c := range kept {
		links = append(links, c.s)
	}
	node.links[l] = links
}

// pickEntry makes the highest node the entry point (after the entry was removed).
func (h *HNSW) pickEntry() {
	h.entry, h.maxLevel = -1, 0
	for s, n := range h.nodes {
		if n.level >= 0 && (h.entry < 0 || int(n.level) > h.maxLevel) {
			h.entry, h.maxLevel = int64(s), int(n.level)
		}
	}
}

// inLevel reports whether slot x is in the graph at level l (false once removed).
func (h *HNSW) inLevel(x uint32, l int) bool { return int(h.nodes[x].level) >= l }

// connect adds a link from n to s at level l, shrinking n's list with the heuristic when full.
func (h *HNSW) connect(n, s uint32, d float32, l int) {
	node := &h.nodes[n]
	if int(node.level) < l {
		return
	}
	if slices.Contains(node.links[l], s) {
		return
	}
	// Drop links to removed nodes first. Remove repairs only the lists of the removed
	// node's own neighbours, so a one-way link to it stays until its list is touched
	// here; kept as a candidate it would also block live ones in the diversity heuristic.
	links := node.links[l][:0]
	for _, x := range node.links[l] {
		if h.inLevel(x, l) {
			links = append(links, x)
		}
	}
	if len(links) < h.maxLinks(l) {
		node.links[l] = append(links, s)
		return
	}
	vn := h.src.Vector(n)
	cs := make([]cand, 0, len(links)+1)
	cs = append(cs, cand{d, s})
	for _, x := range links {
		cs = append(cs, cand{h.dist(vn, x), x})
	}
	sort.Slice(cs, func(i, j int) bool { return cs[i].d < cs[j].d })
	kept := h.selectNeighbors(cs, h.maxLinks(l))
	links = links[:0]
	for _, c := range kept {
		links = append(links, c.s)
	}
	node.links[l] = links
}

// selectNeighbors applies the diversity heuristic to candidates sorted by distance.
func (h *HNSW) selectNeighbors(cs []cand, m int) []cand {
	if len(cs) <= m {
		return cs
	}
	out := make([]cand, 0, m)
	for _, c := range cs {
		if len(out) >= m {
			break
		}
		vc := h.src.Vector(c.s)
		good := true
		for _, r := range out {
			if h.dist(vc, r.s) < c.d {
				good = false
				break
			}
		}
		if good {
			out = append(out, c)
		}
	}
	return out
}

// greedy walks level l towards q and returns the closest node found.
func (h *HNSW) greedy(q []float32, ep cand, l int) cand {
	for changed := true; changed; {
		changed = false
		for _, nb := range h.nodes[ep.s].links[l] {
			if int(h.nodes[nb].level) < l {
				continue // stale link to a re-inserted node
			}
			if d := h.dist(q, nb); d < ep.d {
				ep, changed = cand{d, nb}, true
			}
		}
	}
	return ep
}

// searchLayer is the beam search of the paper (ef best results). accept (level
// 0 only) filters results without stopping the traversal. skip is excluded
// from results (the node being inserted).
func (h *HNSW) searchLayer(q []float32, ep cand, ef, l int, accept func(uint32) bool, vis *visited, skip uint32) []cand {
	vis.reset(len(h.nodes))
	vis.mark(ep.s)
	cands := minHeap{ep}
	var res maxHeap
	if ep.s != skip && (accept == nil || accept(ep.s)) {
		res.push(ep)
	}
	for len(cands) > 0 {
		c := cands.pop()
		if len(res) >= ef && c.d > res[0].d {
			break
		}
		for _, nb := range h.nodes[c.s].links[l] {
			if vis.marked(nb) || int(h.nodes[nb].level) < l {
				continue
			}
			vis.mark(nb)
			d := h.dist(q, nb)
			if len(res) < ef || d < res[0].d {
				cands.push(cand{d, nb})
				if nb != skip && (accept == nil || accept(nb)) {
					res.push(cand{d, nb})
					if len(res) > ef {
						res.pop()
					}
				}
			}
		}
	}
	out := []cand(res)
	sort.Slice(out, func(i, j int) bool { return out[i].d < out[j].d })
	return out
}

// Search returns the k best accepted nodes.
func (h *HNSW) Search(query []float32, k int, accept func(uint32) bool, p SearchParams) []Hit {
	if k <= 0 || h.entry < 0 {
		return nil
	}
	ef := max(orDefault(p.EfSearch, h.efS), k)
	vis := h.getVisited()
	defer h.visited.Put(vis)

	ep := cand{h.dist(query, uint32(h.entry)), uint32(h.entry)}
	for l := h.maxLevel; l > 0; l-- {
		ep = h.greedy(query, ep, l)
	}
	res := h.searchLayer(query, ep, ef, 0, accept, vis, math.MaxUint32)
	if len(res) < k && len(res) < h.inGraph {
		return finalize(h.metric, exactSearch(h.src, h.score, query, k, accept))
	}
	hits := make([]Hit, 0, min(k, len(res)))
	for _, c := range res[:min(k, len(res))] {
		hits = append(hits, Hit{Slot: c.s, Score: -c.d})
	}
	sort.SliceStable(hits, func(i, j int) bool { return worse(hits[j], hits[i]) })
	return finalize(h.metric, hits)
}

// Stats implements Index.
func (h *HNSW) Stats() Stats {
	b := int64(len(h.nodes)) * 32
	edges := 0
	for _, n := range h.nodes {
		for _, l := range n.links {
			b += int64(cap(l))*4 + 24
			edges += len(l)
		}
	}
	return Stats{State: fmt.Sprintf("%d nodes, %d levels, %d links", h.inGraph, h.maxLevel+1, edges), Bytes: b}
}

// visited is an epoch-stamped set of slots, reused across queries via a pool.
type visited struct {
	epoch uint16
	marks []uint16
}

func (h *HNSW) getVisited() *visited {
	if v, ok := h.visited.Get().(*visited); ok {
		return v
	}
	return &visited{}
}

func (v *visited) reset(n int) {
	if len(v.marks) < n {
		v.marks = make([]uint16, n+n/4)
		v.epoch = 0
	}
	v.epoch++
	if v.epoch == 0 { // wrapped: clear
		clear(v.marks)
		v.epoch = 1
	}
}

func (v *visited) mark(s uint32)        { v.marks[s] = v.epoch }
func (v *visited) marked(s uint32) bool { return v.marks[s] == v.epoch }

// minHeap pops the closest candidate; maxHeap keeps the farthest result at the root.
type minHeap []cand
type maxHeap []cand

func (h *minHeap) push(c cand) {
	*h = append(*h, c)
	a := *h
	for i := len(a) - 1; i > 0; {
		p := (i - 1) / 2
		if a[p].d <= a[i].d {
			break
		}
		a[p], a[i] = a[i], a[p]
		i = p
	}
}

func (h *minHeap) pop() cand {
	a := *h
	top := a[0]
	last := len(a) - 1
	a[0] = a[last]
	a = a[:last]
	for i := 0; ; {
		l, m := 2*i+1, i
		if l < len(a) && a[l].d < a[m].d {
			m = l
		}
		if r := l + 1; r < len(a) && a[r].d < a[m].d {
			m = r
		}
		if m == i {
			break
		}
		a[i], a[m] = a[m], a[i]
		i = m
	}
	*h = a
	return top
}

func (h *maxHeap) push(c cand) {
	*h = append(*h, c)
	a := *h
	for i := len(a) - 1; i > 0; {
		p := (i - 1) / 2
		if a[p].d >= a[i].d {
			break
		}
		a[p], a[i] = a[i], a[p]
		i = p
	}
}

func (h *maxHeap) pop() cand {
	a := *h
	top := a[0]
	last := len(a) - 1
	a[0] = a[last]
	a = a[:last]
	for i := 0; ; {
		l, m := 2*i+1, i
		if l < len(a) && a[l].d > a[m].d {
			m = l
		}
		if r := l + 1; r < len(a) && a[r].d > a[m].d {
			m = r
		}
		if m == i {
			break
		}
		a[i], a[m] = a[m], a[i]
		i = m
	}
	*h = a
	return top
}
