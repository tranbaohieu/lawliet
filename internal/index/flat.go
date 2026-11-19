package index

// Flat is an exact (brute-force) index: every accepted vector is scored.
type Flat struct {
	src    Source
	metric Metric
}

// NewFlat returns a flat index over src.
func NewFlat(src Source, metric Metric) *Flat {
	return &Flat{src: src, metric: metric}
}

// Add is a no-op: the flat index reads vectors directly from the source.
func (f *Flat) Add(uint32, []float32) {}

// Remove is a no-op: removed slots are rejected by the accept function.
func (f *Flat) Remove(uint32) {}

// Search scores all accepted slots and keeps the best k with a min-heap.
func (f *Flat) Search(query []float32, k int, accept func(slot uint32) bool, _ SearchParams) []Hit {
	if k <= 0 {
		return nil
	}
	return finalize(f.metric, exactSearch(f.src, rawScore(f.metric), query, k, accept))
}

// Stats implements Index.
func (f *Flat) Stats() Stats { return Stats{State: "exact"} }
