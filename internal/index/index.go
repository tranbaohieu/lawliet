// Package index defines the vector index interface and its implementations.
package index

import (
	"fmt"
	"strings"
)

// Metric is a similarity metric. Scores are always "higher is better" (DESIGN.md §4.3).
type Metric uint8

const (
	Cosine Metric = iota // cosine similarity; vectors are stored L2-normalized
	L2                   // score = -euclidean distance
	IP                   // score = dot product
)

func (m Metric) String() string {
	switch m {
	case Cosine:
		return "cosine"
	case L2:
		return "l2"
	case IP:
		return "ip"
	}
	return fmt.Sprintf("metric(%d)", uint8(m))
}

// ParseMetric parses "cosine", "l2" or "ip" (case-insensitive).
func ParseMetric(s string) (Metric, error) {
	switch strings.ToLower(s) {
	case "cosine":
		return Cosine, nil
	case "l2":
		return L2, nil
	case "ip":
		return IP, nil
	}
	return 0, fmt.Errorf("unknown metric '%s' (expected cosine, l2 or ip)", s)
}

// Hit is a search result: an internal slot and its score.
type Hit struct {
	Slot  uint32
	Score float32
}

// Source gives an index read access to stored vectors.
// Implemented by the collection; called while the collection lock is held.
type Source interface {
	NumSlots() int
	Vector(slot uint32) []float32
}

// Index finds the nearest vectors to a query.
//
// Add/Remove are called on every mutation (under the collection write lock) so
// that indexes can maintain their structure; Search may run concurrently with
// other searches (under the read lock) and must not modify shared state.
type Index interface {
	Add(slot uint32, vec []float32)
	Remove(slot uint32)
	// Search returns at most k hits, best first. accept filters slots
	// (tombstones, expired records, metadata filters); nil accepts all.
	// accept is called at most once per slot unless the index falls back to
	// an exact scan.
	Search(query []float32, k int, accept func(slot uint32) bool, p SearchParams) []Hit
	Stats() Stats
}

// Stats describes an index for VINFO.
type Stats struct {
	State string // e.g. "trained on 100000 records, nlist=316"
	Bytes int64  // memory held by the index, on top of the vectors
}
