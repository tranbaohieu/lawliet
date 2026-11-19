package index

import (
	"fmt"
	"strings"
)

// Kinds of index (VCREATE ... INDEX <kind>).
const (
	KindFlat = "flat" // exact scan; with QUANT it scans compressed codes
	KindHNSW = "hnsw" // graph (Malkov & Yashunin), like Faiss IndexHNSWFlat
	KindIVF  = "ivf"  // inverted lists over k-means cells, like Faiss IndexIVF*
)

// Quant is the vector encoding scanned by flat and ivf indexes.
type Quant uint8

const (
	QuantNone Quant = iota // full float32 vectors
	QuantSQ8               // 8-bit scalar quantizer, 1 byte per dimension
	QuantPQ                // product quantizer, PQ_M bytes per vector (residuals for ivf)
)

func (q Quant) String() string {
	switch q {
	case QuantNone:
		return "none"
	case QuantSQ8:
		return "sq8"
	case QuantPQ:
		return "pq"
	}
	return fmt.Sprintf("quant(%d)", uint8(q))
}

// ParseQuant parses "none", "sq8" or "pq" (case-insensitive).
func ParseQuant(s string) (Quant, error) {
	switch strings.ToLower(s) {
	case "none":
		return QuantNone, nil
	case "sq8":
		return QuantSQ8, nil
	case "pq":
		return QuantPQ, nil
	}
	return 0, fmt.Errorf("unknown QUANT '%s' (expected none, sq8 or pq)", s)
}

// Defaults (Faiss defaults in parentheses where they differ).
const (
	DefaultM              = 16
	DefaultEfConstruction = 100 // (40)
	DefaultEfSearch       = 64  // (16)
	DefaultRerank         = 1
	MaxRerank             = 100
	MaxNList              = 65536
)

// Options configures an index at creation. Zero values mean "default";
// NList and NProbe 0 mean "auto" (see IVF).
type Options struct {
	// hnsw
	M              int
	EfConstruction int
	EfSearch       int
	// ivf
	NList  int
	NProbe int
	// flat and ivf
	Quant  Quant
	PQM    int // PQ sub-quantizers (bytes per code); 0 = auto
	Rerank int // candidates re-scored with full vectors = K × Rerank; 0 = default (1)
}

// SearchParams overrides search-time options for one query (VSEARCH EF / NPROBE / RERANK).
// Zero means "use the index setting".
type SearchParams struct {
	EfSearch int
	NProbe   int
	Rerank   int
}

// Validate checks opts for the given kind and dimension.
func Validate(kind string, dim int, o Options) error {
	set := func(name string, v int) error {
		if v != 0 {
			return fmt.Errorf("%s does not apply to INDEX %s", name, kind)
		}
		return nil
	}
	check := func(errs ...error) error {
		for _, e := range errs {
			if e != nil {
				return e
			}
		}
		return nil
	}
	if o.Quant > QuantPQ { // VCREATE parses names; this catches snapshots and other callers
		return fmt.Errorf("unknown QUANT %d", uint8(o.Quant))
	}
	switch kind {
	case KindFlat:
		if err := check(set("M", o.M), set("EF_CONSTRUCTION", o.EfConstruction), set("EF_SEARCH", o.EfSearch),
			set("NLIST", o.NList), set("NPROBE", o.NProbe)); err != nil {
			return err
		}
	case KindHNSW:
		if err := check(set("NLIST", o.NList), set("NPROBE", o.NProbe), set("PQ_M", o.PQM),
			set("RERANK", o.Rerank)); err != nil {
			return err
		}
		if o.Quant != QuantNone {
			return fmt.Errorf("QUANT does not apply to INDEX hnsw")
		}
		if o.M != 0 && (o.M < 2 || o.M > 128) {
			return fmt.Errorf("M must be between 2 and 128")
		}
		if o.EfConstruction < 0 || o.EfConstruction > 4096 || o.EfSearch < 0 || o.EfSearch > 4096 {
			return fmt.Errorf("EF_CONSTRUCTION and EF_SEARCH must be between 1 and 4096")
		}
	case KindIVF:
		if err := check(set("M", o.M), set("EF_CONSTRUCTION", o.EfConstruction), set("EF_SEARCH", o.EfSearch)); err != nil {
			return err
		}
		if o.NList < 0 || o.NList > MaxNList {
			return fmt.Errorf("NLIST must be between 1 and %d", MaxNList)
		}
		if o.NProbe < 0 || o.NProbe > MaxNList {
			return fmt.Errorf("NPROBE must be between 1 and %d", MaxNList)
		}
	default:
		return fmt.Errorf("unknown index '%s' (expected flat, hnsw or ivf)", kind)
	}
	if o.Quant != QuantPQ && o.PQM != 0 {
		return fmt.Errorf("PQ_M requires QUANT pq")
	}
	if o.Quant == QuantPQ {
		m := o.PQM
		if m == 0 {
			m = autoPQM(dim)
		}
		if m < 1 || dim%m != 0 {
			return fmt.Errorf("PQ_M must divide DIM (%d)", dim)
		}
	}
	if o.Rerank < 0 || o.Rerank > MaxRerank {
		return fmt.Errorf("RERANK must be between 1 and %d", MaxRerank)
	}
	if o.Quant == QuantNone && o.Rerank != 0 {
		return fmt.Errorf("RERANK requires QUANT sq8 or pq")
	}
	return nil
}

// Describe renders the effective options, e.g. "M=16 EF_CONSTRUCTION=100 EF_SEARCH=64".
func Describe(kind string, dim int, o Options) string {
	auto := func(v int) string {
		if v == 0 {
			return "auto"
		}
		return fmt.Sprint(v)
	}
	var parts []string
	switch kind {
	case KindHNSW:
		parts = append(parts, fmt.Sprintf("M=%d EF_CONSTRUCTION=%d EF_SEARCH=%d",
			orDefault(o.M, DefaultM), orDefault(o.EfConstruction, DefaultEfConstruction), orDefault(o.EfSearch, DefaultEfSearch)))
	case KindIVF:
		parts = append(parts, "NLIST="+auto(o.NList), "NPROBE="+auto(o.NProbe))
	}
	if o.Quant != QuantNone {
		parts = append(parts, "QUANT="+o.Quant.String())
		if o.Quant == QuantPQ {
			parts = append(parts, fmt.Sprintf("PQ_M=%d", orDefault(o.PQM, autoPQM(dim))))
		}
		parts = append(parts, fmt.Sprintf("RERANK=%d", orDefault(o.Rerank, DefaultRerank)))
	}
	return strings.Join(parts, " ")
}

func orDefault(v, def int) int {
	if v == 0 {
		return def
	}
	return v
}

// autoPQM is the largest divisor of dim not above dim/16 (64 bytes for dim 1024, as Faiss "PQ64").
func autoPQM(dim int) int {
	m := dim / 16
	for m > 1 && dim%m != 0 {
		m--
	}
	if m < 1 {
		m = 1
	}
	return m
}

// New returns the index of the given kind over src.
func New(src Source, metric Metric, dim int, kind string, o Options) (Index, error) {
	if err := Validate(kind, dim, o); err != nil {
		return nil, err
	}
	switch kind {
	case KindHNSW:
		return NewHNSW(src, metric, o), nil
	case KindIVF:
		return NewIVF(src, metric, dim, o, 0), nil
	}
	if o.Quant == QuantNone {
		return NewFlat(src, metric), nil
	}
	return NewIVF(src, metric, dim, o, 1), nil // one list: a flat scan over codes
}
