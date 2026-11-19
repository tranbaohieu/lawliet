package index

import (
	"math"

	"github.com/tranbaohieu/lawliet/internal/vecmath"
)

// scalarQuantizer encodes each dimension in one byte over its trained range
// [min, max] (Faiss ScalarQuantizer QT_8bit). Values outside are clamped.
type scalarQuantizer struct {
	dim   int
	vmin  []float32
	scale []float32 // (max-min)/255
}

func trainSQ(data []float32, dim int) *scalarQuantizer {
	q := &scalarQuantizer{dim: dim, vmin: make([]float32, dim), scale: make([]float32, dim)}
	vmax := make([]float32, dim)
	for j := 0; j < dim; j++ {
		q.vmin[j], vmax[j] = float32(math.Inf(1)), float32(math.Inf(-1))
	}
	for i := 0; i+dim <= len(data); i += dim {
		for j, x := range data[i : i+dim] {
			q.vmin[j] = min(q.vmin[j], x)
			vmax[j] = max(vmax[j], x)
		}
	}
	for j := range q.scale {
		q.scale[j] = (vmax[j] - q.vmin[j]) / 255
	}
	return q
}

func (q *scalarQuantizer) encode(x []float32, code []byte) {
	for j, v := range x {
		if q.scale[j] == 0 {
			code[j] = 0
			continue
		}
		c := math.Round(float64((v - q.vmin[j]) / q.scale[j]))
		code[j] = byte(max(0, min(255, c)))
	}
}

// sqQuery precomputes a query for scoring codes: higher is better.
type sqQuery struct {
	l2  bool
	c0  float32   // dot: q·min
	w   []float32 // dot: q_j·scale_j ; l2: scale_j
	a   []float32 // l2: q_j - min_j
	dim int
}

func (q *scalarQuantizer) prepare(query []float32, metric Metric) *sqQuery {
	p := &sqQuery{l2: metric == L2, dim: q.dim, w: make([]float32, q.dim)}
	if p.l2 {
		p.a = make([]float32, q.dim)
		for j := range query {
			p.a[j] = query[j] - q.vmin[j]
			p.w[j] = q.scale[j]
		}
		return p
	}
	p.c0 = vecmath.Dot(query, q.vmin)
	for j := range query {
		p.w[j] = query[j] * q.scale[j]
	}
	return p
}

func (p *sqQuery) score(code []byte) float32 {
	if vecmath.AVX2 && p.dim >= 16 {
		if p.l2 {
			return -sqL2SIMD(p.a, p.w, code[:p.dim])
		}
		return p.c0 + sqDotSIMD(p.w, code[:p.dim])
	}
	return p.scoreScalar(code)
}

func (p *sqQuery) scoreScalar(code []byte) float32 {
	code = code[:p.dim]
	w := p.w[:p.dim]
	if p.l2 {
		a := p.a[:p.dim]
		var s0, s1 float32
		n := p.dim &^ 1
		for j := 0; j < n; j += 2 {
			d0 := a[j] - w[j]*float32(code[j])
			d1 := a[j+1] - w[j+1]*float32(code[j+1])
			s0 += d0 * d0
			s1 += d1 * d1
		}
		for j := n; j < p.dim; j++ {
			d := a[j] - w[j]*float32(code[j])
			s0 += d * d
		}
		return -(s0 + s1)
	}
	var s0, s1, s2, s3 float32
	n := p.dim &^ 3
	for j := 0; j < n; j += 4 {
		s0 += w[j] * float32(code[j])
		s1 += w[j+1] * float32(code[j+1])
		s2 += w[j+2] * float32(code[j+2])
		s3 += w[j+3] * float32(code[j+3])
	}
	for j := n; j < p.dim; j++ {
		s0 += w[j] * float32(code[j])
	}
	return p.c0 + s0 + s1 + s2 + s3
}

// productQuantizer splits vectors into m sub-vectors and encodes each as the
// index of its nearest of 256 sub-centroids (Faiss ProductQuantizer, 8 bits).
type productQuantizer struct {
	dim, m, dsub int
	cent         []float32 // m × 256 × dsub
}

const pqK = 256

func trainPQ(data []float32, dim, m int, seed int64) *productQuantizer {
	n := len(data) / dim
	dsub := dim / m
	q := &productQuantizer{dim: dim, m: m, dsub: dsub, cent: make([]float32, m*pqK*dsub)}
	sub := make([]float32, n*dsub)
	for s := 0; s < m; s++ {
		for i := 0; i < n; i++ {
			copy(sub[i*dsub:(i+1)*dsub], data[i*dim+s*dsub:i*dim+(s+1)*dsub])
		}
		c := kmeans(sub, dsub, pqK, 10, false, seed+int64(s))
		copy(q.cent[s*pqK*dsub:], c) // fewer than 256 rows only if n < 256: rest stays zero
	}
	return q
}

func (q *productQuantizer) encode(x []float32, code []byte) {
	for s := 0; s < q.m; s++ {
		xs := x[s*q.dsub : (s+1)*q.dsub]
		cs := q.cent[s*pqK*q.dsub : (s+1)*pqK*q.dsub]
		best, bestD := 0, float32(math.Inf(1))
		for c := 0; c < pqK; c++ {
			if d := vecmath.L2Sq(xs, cs[c*q.dsub:(c+1)*q.dsub]); d < bestD {
				best, bestD = c, d
			}
		}
		code[s] = byte(best)
	}
}

// table fills t (m × 256) for asymmetric distance computation: dot products of
// the query sub-vectors with each sub-centroid (l2=false), or squared distances (l2=true).
func (q *productQuantizer) table(x []float32, l2 bool, t []float32) {
	for s := 0; s < q.m; s++ {
		xs := x[s*q.dsub : (s+1)*q.dsub]
		cs := q.cent[s*pqK*q.dsub : (s+1)*pqK*q.dsub]
		ts := t[s*pqK : (s+1)*pqK]
		for c := 0; c < pqK; c++ {
			if l2 {
				ts[c] = vecmath.L2Sq(xs, cs[c*q.dsub:(c+1)*q.dsub])
			} else {
				ts[c] = vecmath.Dot(xs, cs[c*q.dsub:(c+1)*q.dsub])
			}
		}
	}
}

// adc sums table entries for a code.
func adc(t []float32, code []byte) float32 {
	var s0, s1, s2, s3 float32
	m := len(code)
	n := m &^ 3
	for s := 0; s < n; s += 4 {
		s0 += t[s*pqK+int(code[s])]
		s1 += t[(s+1)*pqK+int(code[s+1])]
		s2 += t[(s+2)*pqK+int(code[s+2])]
		s3 += t[(s+3)*pqK+int(code[s+3])]
	}
	for s := n; s < m; s++ {
		s0 += t[s*pqK+int(code[s])]
	}
	return s0 + s1 + s2 + s3
}
