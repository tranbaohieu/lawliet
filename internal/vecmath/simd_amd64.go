//go:build goexperiment.simd && amd64

package vecmath

import "simd/archsimd"

// AVX2 reports whether Dot/L2Sq (and index code scoring) use AVX2+FMA.
var AVX2 = archsimd.X86.AVX2() && archsimd.X86.FMA()

// The loops take one bounds check per 32-float block: loads from
// constant-index sub-arrays of a *[32]float32 need none. Without this the
// compiler emits ~20 scalar instructions per 8-wide load and AVX2 is no faster
// than the scalar loop (measured).

func dotSIMD(a, b []float32) float32 {
	n := len(a) &^ 31
	b = b[:len(a)]
	var acc0, acc1, acc2, acc3 archsimd.Float32x8
	for i := 0; i < n; i += 32 {
		x := (*[32]float32)(a[i : i+32])
		y := (*[32]float32)(b[i : i+32])
		acc0 = load8(x[0:8]).MulAdd(load8(y[0:8]), acc0)
		acc1 = load8(x[8:16]).MulAdd(load8(y[8:16]), acc1)
		acc2 = load8(x[16:24]).MulAdd(load8(y[16:24]), acc2)
		acc3 = load8(x[24:32]).MulAdd(load8(y[24:32]), acc3)
	}
	s := HSum(acc0.Add(acc1).Add(acc2.Add(acc3)))
	for i := n; i < len(a); i++ {
		s += a[i] * b[i]
	}
	return s
}

func l2SqSIMD(a, b []float32) float32 {
	n := len(a) &^ 31
	b = b[:len(a)]
	var acc0, acc1, acc2, acc3 archsimd.Float32x8
	for i := 0; i < n; i += 32 {
		x := (*[32]float32)(a[i : i+32])
		y := (*[32]float32)(b[i : i+32])
		d0 := load8(x[0:8]).Sub(load8(y[0:8]))
		d1 := load8(x[8:16]).Sub(load8(y[8:16]))
		d2 := load8(x[16:24]).Sub(load8(y[16:24]))
		d3 := load8(x[24:32]).Sub(load8(y[24:32]))
		acc0 = d0.MulAdd(d0, acc0)
		acc1 = d1.MulAdd(d1, acc1)
		acc2 = d2.MulAdd(d2, acc2)
		acc3 = d3.MulAdd(d3, acc3)
	}
	s := HSum(acc0.Add(acc1).Add(acc2.Add(acc3)))
	for i := n; i < len(a); i++ {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

func load8(s []float32) archsimd.Float32x8 { return archsimd.LoadFloat32x8Array((*[8]float32)(s)) }

// HSum adds the 8 lanes of v and ends the AVX section: it clears the upper
// halves of the ymm registers (VZEROUPPER). The compiler does not do it, and
// Go's scalar float code uses legacy SSE encodings; without the clear, every
// later SSE instruction pays the AVX→SSE transition penalty — measured: the
// index tests ran 3× slower with the SIMD path than without it.
// Call it last in every kernel, after the final vector operation.
func HSum(v archsimd.Float32x8) float32 {
	var t [8]float32
	v.StoreArray(&t)
	archsimd.ClearAVXUpperBits()
	return (t[0] + t[1]) + (t[2] + t[3]) + (t[4] + t[5]) + (t[6] + t[7])
}
