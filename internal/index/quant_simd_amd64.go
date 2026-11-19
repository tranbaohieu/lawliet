//go:build goexperiment.simd && amd64

package index

import (
	"simd/archsimd"

	"github.com/tranbaohieu/lawliet/internal/vecmath"
)

// sq8 codes are widened 16 at a time: uint8 → uint16 → uint32 (two halves of
// 8) → float32, then fused multiply-add with the precomputed weights.

func widen16(code []byte) (lo, hi archsimd.Float32x8) {
	c := archsimd.LoadUint8x16Array((*[16]uint8)(code)).ExtendToUint16()
	lo = c.GetLo().ExtendToUint32().AsInt32x8().ConvertToFloat32()
	hi = c.GetHi().ExtendToUint32().AsInt32x8().ConvertToFloat32()
	return lo, hi
}

// sqDotSIMD returns Σ w_j·code_j (len(code) == len(w)).
func sqDotSIMD(w []float32, code []byte) float32 {
	n := len(w) &^ 15
	code = code[:len(w)]
	var acc0, acc1 archsimd.Float32x8
	for j := 0; j < n; j += 16 {
		lo, hi := widen16(code[j : j+16])
		ww := (*[16]float32)(w[j : j+16])
		acc0 = lo.MulAdd(archsimd.LoadFloat32x8Array((*[8]float32)(ww[0:8])), acc0)
		acc1 = hi.MulAdd(archsimd.LoadFloat32x8Array((*[8]float32)(ww[8:16])), acc1)
	}
	s := vecmath.HSum(acc0.Add(acc1))
	for j := n; j < len(w); j++ {
		s += w[j] * float32(code[j])
	}
	return s
}

// sqL2SIMD returns Σ (a_j − w_j·code_j)².
func sqL2SIMD(a, w []float32, code []byte) float32 {
	n := len(w) &^ 15
	code = code[:len(w)]
	a = a[:len(w)]
	var acc0, acc1 archsimd.Float32x8
	for j := 0; j < n; j += 16 {
		lo, hi := widen16(code[j : j+16])
		ww := (*[16]float32)(w[j : j+16])
		aa := (*[16]float32)(a[j : j+16])
		d0 := archsimd.LoadFloat32x8Array((*[8]float32)(aa[0:8])).Sub(lo.Mul(archsimd.LoadFloat32x8Array((*[8]float32)(ww[0:8]))))
		d1 := archsimd.LoadFloat32x8Array((*[8]float32)(aa[8:16])).Sub(hi.Mul(archsimd.LoadFloat32x8Array((*[8]float32)(ww[8:16]))))
		acc0 = d0.MulAdd(d0, acc0)
		acc1 = d1.MulAdd(d1, acc1)
	}
	s := vecmath.HSum(acc0.Add(acc1))
	for j := n; j < len(w); j++ {
		d := a[j] - w[j]*float32(code[j])
		s += d * d
	}
	return s
}
