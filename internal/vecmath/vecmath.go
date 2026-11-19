// Package vecmath provides float32 vector math used by indexes.
//
// Dot and L2Sq use AVX2+FMA when the binary is built with GOEXPERIMENT=simd
// on amd64 and the CPU supports it (simd_amd64.go); otherwise the scalar loops
// below, unrolled by 4 with independent accumulators so the CPU can pipeline
// the additions. Results can differ from the scalar version in the last bits
// (different summation order).
package vecmath

import "math"

// simdMin is the shortest vector worth the SIMD path.
const simdMin = 32

// Dot returns the dot product of a and b. b must be at least as long as a.
func Dot(a, b []float32) float32 {
	if AVX2 && len(a) >= simdMin {
		return dotSIMD(a, b)
	}
	return dotScalar(a, b)
}

// L2Sq returns the squared Euclidean distance between a and b.
func L2Sq(a, b []float32) float32 {
	if AVX2 && len(a) >= simdMin {
		return l2SqSIMD(a, b)
	}
	return l2SqScalar(a, b)
}

// Accel names the vector instructions in use: "avx2+fma" or "none".
func Accel() string {
	if AVX2 {
		return "avx2+fma"
	}
	return "none"
}

func dotScalar(a, b []float32) float32 {
	b = b[:len(a)] // bounds-check elimination hint
	var s0, s1, s2, s3 float32
	n := len(a) &^ 3
	for i := 0; i < n; i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for i := n; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return s0 + s1 + s2 + s3
}

func l2SqScalar(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float32
	n := len(a) &^ 3
	for i := 0; i < n; i += 4 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
	}
	for i := n; i < len(a); i++ {
		d := a[i] - b[i]
		s0 += d * d
	}
	return s0 + s1 + s2 + s3
}

// Norm returns the Euclidean length of v.
func Norm(v []float32) float32 {
	return float32(math.Sqrt(float64(Dot(v, v))))
}

// Normalize scales v in place to unit length.
// It returns false (and leaves v unchanged) if v has zero or non-finite length.
func Normalize(v []float32) bool {
	n := Norm(v)
	if n == 0 || math.IsInf(float64(n), 0) || math.IsNaN(float64(n)) {
		return false
	}
	inv := 1 / n
	for i := range v {
		v[i] *= inv
	}
	return true
}

// IsFinite reports whether every element of v is a finite number.
func IsFinite(v []float32) bool {
	for _, x := range v {
		if math.IsInf(float64(x), 0) || math.IsNaN(float64(x)) {
			return false
		}
	}
	return true
}
