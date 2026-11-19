//go:build !(goexperiment.simd && amd64)

package vecmath

// AVX2 is false: this build has no SIMD path (see simd_amd64.go).
const AVX2 = false

func dotSIMD(a, b []float32) float32  { return dotScalar(a, b) }
func l2SqSIMD(a, b []float32) float32 { return l2SqScalar(a, b) }
