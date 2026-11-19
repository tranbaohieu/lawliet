//go:build !(goexperiment.simd && amd64)

package index

// Never called: vecmath.AVX2 is the constant false in this build.
func sqDotSIMD(w []float32, code []byte) float32   { panic("unreachable") }
func sqL2SIMD(a, w []float32, code []byte) float32 { panic("unreachable") }
