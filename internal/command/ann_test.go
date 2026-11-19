package command

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

func TestANNIndexCommands(t *testing.T) {
	c := newClient(t)
	want(t, c.Do("VCREATE", "g", "DIM", 4, "INDEX", "hnsw", "M", 8, "EF_SEARCH", 32), `+"OK"`)
	want(t, c.Do("VCREATE", "v", "DIM", 4, "INDEX", "ivf", "NLIST", 4, "QUANT", "sq8", "RERANK", 3), `+"OK"`)
	want(t, c.Do("VCREATE", "p", "DIM", 4, "INDEX", "flat", "QUANT", "pq", "PQ_M", 2), `+"OK"`)

	r := rand.New(rand.NewPCG(1, 1))
	for i := 0; i < 300; i++ {
		v := []any{r.Float64(), r.Float64(), r.Float64(), r.Float64()}
		for _, col := range []string{"g", "v", "p"} {
			args := append([]any{"VADD", col, fmt.Sprintf("d%d", i), "VALUES", 4}, v...)
			want(t, c.Do(args...), ":1")
		}
	}

	g := c.Do("VINFO", "g").Map()
	want(t, g["index"], `$"hnsw"`)
	want(t, g["index_params"], `$"M=8 EF_CONSTRUCTION=100 EF_SEARCH=32"`)
	expect(t, describe(g["index_state"])[:12], `$"300 nodes,`)
	v := c.Do("VINFO", "v").Map()
	want(t, v["index_params"], `$"NLIST=4 NPROBE=auto QUANT=sq8 RERANK=3"`)
	want(t, v["index_state"], `$"untrained (exact until 1000 records)"`)
	want(t, c.Do("VINFO", "p").Map()["index_params"], `$"QUANT=pq PQ_M=2 RERANK=1"`)

	// Same top hit as exact search for a stored vector; overrides accepted.
	q := c.Do("VGET", "g", "d7", "WITHVEC").Array[2].Str
	for _, args := range [][]any{
		{"VSEARCH", "g", "FP32", q, "K", 3, "EF", 100},
		{"VSEARCH", "v", "FP32", q, "K", 3, "NPROBE", 4, "RERANK", 2},
		{"VSEARCH", "p", "FP32", q, "K", 3},
	} {
		hits := c.Do(args...)
		if len(hits.Array) != 3 || hits.Array[0].Array[0].Str != "d7" {
			t.Fatalf("%v: %s", args[:2], describe(hits))
		}
	}
	wantErr(t, c.Do("VSEARCH", "g", "FP32", q, "EF", 0), "ERR EF must be an integer between 1 and 4096")
	wantErr(t, c.Do("VSEARCH", "g", "FP32", q, "RERANK", 101), "ERR RERANK must be an integer between 1 and 100")
}
