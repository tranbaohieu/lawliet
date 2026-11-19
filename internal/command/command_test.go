package command

import (
	"strconv"
	"strings"
	"testing"

	"github.com/tranbaohieu/lawliet/internal/resp"
	"github.com/tranbaohieu/lawliet/internal/server"
	"github.com/tranbaohieu/lawliet/internal/server/servertest"
	"github.com/tranbaohieu/lawliet/internal/store"
)

func newClient(t *testing.T) *servertest.Client {
	t.Helper()
	return newClientWithStore(t, store.New())
}

func newClientWithStore(t *testing.T, st *store.Store) *servertest.Client {
	t.Helper()
	addr := servertest.Start(t, func(s *server.Server) { Register(s, st) })
	return servertest.Dial(t, addr)
}

func want(t *testing.T, v resp.Value, expected string) {
	t.Helper()
	if got := servertest.Describe(v); got != expected {
		t.Fatalf("got  %s\nwant %s", got, expected)
	}
}

func wantErr(t *testing.T, v resp.Value, prefix string) {
	t.Helper()
	if v.Type != '-' || !strings.HasPrefix(v.Str, prefix) {
		t.Fatalf("got %s, want error starting with %q", servertest.Describe(v), prefix)
	}
}

func TestCollectionLifecycle(t *testing.T) {
	c := newClient(t)
	want(t, c.Do("VCREATE", "docs", "DIM", 3), `+"OK"`)
	want(t, c.Do("vcreate", "faq", "dim", 2, "metric", "L2", "index", "FLAT"), `+"OK"`)
	wantErr(t, c.Do("VCREATE", "docs", "DIM", 3), "ERR collection already exists")
	want(t, c.Do("VLIST"), `*[$"docs" $"faq"]`)

	want(t, c.Do("VINFO", "docs"),
		`*[$"name" $"docs" $"dim" :3 $"metric" $"cosine" $"index" $"flat" $"count" :0 $"expires" :0 $"max_vectors" :0 $"policy" $"noeviction" $"cache_threshold" nil $"cache_ttl" :0 $"memory_bytes" :0 $"vector_bytes" :0 $"index_params" $"" $"index_state" $"exact" $"index_bytes" :0]`)
	want(t, c.Do("VINFO", "faq").Map()["metric"], `$"l2"`)

	want(t, c.Do("VDROP", "faq"), ":1")
	want(t, c.Do("VDROP", "faq"), ":0")
	wantErr(t, c.Do("VINFO", "faq"), "ERR collection not found")
}

func TestVCreateErrors(t *testing.T) {
	c := newClient(t)
	cases := []struct {
		args []any
		err  string
	}{
		{[]any{"VCREATE", "x", "METRIC", "l2"}, "ERR DIM is required"},
		{[]any{"VCREATE", "x", "DIM", 0}, "ERR DIM must be an integer between 1 and 65536"},
		{[]any{"VCREATE", "x", "DIM", "abc"}, "ERR DIM must be an integer"},
		{[]any{"VCREATE", "x", "DIM", 3, "METRIC", "hamming"}, "ERR unknown metric 'hamming'"},
		{[]any{"VCREATE", "x", "DIM", 3, "INDEX", "annoy"}, "ERR unknown index 'annoy'"},
		{[]any{"VCREATE", "x", "DIM", 3, "INDEX", "hnsw", "NLIST", 10}, "ERR NLIST does not apply to INDEX hnsw"},
		{[]any{"VCREATE", "x", "DIM", 3, "NPROBE", 2}, "ERR NPROBE does not apply to INDEX flat"},
		{[]any{"VCREATE", "x", "DIM", 3, "INDEX", "ivf", "QUANT", "int4"}, "ERR unknown QUANT 'int4'"},
		{[]any{"VCREATE", "x", "DIM", 3, "INDEX", "ivf", "QUANT", "pq", "PQ_M", 2}, "ERR PQ_M must divide DIM (3)"},
		{[]any{"VCREATE", "x", "DIM", 3, "INDEX", "hnsw", "M", 0}, "ERR M must be an integer between 1"},
		{[]any{"VCREATE", "x", "DIM", 3, "BOGUS", 1}, "ERR syntax error"},
		{[]any{"VCREATE", "x", "DIM", 3, "METRIC"}, "ERR syntax error"},
		{[]any{"VCREATE", "x", "DIM"}, "ERR wrong number of arguments"},
	}
	for _, tc := range cases {
		wantErr(t, c.Do(tc.args...), tc.err)
	}
	want(t, c.Do("VLIST"), "*[]")
}

func TestRecordCommands(t *testing.T) {
	c := newClient(t)
	c.Do("VCREATE", "v", "DIM", 2, "METRIC", "ip")

	want(t, c.Do("VADD", "v", "a", "VALUES", 2, "1", "2"), ":1")
	want(t, c.Do("VADD", "v", "a", "FP32", encodeFP32([]float32{3, 4})), ":0") // upsert
	want(t, c.Do("VADD", "v", "a", "VALUES", 2, "9", "9", "NX"), "nil")
	want(t, c.Do("VADD", "v", "b", "VALUES", 2, "9", "9", "XX"), "nil")
	want(t, c.Do("VADD", "v", "b", "VALUES", 2, "5", "6", "nx"), ":1")

	want(t, c.Do("VEXISTS", "v", "a"), ":1")
	want(t, c.Do("VEXISTS", "v", "zzz"), ":0")
	want(t, c.Do("VCOUNT", "v"), ":2")

	want(t, c.Do("VGET", "v", "a"), "*[nil nil]")
	want(t, c.Do("VGET", "v", "a", "WITHVEC"), `*[nil nil $`+strconv.Quote(string(encodeFP32([]float32{3, 4})))+`]`)
	want(t, c.Do("VGET", "v", "missing"), "nil")

	want(t, c.Do("VDEL", "v", "a", "missing"), ":1")
	want(t, c.Do("VCOUNT", "v"), ":1")
	want(t, c.Do("VEXISTS", "v", "a"), ":0")
}

func TestRecordErrors(t *testing.T) {
	c := newClient(t)
	c.Do("VCREATE", "v", "DIM", 3)
	cases := []struct {
		args []any
		err  string
	}{
		{[]any{"VADD", "nope", "a", "VALUES", 1, "1"}, "ERR collection not found"},
		{[]any{"VADD", "v", "a", "VALUES", 2, "1", "2"}, "ERR dimension mismatch: expected 3, got 2"},
		{[]any{"VADD", "v", "a", "FP32", encodeFP32([]float32{1, 2})}, "ERR dimension mismatch: expected 3, got 2"},
		{[]any{"VADD", "v", "a", "FP32", "abcde"}, "ERR invalid FP32 blob: length 5"},
		{[]any{"VADD", "v", "a", "VALUES", 3, "1", "x", "3"}, "ERR invalid number in VALUES: 'x'"},
		{[]any{"VADD", "v", "a", "VALUES", 3, "1", "2"}, "ERR VALUES 3 expects 3 numbers, got 2"},
		{[]any{"VADD", "v", "a", "VALUES", 3, "0", "0", "0"}, "ERR zero vector is not allowed with cosine metric"},
		{[]any{"VADD", "v", "a", "VALUES", 3, "1", "NaN", "0"}, "ERR vector contains NaN or Inf"},
		{[]any{"VADD", "v", "a", "VALUES", 3, "1", "2", "3", "NX", "XX"}, "ERR syntax error"},
		{[]any{"VADD", "v", "a", "VALUES", 3, "1", "2", "3", "TEXT"}, "ERR syntax error"},
		{[]any{"VADD", "v", "a", "VALUES", 3, "1", "2", "3", "META"}, "ERR syntax error"},
		{[]any{"VADD", "v", "a", "VALUES", 3, "1", "2", "3", "META", `{"a":{"b":1}}`}, "ERR invalid metadata: field 'a': nested object not allowed"},
		{[]any{"VADD", "v", "a", "VALUES", 3, "1", "2", "3", "META", `[1]`}, "ERR invalid metadata: must be a JSON object"},
		{[]any{"VADD", "v", "a", "VECTOR", "1"}, "ERR expected FP32 or VALUES, got 'VECTOR'"},
		{[]any{"VGET", "v", "a", "BOGUS"}, "ERR syntax error"},
	}
	for _, tc := range cases {
		wantErr(t, c.Do(tc.args...), tc.err)
	}
	want(t, c.Do("VCOUNT", "v"), ":0")
}

func TestSearch(t *testing.T) {
	c := newClient(t)
	c.Do("VCREATE", "v", "DIM", 2, "METRIC", "ip")
	c.Do("VADD", "v", "a", "VALUES", 2, "1", "0")
	c.Do("VADD", "v", "b", "VALUES", 2, "0.5", "0")
	c.Do("VADD", "v", "c", "VALUES", 2, "-1", "0")

	want(t, c.Do("VSEARCH", "v", "VALUES", 2, "1", "0", "K", 2),
		`*[*[$"a" $"1"] *[$"b" $"0.5"]]`)
	want(t, c.Do("VSEARCH", "v", "FP32", encodeFP32([]float32{1, 0}), "K", 1, "WITHVEC"),
		`*[*[$"a" $"1" $`+strconv.Quote(string(encodeFP32([]float32{1, 0})))+`]]`)

	// Default K is 10; deleted records are not returned.
	c.Do("VDEL", "v", "b")
	want(t, c.Do("VSEARCH", "v", "VALUES", 2, "1", "0"), `*[*[$"a" $"1"] *[$"c" $"-1"]]`)

	wantErr(t, c.Do("VSEARCH", "v", "VALUES", 2, "1", "0", "K", 0), "ERR K must be an integer between 1 and 10000")
	wantErr(t, c.Do("VSEARCH", "v", "VALUES", 2, "1", "0", "K"), "ERR syntax error")
	wantErr(t, c.Do("VSEARCH", "v", "VALUES", 2, "1", "0", "FILTER"), "ERR syntax error")
	wantErr(t, c.Do("VSEARCH", "v", "VALUES", 2, "1", "0", "FILTER", "x"), "ERR invalid filter: expected operator after field 'x'")
	wantErr(t, c.Do("VSEARCH", "v", "VALUES", 1, "1"), "ERR dimension mismatch")
}

func TestSearchRESP3ScoresAreDoubles(t *testing.T) {
	c := newClient(t)
	c.Do("HELLO", "3")
	c.Do("VCREATE", "v", "DIM", 2, "METRIC", "l2")
	c.Do("VADD", "v", "a", "VALUES", 2, "3", "4")

	// l2 score is the negated distance: -5.
	want(t, c.Do("VSEARCH", "v", "VALUES", 2, "0", "0"), `*[*[$"a" ,"-5"]]`)
	want(t, c.Do("VGET", "v", "missing"), "nil")
	if v := c.Do("VINFO", "v"); v.Type != '%' {
		t.Fatalf("VINFO in RESP3 should be a map, got %s", servertest.Describe(v))
	}
}

func TestDecodeEncodeFP32(t *testing.T) {
	in := []float32{1.5, -2, 0, 3.25}
	out, err := decodeFP32(encodeFP32(in))
	if err != nil {
		t.Fatal(err)
	}
	for i := range in {
		if in[i] != out[i] {
			t.Fatalf("round trip: %v != %v", out, in)
		}
	}
	// Little-endian check: 1.0 = 0x3F800000 -> 00 00 80 3F
	if b := encodeFP32([]float32{1}); string(b) != "\x00\x00\x80\x3f" {
		t.Fatalf("not little-endian: % x", b)
	}
}

func TestTextAndMeta(t *testing.T) {
	c := newClient(t)
	c.Do("VCREATE", "docs", "DIM", 2, "METRIC", "ip")

	want(t, c.Do("VADD", "docs", "a", "VALUES", 2, "1", "0",
		"TEXT", "Chính sách nghỉ phép", "META", `{"source":"hr.md","page":3,"tags":["hr","policy"]}`), ":1")
	want(t, c.Do("VGET", "docs", "a"),
		`*[$"Chính sách nghỉ phép" $"{\"page\":3,\"source\":\"hr.md\",\"tags\":[\"hr\",\"policy\"]}"]`)

	// Upsert without TEXT/META clears them.
	want(t, c.Do("VADD", "docs", "a", "VALUES", 2, "1", "0"), ":0")
	want(t, c.Do("VGET", "docs", "a"), "*[nil nil]")

	// Empty META object is stored as "no metadata".
	c.Do("VADD", "docs", "b", "VALUES", 2, "1", "0", "META", "{}", "TEXT", "x")
	want(t, c.Do("VGET", "docs", "b"), `*[$"x" nil]`)
}

func TestSearchFilterAndFields(t *testing.T) {
	c := newClient(t)
	c.Do("VCREATE", "docs", "DIM", 1, "METRIC", "ip")
	docs := []struct {
		id, text, meta string
		v              string
	}{
		{"hr1", "leave policy", `{"source":"hr.md","page":1,"tags":["hr"]}`, "0.9"},
		{"hr2", "leave days", `{"source":"hr.md","page":5,"tags":["hr","draft"]}`, "0.8"},
		{"it1", "vpn setup", `{"source":"it.md","page":2,"tags":["it"]}`, "0.95"},
		{"ops1", "on-call", `{"source":"ops.md","page":3}`, "0.7"},
		{"none", "no meta", ``, "1.0"},
	}
	for _, d := range docs {
		args := []any{"VADD", "docs", d.id, "VALUES", 1, d.v, "TEXT", d.text}
		if d.meta != "" {
			args = append(args, "META", d.meta)
		}
		want(t, c.Do(args...), ":1")
	}

	ids := func(v resp.Value) string {
		var out []string
		for _, h := range v.Array {
			out = append(out, h.Array[0].Str)
		}
		return strings.Join(out, ",")
	}
	q := []any{"VSEARCH", "docs", "VALUES", 1, "1"}
	search := func(extra ...any) resp.Value { return c.Do(append(append([]any{}, q...), extra...)...) }

	if got := ids(search()); got != "none,it1,hr1,hr2,ops1" {
		t.Fatalf("no filter: %s", got)
	}
	if got := ids(search("FILTER", `source == "hr.md"`)); got != "hr1,hr2" {
		t.Fatalf("source filter: %s", got)
	}
	if got := ids(search("FILTER", `(source == "hr.md" or source in ["it.md", "ops.md"]) and page >= 2 and not tags contains "draft"`)); got != "it1,ops1" {
		t.Fatalf("design example: %s", got)
	}
	if got := ids(search("FILTER", `not source == "hr.md"`)); got != "none,it1,ops1" {
		t.Fatalf("not on missing field: %s", got)
	}
	if got := ids(search("FILTER", `page > 100`)); got != "" {
		t.Fatalf("no match: %s", got)
	}
	if got := ids(search("FILTER", `source == "hr.md"`, "K", 1)); got != "hr1" {
		t.Fatalf("filter + K: %s", got)
	}

	// Optional fields come in fixed order: id, score, text, meta, vector — whatever the flag order.
	want(t, search("FILTER", `source == "ops.md"`, "WITHVEC", "WITHMETA", "WITHTEXT"),
		`*[*[$"ops1" $"0.7" $"on-call" $"{\"page\":3,\"source\":\"ops.md\"}" $`+strconv.Quote(string(encodeFP32([]float32{0.7})))+`]]`)
	want(t, search("FILTER", `source == "ops.md"`, "WITHMETA"),
		`*[*[$"ops1" $"0.7" $"{\"page\":3,\"source\":\"ops.md\"}"]]`)
	// Missing meta is null.
	want(t, search("K", 1, "WITHTEXT", "WITHMETA"), `*[*[$"none" $"1" $"no meta" nil]]`)
}

func describe(v resp.Value) string { return servertest.Describe(v) }

func expect(t *testing.T, got, want string) {
	t.Helper()
	if got != want {
		t.Fatalf("got  %s\nwant %s", got, want)
	}
}
