package command

import (
	"sync/atomic"
	"testing"
	"time"

	"github.com/tranbaohieu/lawliet/internal/store"
)

func newTTLClient(t *testing.T) (*storeClock, *store.Store, func(args ...any) string) {
	t.Helper()
	clk := &storeClock{}
	clk.ms.Store(1_000_000)
	st := store.New()
	st.SetClock(clk.ms.Load)
	c := newClientWithStore(t, st)
	c.Do("VCREATE", "v", "DIM", 1, "METRIC", "ip")
	do := func(args ...any) string { return describe(c.Do(args...)) }
	return clk, st, do
}

type storeClock struct{ ms atomic.Int64 }

func (c *storeClock) advance(d time.Duration) { c.ms.Add(d.Milliseconds()) }

func TestVAddWithTTL(t *testing.T) {
	clk, _, do := newTTLClient(t)

	expect(t, do("VADD", "v", "a", "VALUES", 1, "1", "EX", 10), ":1")
	expect(t, do("VADD", "v", "b", "VALUES", 1, "1", "PX", 1500), ":1")
	expect(t, do("VADD", "v", "c", "VALUES", 1, "1"), ":1")

	expect(t, do("VTTL", "v", "a"), ":10")
	expect(t, do("VTTL", "v", "b"), ":2") // 1.5s rounds like Redis TTL
	expect(t, do("VTTL", "v", "c"), ":-1")
	expect(t, do("VTTL", "v", "missing"), ":-2")
	expect(t, do("VINFO", "v"), `*[$"name" $"v" $"dim" :1 $"metric" $"ip" $"index" $"flat" $"count" :3 $"expires" :2 $"max_vectors" :0 $"policy" $"noeviction" $"cache_threshold" nil $"cache_ttl" :0 $"memory_bytes" :306 $"vector_bytes" :4096 $"index_params" $"" $"index_state" $"exact" $"index_bytes" :0]`)

	clk.advance(1500 * time.Millisecond)
	expect(t, do("VEXISTS", "v", "b"), ":0")
	expect(t, do("VGET", "v", "b"), "nil")
	expect(t, do("VTTL", "v", "b"), ":-2")
	expect(t, do("VSEARCH", "v", "VALUES", 1, "1"), `*[*[$"a" $"1"] *[$"c" $"1"]]`)

	// Upsert without EX/PX removes the TTL.
	expect(t, do("VADD", "v", "a", "VALUES", 1, "1"), ":0")
	expect(t, do("VTTL", "v", "a"), ":-1")

	// NX on an expired record inserts.
	expect(t, do("VADD", "v", "d", "VALUES", 1, "1", "PX", 10), ":1")
	clk.advance(time.Second)
	expect(t, do("VADD", "v", "d", "VALUES", 1, "1", "NX"), ":1")
}

func TestVExpireVPersist(t *testing.T) {
	clk, _, do := newTTLClient(t)
	do("VADD", "v", "a", "VALUES", 1, "1")

	expect(t, do("VEXPIRE", "v", "a", 100), ":1")
	expect(t, do("VTTL", "v", "a"), ":100")
	expect(t, do("VEXPIRE", "v", "missing", 100), ":0")

	expect(t, do("VPERSIST", "v", "a"), ":1")
	expect(t, do("VPERSIST", "v", "a"), ":0") // no TTL anymore
	expect(t, do("VTTL", "v", "a"), ":-1")
	expect(t, do("VPERSIST", "v", "missing"), ":0")

	// Non-positive VEXPIRE deletes immediately.
	expect(t, do("VEXPIRE", "v", "a", 0), ":1")
	expect(t, do("VEXISTS", "v", "a"), ":0")
	expect(t, do("VCOUNT", "v"), ":0")

	do("VADD", "v", "b", "VALUES", 1, "1", "EX", 1)
	clk.advance(2 * time.Second)
	expect(t, do("VEXPIRE", "v", "b", 100), ":0") // already expired
}

func TestTTLErrors(t *testing.T) {
	_, _, do := newTTLClient(t)
	errs := []struct {
		args []any
		want string
	}{
		{[]any{"VADD", "v", "a", "VALUES", 1, "1", "EX", 0}, "-\"ERR invalid expire time\""},
		{[]any{"VADD", "v", "a", "VALUES", 1, "1", "EX", -5}, "-\"ERR invalid expire time\""},
		{[]any{"VADD", "v", "a", "VALUES", 1, "1", "PX", "abc"}, "-\"ERR invalid expire time\""},
		{[]any{"VADD", "v", "a", "VALUES", 1, "1", "EX", "99999999999999"}, "-\"ERR invalid expire time\""},
		{[]any{"VADD", "v", "a", "VALUES", 1, "1", "EX", 1, "PX", 1}, "-\"ERR syntax error\""},
		{[]any{"VADD", "v", "a", "VALUES", 1, "1", "EX"}, "-\"ERR syntax error\""},
		{[]any{"VEXPIRE", "v", "a", "x"}, "-\"ERR invalid expire time\""},
		{[]any{"VEXPIRE", "v", "a"}, "-\"ERR wrong number of arguments for 'vexpire' command\""},
		{[]any{"VTTL", "nope", "a"}, "-\"ERR collection not found\""},
	}
	for _, e := range errs {
		expect(t, do(e.args...), e.want)
	}
	expect(t, do("VCOUNT", "v"), ":0")
}

func TestActiveExpiryThroughStore(t *testing.T) {
	clk, st, do := newTTLClient(t)
	for i := 0; i < 50; i++ {
		do("VADD", "v", "k"+string(rune('a'+i%26))+string(rune('a'+i/26)), "VALUES", 1, "1", "PX", 100)
	}
	expect(t, do("VCOUNT", "v"), ":50")
	clk.advance(time.Second)
	// No reads touch the records; only the active cycle reclaims them.
	for i := 0; i < 100 && do("VCOUNT", "v") != ":0"; i++ {
		st.ExpireCycle(time.Now().Add(time.Second))
	}
	expect(t, do("VCOUNT", "v"), ":0")
}
