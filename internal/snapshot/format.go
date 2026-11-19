// Package snapshot writes and reads .lwdb snapshot files (DESIGN.md §10).
//
// Format (little-endian):
//
//	magic "LWDB" | version uint16 | n_coll uint32
//	repeat n_coll:
//	  name str | dim uint32 | metric uint8 | index uint8 | max_vec uint32 | policy uint8
//	  [v2+] has_cache_threshold uint8 | cache_threshold float64 | cache_ttl_sec int64
//	  [v3+] quant uint8 | m, ef_construction, ef_search, nlist, nprobe, pq_m, rerank uint32
//	  n_rec uint32
//	  repeat n_rec: id str | vector dim×float32 | text str | meta str (JSON, "" = none) | expireAt int64
//	crc32 uint32 (IEEE, over everything above)
//
//	str = uint32 length + bytes
//
// Version history: 1 = MVP; 2 = adds the semantic cache defaults per collection;
// 3 = adds hnsw/ivf indexes and their options. Index structures are not stored:
// they are rebuilt while records are restored.
// Readers accept every version up to the current one. Writers use the lowest
// version that holds the data: 2 while every collection is flat with default
// options, so an older binary (v0.2.x) can still read the file.
package snapshot

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"hash"
	"hash/crc32"
	"io"
	"math"

	"github.com/tranbaohieu/lawliet/internal/index"
	"github.com/tranbaohieu/lawliet/internal/meta"
	"github.com/tranbaohieu/lawliet/internal/store"
)

const (
	magic   = "LWDB"
	version = 3

	// maxStrLen bounds strings read from a file, so a corrupt length cannot
	// trigger a huge allocation (same bound as a RESP bulk string).
	maxStrLen = 512 << 20
)

// index type codes in the file
var indexCodes = map[string]uint8{index.KindFlat: 0, index.KindHNSW: 1, index.KindIVF: 2}

// ErrCorrupt reports an unreadable snapshot file.
var ErrCorrupt = errors.New("corrupt snapshot")

func corrupt(format string, a ...any) error {
	return fmt.Errorf("%w: %s", ErrCorrupt, fmt.Sprintf(format, a...))
}

// Stats summarizes a write or a load.
type Stats struct {
	Collections int
	Records     int
	Skipped     int // records expired at load time
}

// --- writing ----------------------------------------------------------------

type encoder struct {
	w   *bufio.Writer
	buf [8]byte
}

func (e *encoder) u8(v uint8)   { e.w.WriteByte(v) }
func (e *encoder) u16(v uint16) { binary.LittleEndian.PutUint16(e.buf[:2], v); e.w.Write(e.buf[:2]) }
func (e *encoder) u32(v uint32) { binary.LittleEndian.PutUint32(e.buf[:4], v); e.w.Write(e.buf[:4]) }
func (e *encoder) i64(v int64) {
	binary.LittleEndian.PutUint64(e.buf[:8], uint64(v))
	e.w.Write(e.buf[:8])
}
func (e *encoder) f64(v float64) { e.i64(int64(math.Float64bits(v))) }
func (e *encoder) str(s string)  { e.u32(uint32(len(s))); e.w.WriteString(s) }
func (e *encoder) bytes(b []byte) {
	e.u32(uint32(len(b)))
	e.w.Write(b)
}
func (e *encoder) vec(v []float32) {
	for _, x := range v {
		e.u32(math.Float32bits(x))
	}
}

// writeVersion is the lowest format version that can hold cols: 3 only when a
// collection uses an index other than plain flat (v3 fields), else 2.
func writeVersion(cols []*store.Collection) uint16 {
	for _, c := range cols {
		if cfg := c.Config(); cfg.Index != index.KindFlat || cfg.IndexOpts != (index.Options{}) {
			return version
		}
	}
	return 2
}

// Write encodes all collections of st to w. Each collection is dumped under
// its read lock, one at a time.
func Write(w io.Writer, st *store.Store) (Stats, error) {
	crc := crc32.NewIEEE()
	bw := bufio.NewWriterSize(io.MultiWriter(w, crc), 1<<20)
	e := &encoder{w: bw}

	var stats Stats
	cols := st.Collections()
	fileVersion := writeVersion(cols)
	e.w.WriteString(magic)
	e.u16(fileVersion)
	e.u32(uint32(len(cols)))
	for _, c := range cols {
		err := c.Dump(
			func(h store.DumpHeader) error {
				code, ok := indexCodes[h.Config.Index]
				if !ok {
					return fmt.Errorf("snapshot: unsupported index %q", h.Config.Index)
				}
				e.str(h.Name)
				e.u32(uint32(h.Config.Dim))
				e.u8(uint8(h.Config.Metric))
				e.u8(code)
				e.u32(uint32(h.Config.MaxVectors))
				e.u8(uint8(h.Config.Policy))
				if h.Config.HasCacheThreshold {
					e.u8(1)
				} else {
					e.u8(0)
				}
				e.f64(h.Config.CacheThreshold)
				e.i64(h.Config.CacheTTL)
				if fileVersion >= 3 {
					o := h.Config.IndexOpts
					e.u8(uint8(o.Quant))
					for _, v := range []int{o.M, o.EfConstruction, o.EfSearch, o.NList, o.NProbe, o.PQM, o.Rerank} {
						e.u32(uint32(v))
					}
				}
				e.u32(uint32(h.Count))
				return nil
			},
			func(id string, rec store.Record) error {
				e.str(id)
				e.vec(rec.Vector)
				e.str(rec.Text)
				if len(rec.Meta) > 0 {
					e.bytes(rec.Meta.JSON())
				} else {
					e.u32(0)
				}
				e.i64(rec.ExpireAt)
				stats.Records++
				return nil
			})
		if err != nil {
			return stats, err
		}
		stats.Collections++
	}
	if err := bw.Flush(); err != nil {
		return stats, err
	}
	// The checksum itself is written to w only (not hashed).
	binary.LittleEndian.PutUint32(e.buf[:4], crc.Sum32())
	_, err := w.Write(e.buf[:4])
	return stats, err
}

// --- reading ----------------------------------------------------------------

type decoder struct {
	r   io.Reader
	buf [8]byte
	err error
}

func (d *decoder) read(n int) []byte {
	if d.err != nil {
		return d.buf[:n]
	}
	if _, err := io.ReadFull(d.r, d.buf[:n]); err != nil {
		d.err = corrupt("unexpected end of file")
	}
	return d.buf[:n]
}
func (d *decoder) u8() uint8    { return d.read(1)[0] }
func (d *decoder) u16() uint16  { return binary.LittleEndian.Uint16(d.read(2)) }
func (d *decoder) u32() uint32  { return binary.LittleEndian.Uint32(d.read(4)) }
func (d *decoder) i64() int64   { return int64(binary.LittleEndian.Uint64(d.read(8))) }
func (d *decoder) f64() float64 { return math.Float64frombits(uint64(d.i64())) }

func (d *decoder) rawStr() []byte {
	n := d.u32()
	if d.err != nil {
		return nil
	}
	if n > maxStrLen {
		d.err = corrupt("string length %d too large", n)
		return nil
	}
	b := make([]byte, n)
	if _, err := io.ReadFull(d.r, b); err != nil {
		d.err = corrupt("unexpected end of file")
	}
	return b
}
func (d *decoder) str() string { return string(d.rawStr()) }

func (d *decoder) vec(dim int) []float32 {
	b := make([]byte, dim*4)
	if d.err == nil {
		if _, err := io.ReadFull(d.r, b); err != nil {
			d.err = corrupt("unexpected end of file")
		}
	}
	v := make([]float32, dim)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return v
}

// hashingReader feeds every byte read into a hash.
type hashingReader struct {
	r io.Reader
	h hash.Hash32
}

func (h *hashingReader) Read(p []byte) (int, error) {
	n, err := h.r.Read(p)
	h.h.Write(p[:n])
	return n, err
}

// Read loads a snapshot into st, which should be empty. Records already
// expired are skipped. On error st may hold a partial load; callers must
// discard it (the server refuses to start).
func Read(r io.Reader, st *store.Store) (Stats, error) {
	br := bufio.NewReaderSize(r, 1<<20)
	hr := &hashingReader{r: br, h: crc32.NewIEEE()}
	d := &decoder{r: hr}
	var stats Stats

	if m := string(d.read(4)); d.err != nil || m != magic {
		return stats, corrupt("not a Lawliet snapshot (bad magic)")
	}
	fileVersion := d.u16()
	if d.err == nil && (fileVersion < 1 || fileVersion > version) {
		return stats, corrupt("unsupported snapshot version %d (this build reads versions 1 to %d)", fileVersion, version)
	}
	nColl := d.u32()
	for i := uint32(0); i < nColl && d.err == nil; i++ {
		name := d.str()
		dim := int(d.u32())
		metric := index.Metric(d.u8())
		indexCode := d.u8()
		maxVec := int(d.u32())
		policy := store.Policy(d.u8())
		var hasThreshold bool
		var threshold float64
		var cacheTTL int64
		if fileVersion >= 2 {
			hasThreshold = d.u8() == 1
			threshold = d.f64()
			cacheTTL = d.i64()
		}
		var opts index.Options
		if fileVersion >= 3 {
			opts.Quant = index.Quant(d.u8())
			for _, p := range []*int{&opts.M, &opts.EfConstruction, &opts.EfSearch, &opts.NList, &opts.NProbe, &opts.PQM, &opts.Rerank} {
				*p = int(d.u32())
			}
		}
		nRec := d.u32()
		if d.err != nil {
			break
		}
		indexName := ""
		for n, code := range indexCodes {
			if code == indexCode {
				indexName = n
			}
		}
		if indexName == "" {
			return stats, corrupt("collection %q: unknown index code %d", name, indexCode)
		}
		col, err := st.Restore(name, store.Config{
			Dim: dim, Metric: metric, Index: indexName, IndexOpts: opts, MaxVectors: maxVec, Policy: policy,
			HasCacheThreshold: hasThreshold, CacheThreshold: threshold, CacheTTL: cacheTTL,
		})
		if err != nil {
			return stats, corrupt("collection %q: %v", name, err)
		}
		stats.Collections++

		for j := uint32(0); j < nRec && d.err == nil; j++ {
			id := d.str()
			vec := d.vec(dim)
			text := d.str()
			metaJSON := d.rawStr()
			expireAt := d.i64()
			if d.err != nil {
				break
			}
			var m meta.Map
			if len(metaJSON) > 0 {
				if m, err = meta.ParseJSON(metaJSON); err != nil {
					return stats, corrupt("collection %q record %q: %v", name, id, err)
				}
			}
			ok, err := col.Restore(id, store.Record{Vector: vec, Text: text, Meta: m, ExpireAt: expireAt})
			if err != nil {
				return stats, corrupt("collection %q record %q: %v", name, id, err)
			}
			if ok {
				stats.Records++
			} else {
				stats.Skipped++
			}
		}
	}
	if d.err != nil {
		return stats, d.err
	}

	want := hr.h.Sum32()
	var tail [4]byte
	if _, err := io.ReadFull(br, tail[:]); err != nil {
		return stats, corrupt("missing checksum")
	}
	if got := binary.LittleEndian.Uint32(tail[:]); got != want {
		return stats, corrupt("checksum mismatch (file %08x, computed %08x)", got, want)
	}
	if _, err := br.ReadByte(); err != io.EOF {
		return stats, corrupt("unexpected data after checksum")
	}
	st.ConsumeDirty(st.Dirty()) // loading is not a change to save
	return stats, nil
}
