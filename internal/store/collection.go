package store

import (
	"slices"
	"sync"

	"github.com/tranbaohieu/lawliet/internal/index"
	"github.com/tranbaohieu/lawliet/internal/meta"
	"github.com/tranbaohieu/lawliet/internal/vecmath"
)

// MaxDim is the largest supported vector dimension.
const MaxDim = 65536

// MaxVectorsLimit is the largest accepted max_vectors value.
const MaxVectorsLimit = 1<<31 - 1

// Config is the configuration of a collection. Dim, Metric and Index are
// immutable; MaxVectors and Policy can be changed with SetMaxVectors / SetPolicy.
type Config struct {
	Dim        int
	Metric     index.Metric
	Index      string        // index.KindFlat, KindHNSW or KindIVF
	IndexOpts  index.Options // zero = defaults
	MaxVectors int           // 0 = unlimited
	Policy     Policy

	// Semantic cache defaults for VCACHE.* (DESIGN.md §16.1). Mutable.
	HasCacheThreshold bool    // false = VCACHE.GET must pass THRESHOLD
	CacheThreshold    float64 // minimum score for a cache hit
	CacheTTL          int64   // seconds; default TTL for VCACHE.SET, 0 = none
}

func (c Config) validate() error {
	if c.Dim < 1 || c.Dim > MaxDim {
		return &ConfigError{Msg: "DIM must be between 1 and 65536"}
	}
	switch c.Metric {
	case index.Cosine, index.L2, index.IP:
	default:
		return &ConfigError{Msg: "invalid metric"}
	}
	if err := index.Validate(c.Index, c.Dim, c.IndexOpts); err != nil {
		return &ConfigError{Msg: err.Error()}
	}
	if c.MaxVectors < 0 || c.MaxVectors > MaxVectorsLimit {
		return &ConfigError{Msg: "MAX_VECTORS must be between 0 and 2147483647"}
	}
	switch c.Policy {
	case NoEviction, AllKeysLRU, VolatileTTL:
	default:
		return &ConfigError{Msg: "invalid policy"}
	}
	if c.HasCacheThreshold {
		if err := validateCacheThreshold(c.CacheThreshold); err != nil {
			return err
		}
	}
	return validateCacheTTL(c.CacheTTL)
}

// AddMode controls VADD behavior for existing IDs.
type AddMode uint8

const (
	Upsert       AddMode = iota // insert or overwrite
	OnlyNew                     // NX: insert only if absent
	OnlyExisting                // XX: overwrite only if present
)

// AddResult reports what Add did.
type AddResult uint8

const (
	Inserted AddResult = iota
	Updated
	Skipped // NX/XX condition not met
)

// Record is the content of one entry. Meta is treated as immutable once stored.
type Record struct {
	Vector   []float32
	Text     string
	Meta     meta.Map // nil = no metadata
	ExpireAt int64    // absolute unix time in ms; 0 = no TTL
}

// SearchOptions controls Search.
type SearchOptions struct {
	K           int
	Filter      func(meta.Map) bool // nil = no filter
	WithVectors bool
	WithText    bool
	WithMeta    bool
	NoTouch     bool // do not count returned records as accessed (LRU); see Touch
	Params      index.SearchParams
}

// Result is one search hit with its external ID. Optional fields are set
// only when requested in SearchOptions.
type Result struct {
	ID     string
	Score  float32
	Vector []float32
	Text   string
	Meta   meta.Map
}

// Info is a snapshot of collection statistics.
type Info struct {
	Name        string
	Config      Config
	Count       int   // live records, including expired ones not yet reclaimed
	Expires     int   // records with a TTL
	Slots       int   // allocated slots, including tombstones
	VectorBytes int64 // memory held by the vector slab
	MemoryBytes int64 // estimated memory of all records (DESIGN.md §8.2)
	IndexStats  index.Stats
}

// Collection holds records of one vector space.
//
// Records are addressed by a user-defined string ID mapped to an internal
// uint32 slot. Per-slot data lives in parallel slices indexed by slot.
// Deleted slots become tombstones and are reused through freeList.
//
// Expired records are invisible to every read immediately; their memory is
// reclaimed lazily (on access) or by the active expiry cycle (see ttl.go).
type Collection struct {
	mu   sync.RWMutex
	name string
	cfg  Config
	st   *Store       // owner: memory accounting and statistics
	now  func() int64 // unix ms; injectable for tests

	vectors    *VectorSlab
	idToSlot   map[string]uint32
	slotToID   []string
	texts      []string
	metas      []meta.Map
	expireAt   []int64
	lastAccess []int64 // unix ms, updated atomically (also under the read lock)
	deleted    []bool
	freeList   []uint32
	ttlSlots   map[uint32]struct{} // slots with expireAt != 0
	live       int
	memBytes   int64      // estimated memory of this collection's records
	stats      cacheStats // VCACHE.GET counters

	idx index.Index
}

func newCollection(name string, cfg Config, st *Store) *Collection {
	c := &Collection{
		name:     name,
		cfg:      cfg,
		st:       st,
		now:      st.now,
		vectors:  NewVectorSlab(cfg.Dim),
		idToSlot: make(map[string]uint32),
		ttlSlots: make(map[uint32]struct{}),
	}
	idx, err := index.New(c, cfg.Metric, cfg.Dim, cfg.Index, cfg.IndexOpts)
	if err != nil {
		panic("store: invalid index config: " + err.Error()) // validated by Create/Restore
	}
	c.idx = idx
	return c
}

// NumSlots implements index.Source. Caller must hold c.mu.
func (c *Collection) NumSlots() int { return len(c.slotToID) }

// Vector implements index.Source. Caller must hold c.mu.
func (c *Collection) Vector(slot uint32) []float32 { return c.vectors.Get(slot) }

// Name returns the collection name.
func (c *Collection) Name() string { return c.name }

// Config returns the collection configuration.
func (c *Collection) Config() Config {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.cfg
}

// prepare validates v and returns the form to store or query with.
// For cosine the vector is normalized into a copy; v itself is never modified.
func (c *Collection) prepare(v []float32) ([]float32, error) {
	if len(v) != c.cfg.Dim {
		return nil, &DimMismatchError{Expected: c.cfg.Dim, Got: len(v)}
	}
	if !vecmath.IsFinite(v) {
		return nil, ErrNonFinite
	}
	if c.cfg.Metric == index.Cosine {
		v = slices.Clone(v)
		if !vecmath.Normalize(v) {
			return nil, ErrZeroVector
		}
	}
	return v, nil
}

// lookup returns the slot of a live, non-expired id. Caller must hold c.mu.
func (c *Collection) lookup(id string, now int64) (slot uint32, ok, expired bool) {
	slot, ok = c.idToSlot[id]
	if ok && c.isExpired(slot, now) {
		return slot, false, true
	}
	return slot, ok, false
}

// Add inserts or updates the record id according to mode.
// An update replaces the whole record: text, metadata and TTL not given are cleared.
// An expired record counts as absent (NX succeeds, XX is skipped).
//
// Limits (DESIGN.md §8.4): maxmemory is enforced first, without holding the
// collection lock (eviction may lock any collection); max_vectors is enforced
// under the lock, for new records only.
func (c *Collection) Add(id string, rec Record, mode AddMode) (AddResult, error) {
	v, err := c.prepare(rec.Vector)
	if err != nil {
		return 0, err
	}
	newSize := recordBytes(c.cfg.Dim, id, rec.Text, rec.Meta)

	// An update only needs the size difference.
	c.mu.RLock()
	var oldSize int64
	if slot, ok, _ := c.lookup(id, c.now()); ok {
		oldSize = c.slotBytes(slot)
	}
	c.mu.RUnlock()
	if err := c.st.freeMemory(newSize - oldSize); err != nil {
		return 0, err
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	now := c.now()

	slot, ok, expired := c.lookup(id, now)
	if expired {
		c.removeSlot(id, slot)
	}
	if ok {
		if mode == OnlyNew {
			return Skipped, nil
		}
		c.addMem(newSize - c.slotBytes(slot))
		c.idx.Remove(slot)
		c.vectors.Set(slot, v)
		c.texts[slot] = rec.Text
		c.metas[slot] = rec.Meta
		c.setExpire(slot, rec.ExpireAt)
		c.touch(slot, now)
		c.idx.Add(slot, v)
		c.st.dirty.Add(1)
		return Updated, nil
	}
	if mode == OnlyExisting {
		return Skipped, nil
	}
	if err := c.makeRoom(); err != nil {
		return 0, err
	}

	if n := len(c.freeList); n > 0 {
		slot = c.freeList[n-1]
		c.freeList = c.freeList[:n-1]
		c.slotToID[slot] = id
		c.texts[slot] = rec.Text
		c.metas[slot] = rec.Meta
		c.deleted[slot] = false
	} else {
		slot = uint32(len(c.slotToID))
		c.slotToID = append(c.slotToID, id)
		c.texts = append(c.texts, rec.Text)
		c.metas = append(c.metas, rec.Meta)
		c.expireAt = append(c.expireAt, 0)
		c.lastAccess = append(c.lastAccess, 0)
		c.deleted = append(c.deleted, false)
	}
	c.setExpire(slot, rec.ExpireAt)
	c.touch(slot, now)
	c.vectors.Set(slot, v)
	c.idToSlot[id] = slot
	c.live++
	c.addMem(newSize)
	c.idx.Add(slot, v)
	c.st.dirty.Add(1)
	return Inserted, nil
}

// removeSlot deletes a record. Caller must hold c.mu for writing.
func (c *Collection) removeSlot(id string, slot uint32) {
	c.st.dirty.Add(1)
	c.addMem(-c.slotBytes(slot))
	c.idx.Remove(slot)
	delete(c.idToSlot, id)
	c.slotToID[slot] = ""
	c.texts[slot] = "" // release memory to the GC
	c.metas[slot] = nil
	c.setExpire(slot, 0)
	c.deleted[slot] = true
	c.freeList = append(c.freeList, slot)
	c.live--
}

// Get returns the record. The vector is a copy (normalized for cosine).
func (c *Collection) Get(id string) (Record, bool) {
	now := c.now()
	c.mu.RLock()
	slot, ok, expired := c.lookup(id, now)
	if !ok {
		c.mu.RUnlock()
		if expired {
			c.reclaim(id)
		}
		return Record{}, false
	}
	rec := Record{
		Vector:   slices.Clone(c.vectors.Get(slot)),
		Text:     c.texts[slot],
		Meta:     c.metas[slot],
		ExpireAt: c.expireAt[slot],
	}
	c.touch(slot, now)
	c.mu.RUnlock()
	return rec, true
}

// Exists reports whether id is present and not expired.
func (c *Collection) Exists(id string) bool {
	now := c.now()
	c.mu.RLock()
	_, ok, expired := c.lookup(id, now)
	c.mu.RUnlock()
	if expired {
		c.reclaim(id)
	}
	return ok
}

// Delete removes the given IDs and returns how many existed (expired ones do not count).
func (c *Collection) Delete(ids ...string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	now := c.now()
	n := 0
	for _, id := range ids {
		slot, ok, expired := c.lookup(id, now)
		if ok || expired {
			c.removeSlot(id, slot)
		}
		if ok {
			n++
		}
	}
	return n
}

// Count returns the number of records, including expired ones not yet
// reclaimed (same as Redis DBSIZE). The active expiry cycle keeps it close.
func (c *Collection) Count() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.live
}

// Search returns up to opt.K records most similar to query, best first.
// Deleted and expired records are skipped; the filter runs before the distance
// computation, so filtered-out records cost only a metadata check.
func (c *Collection) Search(query []float32, opt SearchOptions) ([]Result, error) {
	q, err := c.prepare(query)
	if err != nil {
		return nil, err
	}
	now := c.now()

	c.mu.RLock()
	var expired []uint32 // seen during the scan; reclaimed after releasing the read lock
	accept := func(slot uint32) bool {
		if c.deleted[slot] {
			return false
		}
		if c.isExpired(slot, now) {
			expired = append(expired, slot)
			return false
		}
		return opt.Filter == nil || opt.Filter(c.metas[slot])
	}

	hits := c.idx.Search(q, opt.K, accept, opt.Params)
	out := make([]Result, len(hits))
	for i, h := range hits {
		if !opt.NoTouch {
			c.touch(h.Slot, now)
		}
		r := Result{ID: c.slotToID[h.Slot], Score: h.Score}
		if opt.WithVectors {
			r.Vector = slices.Clone(c.vectors.Get(h.Slot))
		}
		if opt.WithText {
			r.Text = c.texts[h.Slot]
		}
		if opt.WithMeta {
			r.Meta = c.metas[h.Slot]
		}
		out[i] = r
	}
	c.mu.RUnlock()

	if len(expired) > 0 {
		c.reclaimSlots(expired)
	}
	return out, nil
}

// Info returns collection statistics.
func (c *Collection) Info() Info {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return Info{
		Name:        c.name,
		Config:      c.cfg,
		Count:       c.live,
		Expires:     len(c.ttlSlots),
		Slots:       len(c.slotToID),
		VectorBytes: c.vectors.Bytes(),
		MemoryBytes: c.memBytes,
		IndexStats:  c.idx.Stats(),
	}
}
