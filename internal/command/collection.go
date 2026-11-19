package command

import (
	"math"
	"strings"

	"github.com/tranbaohieu/lawliet/internal/index"
	"github.com/tranbaohieu/lawliet/internal/server"
	"github.com/tranbaohieu/lawliet/internal/store"
)

// VCREATE <col> DIM <n> [METRIC cosine|l2|ip] [INDEX flat|hnsw|ivf]
//
//	[M <n>] [EF_CONSTRUCTION <n>] [EF_SEARCH <n>]            (hnsw)
//	[NLIST <n>] [NPROBE <n>]                                 (ivf)
//	[QUANT none|sq8|pq] [PQ_M <n>] [RERANK <n>]              (flat, ivf)
//	[MAX_VECTORS <n>] [POLICY noeviction|allkeys-lru|volatile-ttl]
//	[CACHE_THRESHOLD <score>] [CACHE_TTL <sec>]
//
// POLICY defaults to the server's --maxmemory-policy. CACHE_* are defaults for VCACHE.*.
// Index options are validated against the index kind (DESIGN.md §7.1).
func (h *handlers) vcreate(c *server.Conn, args [][]byte) {
	cfg := store.Config{Metric: index.Cosine, Index: "flat", Policy: h.st.Limits().DefaultPolicy}
	for i := 2; i < len(args); i += 2 {
		if i+1 >= len(args) {
			writeErr(c, errSyntax)
			return
		}
		val := args[i+1]
		switch {
		case isKeyword(args[i], "DIM"):
			n, err := parseInt(val, "DIM", 1, store.MaxDim)
			if err != nil {
				writeErr(c, err)
				return
			}
			cfg.Dim = n
		case isKeyword(args[i], "METRIC"):
			m, err := index.ParseMetric(string(val))
			if err != nil {
				writeErr(c, cmdError(err.Error()))
				return
			}
			cfg.Metric = m
		case isKeyword(args[i], "INDEX"):
			cfg.Index = strings.ToLower(string(val))
		case isKeyword(args[i], "QUANT"):
			q, err := index.ParseQuant(string(val))
			if err != nil {
				writeErr(c, cmdError(err.Error()))
				return
			}
			cfg.IndexOpts.Quant = q
		case indexIntOption(&cfg.IndexOpts, args[i]) != nil:
			name := strings.ToUpper(string(args[i]))
			n, err := parseInt(val, name, 1, index.MaxNList)
			if err != nil {
				writeErr(c, err)
				return
			}
			*indexIntOption(&cfg.IndexOpts, args[i]) = n
		case isKeyword(args[i], "MAX_VECTORS"):
			n, err := parseInt(val, "MAX_VECTORS", 0, store.MaxVectorsLimit)
			if err != nil {
				writeErr(c, err)
				return
			}
			cfg.MaxVectors = n
		case isKeyword(args[i], "POLICY"):
			p, err := store.ParsePolicy(string(val))
			if err != nil {
				writeErr(c, cmdError(err.Error()))
				return
			}
			cfg.Policy = p
		case isKeyword(args[i], "CACHE_THRESHOLD"):
			f, err := parseFloat(val, "CACHE_THRESHOLD")
			if err != nil {
				writeErr(c, err)
				return
			}
			cfg.HasCacheThreshold, cfg.CacheThreshold = true, f
		case isKeyword(args[i], "CACHE_TTL"):
			n, err := parseInt(val, "CACHE_TTL", 0, maxTTLMs/1000)
			if err != nil {
				writeErr(c, err)
				return
			}
			cfg.CacheTTL = int64(n)
		default:
			writeErr(c, errSyntax)
			return
		}
	}
	if cfg.Dim == 0 {
		writeErr(c, cmdError("DIM is required"))
		return
	}
	if _, err := h.st.Create(string(args[1]), cfg); err != nil {
		writeErr(c, err)
		return
	}
	c.W().WriteOK()
}

// VDROP <col>
func (h *handlers) vdrop(c *server.Conn, args [][]byte) {
	if h.st.Drop(string(args[1])) {
		c.W().WriteInt(1)
	} else {
		c.W().WriteInt(0)
	}
}

// VLIST
func (h *handlers) vlist(c *server.Conn, _ [][]byte) {
	names := h.st.List()
	c.W().WriteArrayHeader(len(names))
	for _, n := range names {
		c.W().WriteBulkString(n)
	}
}

// VINFO <col>
func (h *handlers) vinfo(c *server.Conn, args [][]byte) {
	col, err := h.st.Get(string(args[1]))
	if err != nil {
		writeErr(c, err)
		return
	}
	info := col.Info()
	w := c.W()
	w.WriteMapHeader(15)
	w.WriteBulkString("name")
	w.WriteBulkString(info.Name)
	w.WriteBulkString("dim")
	w.WriteInt(int64(info.Config.Dim))
	w.WriteBulkString("metric")
	w.WriteBulkString(info.Config.Metric.String())
	w.WriteBulkString("index")
	w.WriteBulkString(info.Config.Index)
	w.WriteBulkString("count")
	w.WriteInt(int64(info.Count))
	w.WriteBulkString("expires")
	w.WriteInt(int64(info.Expires))
	w.WriteBulkString("max_vectors")
	w.WriteInt(int64(info.Config.MaxVectors))
	w.WriteBulkString("policy")
	w.WriteBulkString(info.Config.Policy.String())
	writeCacheConfig(c, info.Config)
	w.WriteBulkString("memory_bytes")
	w.WriteInt(info.MemoryBytes)
	w.WriteBulkString("vector_bytes")
	w.WriteInt(info.VectorBytes)
	// Added with the ANN indexes: appended so positional RESP2 parsers keep working.
	w.WriteBulkString("index_params")
	w.WriteBulkString(index.Describe(info.Config.Index, info.Config.Dim, info.Config.IndexOpts))
	w.WriteBulkString("index_state")
	w.WriteBulkString(info.IndexStats.State)
	w.WriteBulkString("index_bytes")
	w.WriteInt(info.IndexStats.Bytes)
}

// VCONFIG <col> GET
// VCONFIG <col> SET <param> <value> [<param> <value> ...]
//
// Mutable parameters: POLICY, MAX_VECTORS. All values are validated before any
// change is applied. Lowering MAX_VECTORS below the current count only blocks
// new inserts.
func (h *handlers) vconfig(c *server.Conn, args [][]byte) {
	col, err := h.st.Get(string(args[1]))
	if err != nil {
		writeErr(c, err)
		return
	}
	switch {
	case isKeyword(args[2], "GET") && len(args) == 3:
		cfg := col.Config()
		w := c.W()
		w.WriteMapHeader(4)
		w.WriteBulkString("max_vectors")
		w.WriteInt(int64(cfg.MaxVectors))
		w.WriteBulkString("policy")
		w.WriteBulkString(cfg.Policy.String())
		writeCacheConfig(c, cfg)

	case isKeyword(args[2], "SET") && len(args) >= 5 && len(args)%2 == 1:
		var policy *store.Policy
		maxVectors := -1
		var threshold *float64 // nil = unchanged; NaN = clear
		cacheTTL := int64(-1)
		for i := 3; i < len(args); i += 2 {
			val := args[i+1]
			switch {
			case isKeyword(args[i], "POLICY"):
				p, err := store.ParsePolicy(string(val))
				if err != nil {
					writeErr(c, cmdError(err.Error()))
					return
				}
				policy = &p
			case isKeyword(args[i], "MAX_VECTORS"):
				if maxVectors, err = parseInt(val, "MAX_VECTORS", 0, store.MaxVectorsLimit); err != nil {
					writeErr(c, err)
					return
				}
			case isKeyword(args[i], "CACHE_THRESHOLD"):
				f := math.NaN() // NONE clears the default threshold
				if !isKeyword(val, "NONE") {
					if f, err = parseFloat(val, "CACHE_THRESHOLD"); err != nil {
						writeErr(c, err)
						return
					}
				}
				threshold = &f
			case isKeyword(args[i], "CACHE_TTL"):
				n, err := parseInt(val, "CACHE_TTL", 0, maxTTLMs/1000)
				if err != nil {
					writeErr(c, err)
					return
				}
				cacheTTL = int64(n)
			default:
				writeErr(c, cmdError("unknown or immutable parameter '"+string(args[i])+
					"' (mutable: POLICY, MAX_VECTORS, CACHE_THRESHOLD, CACHE_TTL)"))
				return
			}
		}
		if policy != nil {
			col.SetPolicy(*policy)
		}
		if maxVectors >= 0 {
			if err := col.SetMaxVectors(maxVectors); err != nil {
				writeErr(c, err)
				return
			}
		}
		if threshold != nil {
			if err := col.SetCacheThreshold(!math.IsNaN(*threshold), *threshold); err != nil {
				writeErr(c, err)
				return
			}
		}
		if cacheTTL >= 0 {
			if err := col.SetCacheTTL(cacheTTL); err != nil {
				writeErr(c, err)
				return
			}
		}
		c.W().WriteOK()

	default:
		writeErr(c, errSyntax)
	}
}

// writeCacheConfig writes the cache_threshold (null if unset) and cache_ttl map entries.
func writeCacheConfig(c *server.Conn, cfg store.Config) {
	w := c.W()
	w.WriteBulkString("cache_threshold")
	if cfg.HasCacheThreshold {
		w.WriteDouble(cfg.CacheThreshold)
	} else {
		w.WriteNull()
	}
	w.WriteBulkString("cache_ttl")
	w.WriteInt(cfg.CacheTTL)
}

// indexIntOption returns the integer index option named kw, or nil.
func indexIntOption(o *index.Options, kw []byte) *int {
	for name, p := range map[string]*int{
		"M": &o.M, "EF_CONSTRUCTION": &o.EfConstruction, "EF_SEARCH": &o.EfSearch,
		"NLIST": &o.NList, "NPROBE": &o.NProbe, "PQ_M": &o.PQM, "RERANK": &o.Rerank,
	} {
		if isKeyword(kw, name) {
			return p
		}
	}
	return nil
}
