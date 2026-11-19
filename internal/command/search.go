package command

import (
	"github.com/tranbaohieu/lawliet/internal/filter"
	"github.com/tranbaohieu/lawliet/internal/index"
	"github.com/tranbaohieu/lawliet/internal/server"
	"github.com/tranbaohieu/lawliet/internal/store"
)

// VSEARCH <col> (FP32 <blob> | VALUES <n> v...)
//
//	[K <n>] [FILTER <expr>] [WITHTEXT] [WITHMETA] [WITHVEC]
//	[EF <n>] [NPROBE <n>] [RERANK <n>]
//
// EF / NPROBE / RERANK override the index search options for this query
// (ignored by indexes they do not apply to).
//
// Reply: array of hits, best first. Each hit is
// [id, score, (text), (meta_json), (vector)] — optional parts appear in this
// fixed order, whatever the order of the flags.
func (h *handlers) vsearch(c *server.Conn, args [][]byte) {
	col, err := h.st.Get(string(args[1]))
	if err != nil {
		writeErr(c, err)
		return
	}
	query, i, err := parseVector(args, 2)
	if err != nil {
		writeErr(c, err)
		return
	}
	opt := store.SearchOptions{K: 10}
	for ; i < len(args); i++ {
		switch {
		case isKeyword(args[i], "K") && i+1 < len(args):
			if opt.K, err = parseInt(args[i+1], "K", 1, MaxK); err != nil {
				writeErr(c, err)
				return
			}
			i++
		case isKeyword(args[i], "FILTER") && i+1 < len(args):
			f, err := filter.Parse(string(args[i+1]))
			if err != nil {
				writeErr(c, err)
				return
			}
			opt.Filter = f.Match
			i++
		case isKeyword(args[i], "EF") && i+1 < len(args):
			if opt.Params.EfSearch, err = parseInt(args[i+1], "EF", 1, 4096); err != nil {
				writeErr(c, err)
				return
			}
			i++
		case isKeyword(args[i], "NPROBE") && i+1 < len(args):
			if opt.Params.NProbe, err = parseInt(args[i+1], "NPROBE", 1, index.MaxNList); err != nil {
				writeErr(c, err)
				return
			}
			i++
		case isKeyword(args[i], "RERANK") && i+1 < len(args):
			if opt.Params.Rerank, err = parseInt(args[i+1], "RERANK", 1, index.MaxRerank); err != nil {
				writeErr(c, err)
				return
			}
			i++
		case isKeyword(args[i], "WITHTEXT"):
			opt.WithText = true
		case isKeyword(args[i], "WITHMETA"):
			opt.WithMeta = true
		case isKeyword(args[i], "WITHVEC"):
			opt.WithVectors = true
		default:
			writeErr(c, errSyntax)
			return
		}
	}

	results, err := col.Search(query, opt)
	if err != nil {
		writeErr(c, err)
		return
	}

	fields := 2
	for _, on := range []bool{opt.WithText, opt.WithMeta, opt.WithVectors} {
		if on {
			fields++
		}
	}
	w := c.W()
	w.WriteArrayHeader(len(results))
	for _, r := range results {
		w.WriteArrayHeader(fields)
		w.WriteBulkString(r.ID)
		w.WriteFloat32(r.Score)
		if opt.WithText {
			writeText(c, r.Text)
		}
		if opt.WithMeta {
			writeMeta(c, r.Meta)
		}
		if opt.WithVectors {
			w.WriteBulk(encodeFP32(r.Vector))
		}
	}
}
