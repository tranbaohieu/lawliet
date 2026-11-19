package server

import (
	"bytes"
	"fmt"
	"os"
	"runtime"
	"runtime/metrics"
	"strconv"
	"strings"

	"github.com/tranbaohieu/lawliet/internal/vecmath"
	"github.com/tranbaohieu/lawliet/internal/version"
)

// registerConnCommands registers connection-level "courtesy" commands
// needed by Redis clients (DESIGN.md §5.6).
func registerConnCommands(s *Server) {
	s.Register("ping", -1, cmdPing)
	s.Register("echo", 2, cmdEcho)
	s.Register("hello", -1, cmdHello)
	s.Register("select", 2, cmdSelect)
	s.Register("client", -2, cmdClient)
	s.Register("command", -1, cmdCommand)
	s.Register("info", -1, cmdInfo)
	s.Register("quit", -1, cmdQuit)
}

func cmdPing(c *Conn, args [][]byte) {
	switch len(args) {
	case 1:
		c.w.WriteSimple("PONG")
	case 2:
		c.w.WriteBulk(args[1])
	default:
		c.WrongArgs()
	}
}

func cmdEcho(c *Conn, args [][]byte) { c.w.WriteBulk(args[1]) }

func cmdSelect(c *Conn, args [][]byte) {
	n, err := strconv.Atoi(string(args[1]))
	switch {
	case err != nil:
		c.w.WriteError("ERR value is not an integer or out of range")
	case n != 0:
		c.w.WriteError("ERR DB index is out of range")
	default:
		c.w.WriteOK()
	}
}

// HELLO [protover [AUTH username password] [SETNAME clientname]]
// protover 2 or 3 switches the connection protocol; without it the current one is kept.
func cmdHello(c *Conn, args [][]byte) {
	proto := c.w.Proto()
	if len(args) >= 2 {
		ver, err := strconv.Atoi(string(args[1]))
		if err != nil {
			c.w.WriteError("ERR Protocol version is not an integer or out of range")
			return
		}
		if ver != 2 && ver != 3 {
			c.w.WriteError("NOPROTO unsupported protocol version")
			return
		}
		proto = ver
	}
	name := c.name
	for i := 2; i < len(args); i++ {
		switch strings.ToLower(string(args[i])) {
		case "auth":
			if i+2 >= len(args) {
				c.w.WriteError("ERR syntax error")
				return
			}
			c.w.WriteError("ERR AUTH is not supported")
			return
		case "setname":
			if i+1 >= len(args) {
				c.w.WriteError("ERR syntax error")
				return
			}
			if !validClientName(args[i+1]) {
				c.w.WriteError("ERR Client names cannot contain spaces, newlines or special characters.")
				return
			}
			name = string(args[i+1])
			i++
		default:
			c.w.WriteError("ERR syntax error")
			return
		}
	}
	// All arguments are valid: apply changes. The HELLO reply itself uses the new protocol.
	c.name = name
	c.w.SetProto(proto)

	c.w.WriteMapHeader(7)
	c.w.WriteBulkString("server")
	c.w.WriteBulkString("lawliet")
	c.w.WriteBulkString("version")
	c.w.WriteBulkString(version.Version)
	c.w.WriteBulkString("proto")
	c.w.WriteInt(int64(proto))
	c.w.WriteBulkString("id")
	c.w.WriteInt(c.id)
	c.w.WriteBulkString("mode")
	c.w.WriteBulkString("standalone")
	c.w.WriteBulkString("role")
	c.w.WriteBulkString("master")
	c.w.WriteBulkString("modules")
	c.w.WriteArrayHeader(0)
}

// CLIENT SETNAME|GETNAME|SETINFO|ID
func cmdClient(c *Conn, args [][]byte) {
	sub := strings.ToLower(string(args[1]))
	switch {
	case sub == "setname" && len(args) == 3:
		if !validClientName(args[2]) {
			c.w.WriteError("ERR Client names cannot contain spaces, newlines or special characters.")
			return
		}
		c.name = string(args[2])
		c.w.WriteOK()
	case sub == "getname" && len(args) == 2:
		if c.name == "" {
			c.w.WriteNull()
			return
		}
		c.w.WriteBulkString(c.name)
	case sub == "setinfo" && len(args) == 4:
		// Sent by redis-py / go-redis on connect (lib-name, lib-ver). Accepted and ignored.
		c.w.WriteOK()
	case sub == "id" && len(args) == 2:
		c.w.WriteInt(c.id)
	case sub == "setname" || sub == "getname" || sub == "setinfo" || sub == "id":
		c.w.WriteError("ERR wrong number of arguments for 'client|" + sub + "' command")
	default:
		c.w.WriteError(fmt.Sprintf("ERR unknown subcommand '%s'. Try CLIENT HELP.", truncate(sub, 64)))
	}
}

// COMMAND [COUNT|DOCS|...]: minimal implementation so clients and tools don't fail.
func cmdCommand(c *Conn, args [][]byte) {
	if len(args) >= 2 && strings.EqualFold(string(args[1]), "count") {
		c.w.WriteInt(int64(len(c.srv.commands)))
		return
	}
	c.w.WriteArrayHeader(0)
}

func cmdQuit(c *Conn, _ [][]byte) {
	c.w.WriteOK()
	c.closeAfterReply = true
}

// Names may not contain spaces, newlines or non-printable characters (same rule as Redis).
func validClientName(b []byte) bool {
	for _, ch := range b {
		if ch < '!' || ch > '~' {
			return false
		}
	}
	return true
}

// INFO [section]
func cmdInfo(c *Conn, args [][]byte) {
	section := "default"
	if len(args) == 2 {
		section = strings.ToLower(string(args[1]))
	} else if len(args) > 2 {
		c.WrongArgs()
		return
	}
	all := section == "default" || section == "all" || section == "everything"

	s := c.srv
	var buf bytes.Buffer
	add := func(name string, kv [][2]string) {
		if !all && section != name {
			return
		}
		for _, p := range s.info {
			if p.section == name {
				kv = append(kv, p.fn()...)
			}
		}
		if buf.Len() > 0 {
			buf.WriteString("\r\n")
		}
		buf.WriteString("# " + strings.ToUpper(name[:1]) + name[1:] + "\r\n")
		for _, p := range kv {
			buf.WriteString(p[0] + ":" + p[1] + "\r\n")
		}
	}

	port := ""
	if a := s.Addr(); a != nil {
		if i := strings.LastIndexByte(a.String(), ':'); i >= 0 {
			port = a.String()[i+1:]
		}
	}
	add("server", [][2]string{
		{"lawliet_version", version.Version},
		{"go_version", runtime.Version()},
		{"os", runtime.GOOS},
		{"arch", runtime.GOARCH},
		{"process_id", strconv.Itoa(os.Getpid())},
		{"vector_simd", vecmath.Accel()},
		{"tcp_port", port},
		{"uptime_in_seconds", strconv.FormatInt(int64(s.Uptime().Seconds()), 10)},
	})
	add("clients", [][2]string{
		{"connected_clients", strconv.Itoa(s.ConnectedClients())},
	})
	mem := runtimeMemory()
	add("memory", [][2]string{
		{"used_memory_runtime", strconv.FormatUint(mem.total, 10)},
		{"used_memory_heap_live", strconv.FormatUint(mem.heapLive, 10)},
	})
	add("stats", [][2]string{
		{"total_connections_received", strconv.FormatInt(s.stats.ConnectionsReceived.Load(), 10)},
		{"total_commands_processed", strconv.FormatInt(s.stats.CommandsProcessed.Load(), 10)},
	})
	seen := map[string]bool{"server": true, "clients": true, "memory": true, "stats": true}
	for _, p := range s.info {
		if !seen[p.section] {
			seen[p.section] = true
			add(p.section, nil)
		}
	}
	c.w.WriteBulkString(buf.String())
}

type runtimeMem struct {
	total    uint64 // all memory mapped by the Go runtime
	heapLive uint64 // heap still in use after the last GC
}

// runtimeMemory reads Go runtime memory metrics (cheap, no stop-the-world).
func runtimeMemory() runtimeMem {
	samples := []metrics.Sample{
		{Name: "/memory/classes/total:bytes"},
		{Name: "/gc/heap/live:bytes"},
	}
	metrics.Read(samples)
	u := func(s metrics.Sample) uint64 {
		if s.Value.Kind() != metrics.KindUint64 {
			return 0
		}
		return s.Value.Uint64()
	}
	return runtimeMem{total: u(samples[0]), heapLive: u(samples[1])}
}
