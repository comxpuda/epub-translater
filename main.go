package main

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	htmlpkg "golang.org/x/net/html"
)

var version = "0.5.0"

// ===== CLI flags =====
var (
	inPath        = flag.String("in", "", "Input .epub path")
	outPath       = flag.String("out", "", "Output .epub path (default: <stem>.<to>.epub)")
	srcLang       = flag.String("from", "auto", "Source language code or 'auto'")
	tgtLang       = flag.String("to", "", "Target language code (required)")
	model         = flag.String("model", "qwen2.5:14b", "Model name (OpenAI-compatible / Ollama compatible)")
	apiKey        = flag.String("apikey", os.Getenv("OPENAI_API_KEY"), "API key (optional; set for OpenAI/compatible clouds)")
	baseURL       = flag.String("base", envOr("OPENAI_BASE_URL", "http://localhost:11434/v1"), "OpenAI-compatible base URL")
	batchSize     = flag.Int("batch", 16, "Batch size (texts per request)")
	temp          = flag.Float64("temp", 0.0, "Sampling temperature")
	includeAlt    = flag.Bool("include-alt", false, "Also translate <img alt=...>")
	timeoutSecs   = flag.Int("timeout", 150, "HTTP timeout in seconds")
	workdir       = flag.String("workdir", "", "Working directory for resume/snapshots (default: .epubtrans/<stem>)")
	resume        = flag.Bool("resume", false, "Resume from previous run using state in workdir")
	snapshotEvery = flag.Int("snapshot-every", 5, "Write a partial EPUB snapshot every N finished HTML files (0=off)")
	quiet         = flag.Bool("quiet", false, "Less verbose progress")
)

// ===== Utils =====
func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
func clampBatch(n int) int {
	if n < 1 {
		return 1
	}
	return n
}
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
func pct(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return (float64(a) * 100) / float64(b)
}
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

// ===== LLM client =====
const systemPrompt = "You are a professional literary translator. Translate the given items into the target language while preserving meaning, tone, and inline punctuation. For learning, if target language is Japanese, keep difficulty around JLPT N2 to N1 level. If there are people's names, keep the original (do not translate). Output ONLY a JSON array of strings in the same order; no commentary. Return strictly JSON like: [\"訳文1\", \"訳文2\"]."

func userPrompt(src, tgt string, items []string) string {
	itemsJSON, _ := json.Marshal(items)
	return fmt.Sprintf(
		"Target language: %s.\nSource language hint: %s. (If 'auto', detect.)\nNotes:\n- Keep numbers and inline punctuation intact.\n- Preserve semantics and tone.\nItems to translate (JSON array):\n%s",
		tgt, src, string(itemsJSON),
	)
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}
type chatRequest struct {
	Model       string        `json:"model"`
	Temperature float64       `json:"temperature"`
	Messages    []chatMessage `json:"messages"`
}
type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

// translateBatch calls an OpenAI-compatible /chat/completions endpoint.
// Rebuilds the request each attempt; retries 408/429/5xx with exponential backoff and visible logs.
func translateBatch(httpc *http.Client, items []string) ([]string, error) {
	payload := chatRequest{
		Model:       *model,
		Temperature: *temp,
		Messages: []chatMessage{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userPrompt(*srcLang, *tgtLang, items)},
		},
	}
	buf, _ := json.Marshal(payload)
	url := strings.TrimRight(*baseURL, "/") + "/chat/completions"

	var lastErr error
	for attempt := 1; attempt <= 3; attempt++ {
		if attempt > 1 {
			wait := time.Duration(1<<uint(min(attempt-1, 4))) * time.Second
			fmt.Fprintf(os.Stderr, "[info] retry %d in %v\n", attempt, wait)
			time.Sleep(wait)
		}

		ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*timeoutSecs)*time.Second)
		req, _ := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(buf))
		req.Header.Set("Content-Type", "application/json")
		if ak := strings.TrimSpace(*apiKey); ak != "" {
			req.Header.Set("Authorization", "Bearer "+ak)
		}

		resp, err := httpc.Do(req)
		if err != nil {
			cancel()
			// retry on timeout
			if errors.Is(err, context.DeadlineExceeded) {
				lastErr = fmt.Errorf("deadline exceeded: %w", err)
				continue
			}
			var nerr net.Error
			if errors.As(err, &nerr) && nerr.Timeout() {
				lastErr = fmt.Errorf("timeout: %w", err)
				continue
			}
			lastErr = err
			continue
		}

		b, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		cancel()

		status := resp.StatusCode
		if status == 408 || status == 429 || (status >= 500 && status <= 599) {
			lastErr = fmt.Errorf("retryable http %d: %s", status, truncate(string(b), 512))
			continue
		}
		if status >= 300 {
			return nil, fmt.Errorf("http %d: %s", status, truncate(string(b), 512))
		}

		var cr chatResponse
		if err := json.Unmarshal(b, &cr); err != nil {
			return nil, fmt.Errorf("bad json: %w", err)
		}
		if len(cr.Choices) == 0 {
			return nil, errors.New("no choices from LLM")
		}
		content := strings.TrimSpace(cr.Choices[0].Message.Content)

		// strip <think>...</think>
		for {
			start := strings.Index(content, "<think>")
			end := strings.Index(content, "</think>")
			if start != -1 && end != -1 && end > start {
				content = content[:start] + content[end+len("</think>"):]
			} else {
				break
			}
		}
		content = strings.TrimSpace(content)

		var out []string
		if err := json.Unmarshal([]byte(content), &out); err != nil {
			// try to salvage a JSON array
			start := strings.Index(content, "[")
			end := strings.LastIndex(content, "]")
			if start != -1 && end != -1 && end > start {
				arr := content[start : end+1]
				if err2 := json.Unmarshal([]byte(arr), &out); err2 != nil {
					trimmed := strings.Trim(arr, "[]\"")
					if strings.Contains(trimmed, "\n") {
						out = strings.Split(trimmed, "\n")
					} else {
						out = []string{trimmed}
					}
				}
			} else {
				trimmed := strings.Trim(content, "[]\"")
				if strings.Contains(trimmed, "\n") {
					out = strings.Split(trimmed, "\n")
				} else {
					out = []string{trimmed}
				}
			}
		}
		return out, nil
	}
	return nil, lastErr
}

// ===== HTML traversal =====
var excludedTags = map[string]bool{"code": true, "pre": true, "script": true, "style": true}

type textTarget interface {
	Get() string
	Set(string)
}
type nodeText struct{ n *htmlpkg.Node }

func (t nodeText) Get() string  { return t.n.Data }
func (t nodeText) Set(s string) { t.n.Data = s }

type attrText struct {
	n   *htmlpkg.Node
	key string
}

func (t attrText) Get() string {
	for i := range t.n.Attr {
		if t.n.Attr[i].Key == t.key {
			return t.n.Attr[i].Val
		}
	}
	return ""
}
func (t attrText) Set(s string) {
	for i := range t.n.Attr {
		if t.n.Attr[i].Key == t.key {
			t.n.Attr[i].Val = s
			return
		}
	}
	t.n.Attr = append(t.n.Attr, htmlpkg.Attribute{Key: t.key, Val: s})
}

func collectTextTargets(root *htmlpkg.Node, includeAlt bool) []textTarget {
	var targets []textTarget
	var walk func(*htmlpkg.Node, *htmlpkg.Node)
	walk = func(n, parent *htmlpkg.Node) {
		if includeAlt && n.Type == htmlpkg.ElementNode && strings.EqualFold(n.Data, "img") {
			targets = append(targets, attrText{n: n, key: "alt"})
		}
		if n.Type == htmlpkg.TextNode {
			if parent != nil && parent.Type == htmlpkg.ElementNode && excludedTags[strings.ToLower(parent.Data)] {
				return
			}
			trim := strings.TrimSpace(n.Data)
			if trim == "" || onlyPunctOrSpace(trim) {
				return
			}
			targets = append(targets, nodeText{n: n})
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c, n)
		}
	}
	walk(root, nil)
	return targets
}
func onlyPunctOrSpace(s string) bool {
	for _, r := range s {
		if !(r == ' ' || r == '\n' || r == '\t' || strings.ContainsRune(".,:;!?()[]{}'\"-–—/\\", r)) {
			return false
		}
	}
	return true
}

// ===== Resume state =====
type stateFile struct {
	InputEPUB  string         `json:"input_epub"`
	OutputEPUB string         `json:"output_epub"`
	Workdir    string         `json:"workdir"`
	Model      string         `json:"model"`
	BaseURL    string         `json:"base_url"`
	From       string         `json:"from"`
	To         string         `json:"to"`
	IncludeAlt bool           `json:"include_alt"`
	Batch      int            `json:"batch"`
	HTMLFiles  []string       `json:"html_files"`
	Offsets    map[string]int `json:"offsets"`
	Totals     map[string]int `json:"totals"`
	TotalAll   int            `json:"total_all"`
	DoneAll    int            `json:"done_all"`
	FailedAll  int            `json:"failed_all"`
	StartedAt  time.Time      `json:"started_at"`
	UpdatedAt  time.Time      `json:"updated_at"`
}

func (s *stateFile) save(path string) error {
	s.UpdatedAt = time.Now()
	b, _ := json.MarshalIndent(s, "", "  ")
	return os.WriteFile(path, b, 0644)
}

// ===== EPUB helpers =====
func copyEPUBToWorkdir(r *zip.ReadCloser, wd string) error {
	for _, f := range r.File {
		dst := filepath.Join(wd, f.Name)

		// If this is META-INF but a file, skip it (we'll enforce directory)
		if f.Name == "META-INF" {
			fmt.Fprintln(os.Stderr, "[warn] EPUB has invalid META-INF as file, ignoring")
			continue
		}
		// Directory entry?
		if strings.HasSuffix(f.Name, "/") {
			if err := os.MkdirAll(dst, 0755); err != nil {
				return fmt.Errorf("mkdir: %w", err)
			}
			continue
		}
		// Ensure parent dirs exist
		if err := os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
			return fmt.Errorf("mkdir: %w", err)
		}

		rc, err := f.Open()
		if err != nil {
			return fmt.Errorf("open %s: %w", f.Name, err)
		}
		b, err := io.ReadAll(rc)
		rc.Close()
		if err != nil {
			return fmt.Errorf("read %s: %w", f.Name, err)
		}

		// Normalize mimetype content
		if f.Name == "mimetype" {
			b = []byte("application/epub+zip")
		}
		if err := os.WriteFile(dst, b, 0644); err != nil {
			return fmt.Errorf("write %s: %w", dst, err)
		}
	}
	// Ensure META-INF/ exists (even if missing or bogus in source)
	_ = os.MkdirAll(filepath.Join(wd, "META-INF"), 0755)
	return nil
}

func makeEPUBFromDir(dir, outPath string) error {
	// Collect files
	var files []string
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		rel, _ := filepath.Rel(dir, path)
		files = append(files, rel)
		return nil
	})
	if err != nil {
		return err
	}

	// Ensure 'mimetype' exists and is written FIRST (stored)
	sort.Strings(files)
	hasMimetype := false
	for _, f := range files {
		if f == "mimetype" {
			hasMimetype = true
			break
		}
	}
	if !hasMimetype {
		if err := os.WriteFile(filepath.Join(dir, "mimetype"), []byte("application/epub+zip"), 0644); err != nil {
			return fmt.Errorf("ensure mimetype: %w", err)
		}
		files = append(files, "mimetype")
		sort.Strings(files)
	}

	f, err := os.Create(outPath)
	if err != nil {
		return err
	}
	zw := zip.NewWriter(f)

	// 1) 'mimetype' first, stored, exact content
	{
		hdr := zip.FileHeader{Name: "mimetype"}
		hdr.SetMode(0644)
		hdr.Method = zip.Store
		w, err := zw.CreateHeader(&hdr)
		if err != nil {
			_ = zw.Close()
			_ = f.Close()
			return err
		}
		if _, err := w.Write([]byte("application/epub+zip")); err != nil {
			_ = zw.Close()
			_ = f.Close()
			return err
		}
	}

	// 2) other files deflated
	for _, rel := range files {
		if rel == "mimetype" {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, rel))
		if err != nil {
			_ = zw.Close()
			_ = f.Close()
			return err
		}
		hdr := zip.FileHeader{Name: rel}
		hdr.SetMode(0644)
		hdr.Method = zip.Deflate
		w, err := zw.CreateHeader(&hdr)
		if err != nil {
			_ = zw.Close()
			_ = f.Close()
			return err
		}
		if _, err := w.Write(b); err != nil {
			_ = zw.Close()
			_ = f.Close()
			return err
		}
	}

	if err := zw.Close(); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}

// ===== main =====
func main() {
	flag.Parse()
	if *inPath == "" || *tgtLang == "" {
		fmt.Printf("epub-translator-cli v%s\n", version)
		flag.Usage()
		os.Exit(2)
	}
	*batchSize = clampBatch(*batchSize)

	// output name
	out := *outPath
	stem := strings.TrimSuffix(filepath.Base(*inPath), filepath.Ext(*inPath))
	if out == "" {
		out = stem + "." + *tgtLang + ".epub"
	}

	// workdir
	wd := *workdir
	if wd == "" {
		wd = filepath.Join(".epubtrans", stem)
	}
	if err := os.MkdirAll(wd, 0755); err != nil {
		log.Fatalf("workdir: %v", err)
	}
	statePath := filepath.Join(wd, "state.json")

	// Open source EPUB
	r, err := zip.OpenReader(*inPath)
	if err != nil {
		log.Fatalf("open epub: %v", err)
	}
	defer r.Close()

	// Pre-scan: determine HTML files and totals
	htmlFiles := make([]string, 0, len(r.File))
	totals := map[string]int{}
	totalSegments := 0
	for _, f := range r.File {
		name := f.Name
		lower := strings.ToLower(name)
		if strings.HasSuffix(lower, ".xhtml") || strings.HasSuffix(lower, ".html") || strings.HasSuffix(lower, ".htm") {
			rc, err := f.Open()
			if err != nil {
				log.Fatalf("read entry: %v", err)
			}
			b, err := io.ReadAll(rc)
			rc.Close()
			if err != nil {
				log.Fatalf("read body: %v", err)
			}
			doc, perr := htmlpkg.Parse(bytes.NewReader(b))
			if perr == nil {
				ts := collectTextTargets(doc, *includeAlt)
				totals[name] = len(ts)
				totalSegments += len(ts)
			} else {
				totals[name] = 0
			}
			htmlFiles = append(htmlFiles, name)
		}
	}
	sort.Strings(htmlFiles)

	// Prepare workdir contents (idempotent)
	if !*resume {
		if err := copyEPUBToWorkdir(r, wd); err != nil {
			log.Fatalf("prime workdir: %v", err)
		}
	} else {
		empty := true
		_ = filepath.Walk(wd, func(_ string, info os.FileInfo, _ error) error {
			if info != nil && !info.IsDir() {
				empty = false
			}
			return nil
		})
		if empty {
			if err := copyEPUBToWorkdir(r, wd); err != nil {
				log.Fatalf("prime workdir: %v", err)
			}
		}
	}

	// Load or initialize state
	st := &stateFile{
		InputEPUB:  *inPath,
		OutputEPUB: out,
		Workdir:    wd,
		Model:      *model,
		BaseURL:    *baseURL,
		From:       *srcLang,
		To:         *tgtLang,
		IncludeAlt: *includeAlt,
		Batch:      *batchSize,
		HTMLFiles:  htmlFiles,
		Offsets:    map[string]int{},
		Totals:     totals,
		TotalAll:   totalSegments,
		DoneAll:    0,
		FailedAll:  0,
		StartedAt:  time.Now(),
		UpdatedAt:  time.Now(),
	}
	if *resume {
		if b, err := os.ReadFile(statePath); err == nil {
			_ = json.Unmarshal(b, st)
			// warn on parameter mismatch to avoid confusing resumes
			if st.InputEPUB != *inPath || st.Model != *model || st.BaseURL != *baseURL ||
				st.To != *tgtLang || st.From != *srcLang || st.IncludeAlt != *includeAlt {
				fmt.Fprintf(os.Stderr, "[warn] resume parameters differ from saved state (input/model/base/from/to/includeAlt). Consider restarting without -resume or repeat previous flags.\n")
			}
		}
	}
	if st.Offsets == nil {
		st.Offsets = map[string]int{}
	}
	if st.Totals == nil {
		st.Totals = totals
	}
	_ = st.save(statePath)

	// Graceful Ctrl-C: snapshot & save state
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-stop
		fmt.Println("\nSignal caught → snapshot & save state …")
		_ = st.save(statePath)
		partial := fmt.Sprintf("%s.partial.%s.epub", out, time.Now().Format("20060102-150405"))
		if err := makeEPUBFromDir(wd, partial); err != nil {
			fmt.Fprintf(os.Stderr, "[warn] snapshot error: %v\n", err)
		} else {
			fmt.Printf("[intermediate] Saved partial EPUB: %s\n", partial)
		}
		os.Exit(130)
	}()

	httpc := &http.Client{Timeout: time.Duration(*timeoutSecs) * time.Second}
	translatedFiles := 0

	// Translate each HTML in deterministic order
	for _, name := range st.HTMLFiles {
		inp := filepath.Join(wd, name)
		b, err := os.ReadFile(inp)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[warn] cannot read %s: %v (skipping)\n", name, err)
			continue
		}
		doc, perr := htmlpkg.Parse(bytes.NewReader(b))
		if perr != nil {
			fmt.Fprintf(os.Stderr, "[warn] parse failed for %s: %v (keeping original)\n", name, perr)
			continue
		}

		targets := collectTextTargets(doc, st.IncludeAlt)
		fileTotal := len(targets)
		off := st.Offsets[name]
		if off > fileTotal {
			off = fileTotal
		}

		for off < fileTotal {
			end := min(off+st.Batch, fileTotal)

			// Gather batch texts
			batch := make([]string, end-off)
			for j := off; j < end; j++ {
				batch[j-off] = targets[j].Get()
			}

			outTexts, err := translateBatch(httpc, batch)
			applied := 0
			if err != nil {
				// continue; leave originals; count failures; skip this window
				st.FailedAll += (end - off)
				fmt.Fprintf(os.Stderr, "[warn] translate batch failed at %s [%d:%d): %v\n", name, off, end, err)
				off = end
			} else {
				n := min(len(outTexts), end-off)
				for j := 0; j < n; j++ {
					txt := strings.TrimSpace(outTexts[j])
					if txt == "" {
						// don't apply empty salvage; count as failed
						st.FailedAll++
						continue
					}
					targets[off+j].Set(outTexts[j])
					st.DoneAll++
					applied++
				}
				if n < (end - off) {
					miss := (end - off) - n
					st.FailedAll += miss
					fmt.Fprintf(os.Stderr, "[warn] model returned %d/%d items at %s [%d:%d)\n", n, end-off, name, off, end)
				}
				// If nothing applied, avoid infinite loop by skipping this window
				if applied == 0 {
					off = end
				} else {
					off += applied // retry remainder next iteration
				}
			}

			// Flush HTML to disk after each batch
			var buf bytes.Buffer
			if err := htmlpkg.Render(&buf, doc); err != nil {
				fmt.Fprintf(os.Stderr, "[warn] render failed for %s batch up to %d: %v (keeping previous)\n", name, off, err)
			} else {
				if err := os.MkdirAll(filepath.Dir(inp), 0755); err != nil {
					log.Fatalf("mkdir: %v", err)
				}
				if err := os.WriteFile(inp, buf.Bytes(), 0644); err != nil {
					log.Fatalf("write translated: %v", err)
				}
			}

			// Update per-file offset & persist state AFTER successful flush
			st.Offsets[name] = off
			_ = st.save(statePath)

			if !*quiet {
				fmt.Printf("[%s] %d/%d (%.1f%%), overall: %d/%d (%.1f%%)\n",
					name, off, fileTotal, pct(off, fileTotal),
					st.DoneAll, st.TotalAll, pct(st.DoneAll, max(1, st.TotalAll)))
			}
			time.Sleep(120 * time.Millisecond)
		}

		translatedFiles++
		// Periodic snapshot
		if *snapshotEvery > 0 && translatedFiles%*snapshotEvery == 0 {
			partial := fmt.Sprintf("%s.partial.%s.epub", out, time.Now().Format("20060102-150405"))
			if err := makeEPUBFromDir(wd, partial); err == nil {
				if !*quiet {
					fmt.Printf("[intermediate] Saved partial EPUB: %s\n", partial)
				}
			} else {
				fmt.Fprintf(os.Stderr, "[warn] failed to write partial EPUB: %v\n", err)
			}
		}
	}

	// Final EPUB
	if err := makeEPUBFromDir(wd, out); err != nil {
		log.Fatalf("final EPUB: %v", err)
	}
	fmt.Printf("Done. Translated %d segments, %d failed (see stderr warnings for details).\n", st.DoneAll, st.FailedAll)
	fmt.Printf("Output saved to: %s\n", out)
}
