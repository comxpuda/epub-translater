# EPUB Translator CLI

A command-line tool to **translate EPUB files** using an OpenAI-compatible API (such as OpenAI, Ollama, or other LLM backends).
It extracts all translatable text from the EPUB, sends them in batches to a model, and rewrites the EPUB with translated content while preserving structure and formatting.

---

## Features

* 📚 Translate entire EPUB books between languages.
* 🌍 Auto-detect source language (or specify manually).
* 🔄 Resume interrupted translations using a working directory.
* 💾 Periodic partial EPUB snapshots for crash recovery.
* 🎨 Optionally translate `<img alt="...">` attributes.
* ⚡ Batch translation with configurable size.
* 🚦 Graceful Ctrl+C handling (saves progress before exit).
* 🔌 Works with OpenAI-compatible APIs (OpenAI, Ollama, custom deployments).

---

## Installation

Clone and build:

```bash
go build -o epub-translator main.go
```

---

## Usage

Basic command:

```bash
./epub-translator -in input.epub -to ja
```

This will:

* Take `input.epub`
* Translate it into **Japanese**
* Save the result as `input.ja.epub`

---

### Command-Line Options

| Flag              | Description                                                             |
| ----------------- | ----------------------------------------------------------------------- |
| `-in`             | Input `.epub` file (required)                                           |
| `-out`            | Output `.epub` path (default: `<stem>.<to>.epub`)                       |
| `-from`           | Source language code (`auto` for autodetect, default: `auto`)           |
| `-to`             | Target language code (required)                                         |
| `-model`          | Model name (default: `qwen2.5:14b`)                                     |
| `-apikey`         | API key for OpenAI/compatible providers (default: env `OPENAI_API_KEY`) |
| `-base`           | OpenAI-compatible base URL (default: `http://localhost:11434/v1`)       |
| `-batch`          | Batch size (texts per request, default: 16)                             |
| `-temp`           | Sampling temperature (default: `0.0`)                                   |
| `-include-alt`    | Also translate `<img alt=...>` attributes                               |
| `-timeout`        | HTTP timeout in seconds (default: 100)                                  |
| `-workdir`        | Working directory for state/snapshots (default: `.epubtrans/<stem>`)    |
| `-resume`         | Resume from previous run using state in `workdir`                       |
| `-snapshot-every` | Write partial EPUB every N HTML files (default: 5, `0` = off)           |
| `-quiet`          | Reduce verbosity                                                        |

---

### Examples

#### Translate an EPUB to Japanese

```bash
./epub-translator -in novel.epub -to ja
```

#### Translate with Ollama locally

```bash
OLLAMA_HOST=localhost:11434 \
./epub-translator -in novel.epub -to en -model qwen2.5:14b -base http://localhost:11434/v1
```

#### Resume after interruption

```bash
./epub-translator -in novel.epub -to fr -resume
```

#### Translate and include `<img alt>` text

```bash
./epub-translator -in textbook.epub -to de -include-alt
```

---

## Output

* Final translated EPUB:

  ```
  <stem>.<to>.epub
  ```

  Example: `novel.ja.epub`

* Partial EPUB snapshots (if enabled):

  ```
  <stem>.<to>.epub.partial.epub
  ```

* Resume state (JSON file):

  ```
  .epubtrans/<stem>/state.json
  ```

---

## Requirements

* Go 1.20+
* An OpenAI-compatible API (OpenAI, Ollama, vLLM, etc.)

