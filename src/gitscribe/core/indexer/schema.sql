CREATE TABLE IF NOT EXISTS symbols (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('function', 'class', 'method', 'import')),
  file TEXT NOT NULL,
  lineno INTEGER NOT NULL,
  end_lineno INTEGER,
  parent TEXT,
  file_hash TEXT,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);

CREATE TABLE IF NOT EXISTS edges (
  id INTEGER PRIMARY KEY,
  source_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
  target_id INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
  target_name TEXT NOT NULL,
  edge_type TEXT NOT NULL CHECK (edge_type IN ('calls', 'imports', 'inherits')),
  resolved BOOLEAN NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);

CREATE TABLE IF NOT EXISTS embeddings (
  symbol_id INTEGER PRIMARY KEY REFERENCES symbols(id) ON DELETE CASCADE,
  vector BLOB NOT NULL,
  model TEXT NOT NULL
);

-- Merkle incremental indexing (spec §2.3)
CREATE TABLE IF NOT EXISTS file_hashes (
  path TEXT PRIMARY KEY,
  content_hash TEXT NOT NULL,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS index_runs (
  id INTEGER PRIMARY KEY,
  root_hash TEXT NOT NULL,
  files_changed INTEGER,
  files_skipped INTEGER,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Code review findings (spec §3.1) — both lint and agentic passes write here
CREATE TABLE IF NOT EXISTS review_findings (
  id INTEGER PRIMARY KEY,
  symbol_id INTEGER,
  source TEXT NOT NULL CHECK (source IN ('lint', 'agentic')),
  severity TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'error')),
  rule_or_reason TEXT NOT NULL,
  message TEXT NOT NULL,
  line_start INTEGER,
  line_end INTEGER,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (symbol_id) REFERENCES symbols(id)
);
CREATE INDEX IF NOT EXISTS idx_review_findings_symbol ON review_findings(symbol_id);
CREATE INDEX IF NOT EXISTS idx_review_findings_source ON review_findings(source);
