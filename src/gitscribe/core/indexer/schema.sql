CREATE TABLE IF NOT EXISTS symbols (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('function', 'class', 'method', 'import')),
  file TEXT NOT NULL,
  lineno INTEGER NOT NULL,
  end_lineno INTEGER,
  parent TEXT,
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
