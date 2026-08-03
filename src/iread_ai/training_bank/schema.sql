CREATE TABLE IF NOT EXISTS bank_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS learning_units (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_type TEXT NOT NULL CHECK (unit_type IN ('CONSONANT', 'VOWEL', 'SYLLABLE', 'WORD')),
    surface TEXT NOT NULL,
    spoken_text TEXT NOT NULL,
    pronunciation TEXT NOT NULL,
    onset TEXT,
    vowel TEXT,
    coda TEXT,
    difficulty INTEGER NOT NULL CHECK (difficulty BETWEEN 1 AND 5),
    familiarity INTEGER NOT NULL CHECK (familiarity BETWEEN 1 AND 5),
    trace_asset_key TEXT,
    source TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (unit_type, surface)
);

CREATE INDEX IF NOT EXISTS idx_learning_units_selection
    ON learning_units (unit_type, is_active, difficulty, familiarity);

CREATE TABLE IF NOT EXISTS learning_unit_features (
    learning_unit_id INTEGER NOT NULL,
    feature_code TEXT NOT NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 1 CHECK (occurrence_count > 0),
    is_verified INTEGER NOT NULL DEFAULT 1 CHECK (is_verified IN (0, 1)),
    PRIMARY KEY (learning_unit_id, feature_code),
    FOREIGN KEY (learning_unit_id) REFERENCES learning_units (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_learning_unit_feature_code
    ON learning_unit_features (feature_code, is_verified);

CREATE TABLE IF NOT EXISTS learning_unit_confusions (
    learning_unit_id INTEGER NOT NULL,
    confusing_unit_id INTEGER NOT NULL,
    confusion_type TEXT NOT NULL CHECK (confusion_type IN ('SOUND', 'SHAPE', 'STRUCTURE')),
    priority INTEGER NOT NULL DEFAULT 1 CHECK (priority BETWEEN 1 AND 5),
    PRIMARY KEY (learning_unit_id, confusing_unit_id, confusion_type),
    FOREIGN KEY (learning_unit_id) REFERENCES learning_units (id) ON DELETE CASCADE,
    FOREIGN KEY (confusing_unit_id) REFERENCES learning_units (id) ON DELETE CASCADE,
    CHECK (learning_unit_id <> confusing_unit_id)
);
