
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE voices (
    id UUID PRIMARY KEY,
    embedding VECTOR(512) NOT NULL,
    model_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE persons (
    id UUID PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    middle_name TEXT,
    voice_id UUID NOT NULL UNIQUE
        REFERENCES voices(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE vector_index_config (
    id SMALLINT PRIMARY KEY CHECK (id = 1),
    embedding_dimension INTEGER NOT NULL
        CHECK (embedding_dimension = 512),
    model_version TEXT NOT NULL,
    distance_metric TEXT NOT NULL
        CHECK (distance_metric = 'cosine')
);
