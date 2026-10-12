
CREATE INDEX voices_embedding_hnsw_idx
    ON voices
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
