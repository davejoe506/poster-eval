-- PosterEval bandit log schema (PostgreSQL dialect; SQLite-compatible with minor tweaks)

CREATE TABLE IF NOT EXISTS titles (
    title_id        INTEGER PRIMARY KEY,
    title_name      VARCHAR(255) NOT NULL,
    genres          TEXT NOT NULL,           -- JSON array of genre strings
    popularity_score DOUBLE PRECISION NOT NULL,
    popularity_tier  VARCHAR(10) NOT NULL CHECK (popularity_tier IN ('head', 'mid', 'tail'))
);

CREATE TABLE IF NOT EXISTS artwork_variants (
    artwork_id      INTEGER PRIMARY KEY,
    title_id        INTEGER NOT NULL REFERENCES titles(title_id),
    variant_index   INTEGER NOT NULL,
    poster_path     VARCHAR(512)
);

CREATE TABLE IF NOT EXISTS user_contexts (
    user_id             INTEGER PRIMARY KEY,
    genre_affinity      TEXT NOT NULL,       -- JSON object genre -> weight
    device_type         VARCHAR(10) NOT NULL CHECK (device_type IN ('mobile', 'tv', 'web')),
    account_tenure_days INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS bandit_logs (
    impression_id       BIGINT PRIMARY KEY,
    timestamp           TIMESTAMP NOT NULL,
    user_id             INTEGER NOT NULL REFERENCES user_contexts(user_id),
    title_id            INTEGER NOT NULL REFERENCES titles(title_id),
    artwork_id          INTEGER NOT NULL REFERENCES artwork_variants(artwork_id),
    variant_index       INTEGER NOT NULL,
    genre_emphasis      VARCHAR(32),
    context_features    TEXT NOT NULL,       -- JSON
    logging_propensity  DOUBLE PRECISION NOT NULL CHECK (logging_propensity > 0),
    reward              DOUBLE PRECISION NOT NULL CHECK (reward IN (0, 1)),
    logging_policy      VARCHAR(64) NOT NULL,
    is_randomized       BOOLEAN NOT NULL DEFAULT FALSE,
    popularity_tier     VARCHAR(10) NOT NULL
);

CREATE TABLE IF NOT EXISTS policy_evaluations (
    eval_id             SERIAL PRIMARY KEY,
    eval_timestamp      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    candidate_policy    VARCHAR(64) NOT NULL,
    estimator           VARCHAR(16) NOT NULL CHECK (estimator IN ('IPS', 'SNIPS', 'DR')),
    estimated_value     DOUBLE PRECISION NOT NULL,
    std_error           DOUBLE PRECISION,
    popularity_tier     VARCHAR(10),
    ground_truth_value  DOUBLE PRECISION,
    estimation_error    DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_bandit_logs_randomized ON bandit_logs(is_randomized);
CREATE INDEX IF NOT EXISTS idx_bandit_logs_tier ON bandit_logs(popularity_tier);
