.PHONY: setup data up down logs replay features reset train parity bench drift api dashboard mlflow test lint format demo clean

# ── Setup ────────────────────────────────────────────────────────────────────
setup:            ## create venv + install all dependency groups
	uv sync --all-extras
	@echo "→ copy .env.example to .env and fill KAGGLE_USERNAME / KAGGLE_KEY"

data:             ## download IEEE-CIS dataset into data/
	uv run python scripts/download_data.py --source ieee

# ── Infrastructure ───────────────────────────────────────────────────────────
up:               ## start broker, redis, spark, prometheus, grafana
	docker compose up -d
	@echo "→ Redpanda console: http://localhost:8085  ·  Grafana: http://localhost:3000"

down:             ## stop infrastructure (keeps volumes)
	docker compose down

logs:             ## tail infra logs
	docker compose logs -f --tail=100

clean:            ## stop infra and delete volumes (fresh start)
	docker compose down -v

# ── Pipeline ─────────────────────────────────────────────────────────────────
replay:           ## replay dataset into transactions.raw (limit with REPLAY_LIMIT=1000)
	uv run python -m streamguard.producer.replay

features:         ## run the Spark Structured Streaming feature job (in container)
	docker compose up --build spark-features

reset:            ## fresh stream: drop topics, Redis card hashes and Spark state (keeps infra up)
	-docker compose rm -sf spark-features
	-docker volume rm streamguard_spark-checkpoints
	-docker exec sg-redpanda rpk topic delete transactions.raw transactions.scored transactions.alerts
	docker exec sg-redis sh -c "redis-cli --scan --pattern 'sg:card:*' | xargs -r redis-cli del" >/dev/null

train:            ## train XGBoost + AE, log to MLflow, evaluate on temporal test
	uv run python -m streamguard.models.train

parity:           ## online vs offline feature parity check (fails CI on drift)
	uv run python -m streamguard.eval.parity

bench:            ## end-to-end latency benchmark → artifacts/bench.json
	uv run python -m streamguard.eval.latency

drift:            ## PSI drift report vs training reference
	uv run python -m streamguard.eval.drift

# ── Services (host) ──────────────────────────────────────────────────────────
api:              ## FastAPI scorer + alerts (expects infra up)
	uv run uvicorn streamguard.serving.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:        ## Streamlit alert dashboard (expects api up)
	uv run streamlit run dashboard/app.py

mlflow:           ## MLflow tracking server + registry on :5000
	uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlruns

# ── Quality ──────────────────────────────────────────────────────────────────
test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

demo:             ## the demo sequence (run each in its own terminal)
	@echo "1) make up        2) make mlflow       3) make replay"
	@echo "4) make features  5) make api          6) make dashboard"
