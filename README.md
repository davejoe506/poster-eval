# PosterEval

**Off-policy evaluation for streaming artwork personalization.**

Evaluate new poster recommendation models against historical user data prior to live deployment.

[Walkthrough notebook](notebooks/01_postereval_walkthrough.ipynb) · [API](src/api/main.py)

---

## Project Context

Streaming platforms use something called a contextual bandit to select the visual artwork that accompanies each movie or TV title for a given user. A contextual bandit is a system that performs an action (i.e., showing movie posters) based on some sort of continuously-updated context (i.e., user genre preferences, streaming device, title popularity, etc.). 

Contextual bandits are great at adjusting what is shown based on a dynamic, always-evolving user behavior. However, evaluating new contextual bandit systems through online A/B testing can be slow, costly, and risks user engagement if the untested policy underperforms.

Offline policy evaluation (OPE) aims to solve this evaluation issue by reweighting historical data by propensity, or the probability the current, baseline poster policy would show the artwork that was actually shown. The reweighting uses the ratio of how likely the new policy would have been to show that poster versus how likely the old policy was to show it. That ratio lets you compare the current, control poster recommendation policy to a new, treatment poster recommendation policy on a more apples-to-apples basis. This lets streaming platforms test and weed out underperforming policies offline without wasting engineering resources or risking user engagement on costly online experiments.

This project simulates historical poster impressions: which artwork was shown, whether it got a click, and how likely the current policy was to show it. It uses those logs to estimate a user’s click rate under the new policy, then checks these estimates on randomly shown posters. An automated pipeline, orchestrated through Airflow, generates the logs, fits a propensity model, runs the offline click rate estimators, validates the estimators against a holdout dataset, and writes results that can be inspected through an API.

---



## Tech Stack

- **Language & Core:** Python 3.11+, SQL
- **Data Processing & OPE:** NumPy, Pandas (IPS, SNIPS, Doubly Robust estimation)
- **Modeling:** Scikit-Learn, XGBoost (Propensity scoring & reward modeling)
- **Orchestration:** Apache Airflow
- **API & Serving:** FastAPI, Uvicorn
- **Packaging & Storage:** Docker Compose, Parquet, SQLite

---



## Installation & Setup

**1. Prerequisites**

- Docker Desktop (recommended path: Airflow + API)
- Python 3.11 or higher (only if you skip Docker)

**2. Environment Setup**

Clone the repo:

```bash
git clone https://github.com/davejoe506/poster-eval.git
cd poster-eval
```

**3. Configuration**

Copy the example env file. Default Airflow login is `admin` / `admin`:

```bash
cp .env.example .env
# Windows: copy .env.example .env
```

**4. Start the Stack**

Launch Airflow and the API:

```bash
docker compose up --build
```


| Service  | URL                                                      |
| -------- | -------------------------------------------------------- |
| Airflow  | [http://localhost:8080](http://localhost:8080)           |
| API docs | [http://localhost:8000/docs](http://localhost:8000/docs) |


The API reads `data/processed/postereval.db` after a pipeline run. Triggering the DAG is under **Running the Pipeline**.

**5. Local Python (optional)**

Use this if you want to run the walkthrough notebook or pytest without Compose:

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

```bash
python -m pytest tests -q
```

Generating logs from the command line is under **Running the Pipeline**.

---



## Project Structure

```
poster-eval/
├── dags/                      # Airflow pipeline
├── docker-compose.yml         # Local Airflow + API
├── Dockerfile
├── notebooks/                 # Walkthrough
├── requirements.txt
├── requirements-api.txt
├── scripts/                   # Airflow container entrypoint
├── sql/schema.sql             # Reference table definitions (not applied at runtime)
├── src/api/                   # FastAPI
├── src/simulator/             # Simulated show/click logs
├── src/propensity/            # Model: chance the old policy shows a poster
├── src/ope/                   # IPS, SNIPS, Doubly Robust
├── src/validation/            # Random poster holdout check
├── src/pipeline/              # Functions the Airflow tasks call
├── src/reporting/             # Bias / RMSE helpers
└── tests/
```

---



## Data Pipeline

The data pipeline works as follows:

1. **Simulate:** contextual bandit artwork impression data derived using epsilon-greedy mixture propensity $\pi_e(a \mid x) = \varepsilon/K + (1-\varepsilon)\pi_{\text{softmax}}(a \mid x)$.
2. **Train:** P(a $\mid$ x) XGBoost model predicting propensities from context data in case propensities are not simulated and/or not readily available in production data.
3. **Estimate:** candidate policy click rate with IPS, SNIPS, and Doubly Robust (using a click model on the poster that was actually shown).
4. **Validate:** against a random holdout dataset; segment by title popularity (`head`/`mid`/`tail`).
5. **Orchestrate:** with Airflow; display results using FastAPI.

---



## Evaluation & Validation Results

The pipeline is evaluated on simulated data not to claim real-world production metrics, but to verify the OPE measurement tools against known (albeit fake) values that were generated from defined probabilities. This setup is a check that the propensity model and click-rate estimators are implemented correctly, not a stand-in for live user traffic.

The candidate model, evaluated on normal, non-holdout impression data, is benchmarked against an unbiased, holdout group that is considered ground-truth (`is_randomized=True`, $n=2027$, $\pi_e = 1/K$):


| Estimator | Candidate Estimated Click Rate | Holdout Ground Truth | Absolute Bias |
| --------- | ------------------------------ | -------------------- | ------------- |
| **IPS**   | 0.4265                         | 0.4129               | 0.0136        |
| **SNIPS** | 0.4279                         | 0.4129               | 0.0150        |
| **DR**    | 0.4274                         | 0.4129               | 0.0145        |


The three estimators agree closely and sit within 1.5 percentage points of the holdout. This is a check that the simulation and the click-rate math are working as intended, but it's not enough to roll out the new policy to live users.

---



## Running the Pipeline

**Docker**

1. Open [http://localhost:8080](http://localhost:8080) and sign in with `admin` / `admin`.
2. Turn on `postereval_pipeline` and trigger it.
3. When the run is green, open [http://localhost:8000/docs](http://localhost:8000/docs) and try `GET /policies/compare`.

**No Docker**

These commands generate logs and write a JSON evaluation report, but do not create the SQLite file the API reads.

```bash
python -m src.simulator.generate_logs --output data/processed/bandit_logs.parquet
python -m src.ope.evaluate --logs data/processed/bandit_logs.parquet --output data/processed/ope_report.json
```

The FastAPI endpoints below read `data/processed/postereval.db`, which is written when `postereval_pipeline` is triggered in Airflow. Start the `/docs` page after that pipeline run is green.


| Method | Path                  | What You Get                           |
| ------ | --------------------- | -------------------------------------- |
| `GET`  | `/health`             | Whether the SQLite file is there       |
| `GET`  | `/policies/compare`   | IPS, SNIPS, DR vs the current policy   |
| `GET`  | `/validation/holdout` | Holdout check and bias                 |
| `GET`  | `/reports/stratified` | Same scores by `head` / `mid` / `tail` |


Optional: `?policy_id=bandit_v2`.

Example response from `GET /policies/compare` (illustrating a 20k-row local run):

```json
{
  "policy_id": "bandit_v2",
  "eval_timestamp": "2026-08-25T15:22:40Z",
  "estimators": {
    "IPS":  {"value": 0.427, "std_error": 0.004},
    "SNIPS": {"value": 0.426, "std_error": null},
    "DR":   {"value": 0.426, "std_error": 0.004}
  },
  "logging_policy_value": 0.424,
  "lift_vs_logging": 0.002,
  "recommendation": "DEPLOY_WITH_CAUTION",
  "ground_truth_value": 0.413
}
```

This example run shows that the Airflow pipeline and FastAPI endpoints run end-to-end. In this example, the new policy provides a 0.2 percentage point lift in click rate above the old one (0.426 vs 0.424), but the lift is within the estimator's standard error (0.004). Instead of automatically approving the policy change, the API flags it as "DEPLOY_WITH_CAUTION", indicating that the lift would need to be more decisive to deploy into production with full confidence.

---



## Challenges, Current Limitations & Future Work



### Challenges & Current Limitations

- **Synthetic Policy Lift:** Because user impression and context data are simulated in this project, estimated lift metrics are used to validate pipeline execution rather than to reflect true business impact. Evaluating the system on production logs or real-world open datasets would provide much more meaningful recommendations and business insights.
- **Missing Production Propensities:** When the logging impression data excludes the probability of showing artwork (often the case in production data), estimating it with a propensity model adds an extra step. Logging the exact propensities when the poster is shown would be the best approach, as estimating probabilities after the fact carries more uncertainty.
- **Simplified No Click/Click Model:** The simulator assumes that a user immediately clicks play or does not click (0 or 1). In reality, user feedback is messier, as users might click minutes later, click away from titles early, or share an account with multiple family members.



### Future Work

- **Logging Propensities When Artwork Is Shown:** While the simulation component of this project provides exact logging propensities, streaming platforms ideally should record these propensities directly into logs as posters are displayed to remove the need to estimate them later in production.
- **Richer Engagement Models:** Expand beyond a simple no click/click model to track more intricate streaming behavior, such as delayed play starts or watch duration, in an effort to better measure genuine user interest.

