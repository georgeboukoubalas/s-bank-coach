# S-Bank Coach Backend

FastAPI backend for the S-Bank AI Financial Coach. It uses scikit-learn models
to summarize spending and find peer benchmarks, and the OpenAI-compatible Groq
API for chat responses.

## Setup

Use Python 3.12 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set `GROQ_API_KEY` in your environment or in a local `.env` file. Never commit
that file.

## Local data and models

Customer and transaction CSVs and trained model files are intentionally excluded
from this repository because they may contain sensitive financial information.
For local use, provide the CSVs expected by `data_loader.py` in `data/`:

- `tx_customers_junction.csv`
- `tx_junction_transactions.csv`

Train the model artifacts before starting the API:

```bash
python train_model.py
python train_twin.py
```

These commands create `basket_classifier.pkl` and `financial_twin.pkl`; both
are ignored by Git. Run commands from this directory so the relative data paths
resolve correctly.

## Run the API

```bash
uvicorn main:app --reload
```
