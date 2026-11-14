# 🏦 S-Bank AI Financial Coach (Backend API)

**Built for the S-Bank Challenge at Junction 2026 by Team Aegis
Shield.**

This repository contains the machine learning and conversational AI
backend for the S-Bank AI Financial Coach. The application empowers
users to understand their everyday spending habits, compare themselves
to demographic peers, and receive personalized, actionable financial
advice through a conversational interface.

------------------------------------------------------------------------

## 🎯 The Context: Why We Built This

S-Bank challenged us to build a solution that improves financial
literacy using customer transaction data.

Instead of confusing users with complex spreadsheets or static
dashboards, we built an **AI-driven coaching experience**. We
specifically excluded static data (like mortgages) to focus purely on
**Everyday Spending**---the area where users have the most flexibility
to change their habits.

The backend automatically categorizes user transactions into the
pedagogical **"Three Baskets"**:

1.  **Mandatory Regular** (Rent, bills)
2.  **Mandatory Irregular** (Emergencies, car repairs)
3.  **Everyday Spending** (Coffee, dining, shopping)

------------------------------------------------------------------------

## 🧠 System Architecture & Logic

This backend is powered by **Python and FastAPI**, acting as the bridge
between raw data, Machine Learning models, and a Large Language Model
(LLM).

1.  **Transaction Categorization (Supervised ML):** We trained a
    K-Nearest Neighbors (KNN) classifier (`basket_classifier.pkl`) on
    advanced mathematical features (coefficient of variation,
    transaction frequency, average volume) to automatically sort raw
    transactions into the Three Baskets.
2.  **Financial Twin Benchmarking (Unsupervised ML):** We built an
    unsupervised `NearestNeighbors` model (`financial_twin.pkl`) that
    maps users into a multi-dimensional space based on their age and
    total spending volume. It finds the user's 5 closest demographic
    "twins" to provide peer-to-peer benchmarking (e.g., *"You spend 10%
    less than people your age!"*).
3.  **Conversational AI (Groq API):** The mathematical outputs of the
    KNN models are injected into a hidden system prompt. We use the
    blazing-fast Groq API (`openai/gpt-oss-20b` model) to translate
    these hard numbers into friendly, plain-language financial advice
    with a memory of the chat history.

------------------------------------------------------------------------

## 🔗 Frontend Integration (The API Contract)

The frontend (React/Vercel) and backend (FastAPI/Render) are fully
decoupled. The frontend communicates with this backend via a single REST
endpoint.

-  **GitHub link** `https://github.com/TechieTad/s-coach-frontend/tree/main`
-   **Endpoint:** `POST /chat`
-   **Content-Type:** `application/json`

### What the Frontend Sends (Request)

The frontend must maintain the chat history and send the entire
conversation array on every request so the AI remembers the context.

``` json
{
  "customer_id": "CUST00001",
  "conversation": [
    { "role": "user", "content": "Where does most of my money go?" },
    { "role": "assistant", "content": "Most of it goes to Mandatory Regular expenses..." },
    { "role": "user", "content": "How can I cut back on my everyday spending?" }
  ]
}
```

### What the Backend Returns (Response)

``` json
{
  "reply": "To cut back, try brewing coffee at home 3 days a week! ☕ Here are some other tips...",
  "data_context": {
    "Everyday": 26631.53,
    "Mandatory Irregular": 11155.49,
    "Mandatory Regular": 75546.35
  },
  "benchmark": "The user spends 0.7% LESS than their 5 closest demographic twins (Avg twin spend: €114079.22)."
}
```

*The frontend uses `reply` for the chat UI, `data_context` to draw
Recharts pie/bar graphs, and `benchmark` to display a peer-comparison
badge.*

------------------------------------------------------------------------

## 💻 Local Development Setup

Want to run the API on your own machine? Follow these steps:

### 1. Clone the repository and enter the directory

``` bash
git clone <your-repo-url>
cd backend
```

### 2. Create and activate a virtual environment

``` bash
# Windows / WSL / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Set up your Environment Variables

Create a `.env` file in the root directory and add your Groq API key:

``` ini
GROQ_API_KEY=gsk_your_api_key_here
```

### 5. Start the FastAPI Server

``` bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. You can test it
interactively at `http://127.0.0.1:8000/docs`.

------------------------------------------------------------------------

## 🚀 Production Server Deployment (Render)

This API is designed to be hosted on free cloud tiers (like Render.com).
Because free tiers have strict memory limits (usually 512MB RAM), **we
do not process the massive raw CSV files on the server.**

Instead, the mathematical features and models are pre-computed locally
and saved as highly compressed `.pkl` files (`user_features.pkl`,
`basket_classifier.pkl`, `financial_twin.pkl`).

### To deploy to Render

1.  Ensure your `.gitignore` file includes `.env` and `data/` (so
    secrets and heavy CSVs are NOT pushed to GitHub).
2.  Ensure your `.pkl` files **ARE** pushed to GitHub.
3.  Connect the GitHub repository to a new **Render Web Service**.
4.  **Build Command:** `pip install -r requirements.txt`
5.  **Start Command:** `uvicorn main:app --host 0.0.0.0 --port 10000`
6.  Go to the "Environment Variables" tab in Render and add your
    `GROQ_API_KEY`.
7.  Deploy! The server will boot in seconds, load the lightweight `.pkl`
    files into memory, and begin serving requests.
