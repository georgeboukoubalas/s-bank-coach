from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import openai
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import List

class ChatMessage(BaseModel):
    role: str      # "user" or "assistant"
    content: str   # The text of the message

class ChatRequest(BaseModel):
    customer_id: str
    conversation: List[ChatMessage]  # The frontend sends the whole history

# 1. Setup API Keys
load_dotenv()

# Initialize the Groq client
client = openai.OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

app = FastAPI()

# Add this immediately after creating the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (perfect for hackathons)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# 2. Load BOTH ML Models at startup
print("Booting up AI Coach Backend...")
try:
    basket_model = joblib.load("basket_classifier.pkl")
    
    twin_bundle = joblib.load("financial_twin.pkl")
    twin_model = twin_bundle['model']
    twin_scaler = twin_bundle['scaler']
    twin_db = twin_bundle['user_data']
    
    # NEW CODE: Just load the pre-calculated features!
    all_user_features = joblib.load("user_features.pkl")
    
    print("Models and Data loaded into memory.")
except Exception as e:
    print(f"Startup warning: {e}")

def get_user_baskets(customer_id: str):
    """Filters the user's data, runs it through KNN, and sums up the baskets."""
    user_data = all_user_features[all_user_features['customer_id'] == customer_id].copy()
    
    if user_data.empty:
        return None
        
    X_user = user_data[['tx_count', 'avg_amount_abs', 'std_amount', 'max_amount_abs', 'cv_amount']]
    user_data['predicted_basket'] = basket_model.predict(X_user)
    user_data['total_spend'] = user_data['tx_count'] * user_data['avg_amount_abs']
    
    basket_totals = user_data.groupby('predicted_basket')['total_spend'].sum().to_dict()
    return basket_totals

# ... [Keep ChatRequest and get_user_baskets classes here] ...

def get_financial_twin_benchmark(customer_id: str, user_total_spend: float):
    """Finds the 5 most similar users and calculates their average spend."""
    user_row = twin_db[twin_db['customer_id'] == customer_id]
    if user_row.empty:
        return "No benchmark available."
        
    # Get the user's raw features and scale them
    X_user = user_row[['age', 'total_volume', 'total_transactions']]
    X_user_scaled = twin_scaler.transform(X_user)
    
    # Find the 6 closest people (Index 0 is the user, 1-5 are the twins)
    distances, indices = twin_model.kneighbors(X_user_scaled)
    
    # Get the twins' data
    twin_indices = indices[0][1:] # Skip the first one
    twins = twin_db.iloc[twin_indices]
    
    twin_avg_spend = twins['total_volume'].mean()
    
    # Calculate the difference
    diff_percent = ((user_total_spend - twin_avg_spend) / twin_avg_spend) * 100
    
    if diff_percent > 0:
        return f"The user spends {abs(diff_percent):.1f}% MORE than their 5 closest demographic twins (Avg twin spend: €{twin_avg_spend:.2f})."
    else:
        return f"The user spends {abs(diff_percent):.1f}% LESS than their 5 closest demographic twins (Avg twin spend: €{twin_avg_spend:.2f})."

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
        baskets = get_user_baskets(req.customer_id)
        if not baskets:
            raise HTTPException(status_code=404, detail="Customer not found")
            
        total_spend = sum(baskets.values())
        benchmark_insight = get_financial_twin_benchmark(req.customer_id, total_spend)
        
        # Updated system prompt to forbid Markdown tables and encourage natural chat
        system_prompt = f"""
        You are the S-Bank AI Financial Coach named Taimi. Speak in short, punchy, conversational paragraphs. 
        Use emojis naturally. Do NOT use Markdown tables. Use bullet points instead. Focus on actionable, friendly advice.
        
        CRITICAL INSTRUCTION: Always reply in the exact same language that the user is speaking (e.g., Finnish or English).

        USER SPENDING PROFILE:
        - Mandatory Regular: €{baskets.get('Mandatory Regular', 0):.2f}
        - Mandatory Irregular: €{baskets.get('Mandatory Irregular', 0):.2f}
        - Everyday Spending: €{baskets.get('Everyday', 0):.2f}
        
        PEER BENCHMARKING:
        {benchmark_insight}
        """
        
        # --- NEW CODE GOES HERE ---
        
        # 1. Create the system prompt message
        messages_for_llm = [{"role": "system", "content": system_prompt}]

        # 2. Add the entire conversation history sent by the frontend
        for msg in req.conversation:
            messages_for_llm.append({"role": msg.role, "content": msg.content})

        # 3. Send it all to Groq
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b", 
            messages=messages_for_llm
        )
        
        # --------------------------
        
        return {
            "reply": response.choices[0].message.content,
            "data_context": baskets,
            "benchmark": benchmark_insight
        }