import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from data_loader import load_and_merge_data, extract_features_for_baskets, extract_features_for_twins

def train_financial_twin():
    print("Loading data for Financial Twin...")
    master_data = load_and_merge_data()
    basket_features = extract_features_for_baskets(master_data)
    twin_df = extract_features_for_twins(master_data, basket_features)
    
    print(f"Mapping {len(twin_df)} users into the demographic space...")
    
    # 1. Define the features that make people "similar"
    # We use Age, Total Money Spent, and Transaction Frequency
    X = twin_df[['age', 'total_volume', 'total_transactions']]
    
    # 2. Scale the data! Crucial because age is ~20-80, but volume is in the thousands
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Train the Unsupervised KNN
    # We look for 6 neighbors (1 is the user themselves, the other 5 are the twins)
    twin_model = NearestNeighbors(n_neighbors=6, algorithm='auto')
    twin_model.fit(X_scaled)
    
    # 4. Save the Model, the Scaler, and the Lookup Table
    # We must save the scaler to scale future queries, and the twin_df to look up who the twins actually are
    joblib.dump({
        'model': twin_model,
        'scaler': scaler,
        'user_data': twin_df
    }, 'financial_twin.pkl')
    
    print("Financial Twin model and scaler successfully saved to 'financial_twin.pkl'")

if __name__ == "__main__":
    train_financial_twin()