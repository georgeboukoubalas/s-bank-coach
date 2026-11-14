import pandas as pd

def load_and_merge_data():
    """Loads the CSVs, handles the BOM encoding, and merges demographics."""
    
    # 1. Load demographics (Age, Postal Code)
    customers_df = pd.read_csv("data/tx_customers_junction.csv", encoding="utf-8-sig")
    
    # 2. Load transactions
    tx_df = pd.read_csv("data/tx_junction_transactions.csv", encoding="utf-8-sig")
    
    # 3. Parse dates with EXACT format strings (fixes the crash AND is much faster)
    # tx_date format: 2024-10-03 -> %Y-%m-%d
    tx_df['tx_date'] = pd.to_datetime(tx_df['tx_date'], format='%Y-%m-%d')
    
    # tx_timestamp format: 03OCT24:09:04:04 -> %d%b%y:%H:%M:%S
    tx_df['tx_timestamp'] = pd.to_datetime(tx_df['tx_timestamp'], format='%d%b%y:%H:%M:%S')
    
    # 4. Merge them together on customer_id
    master_df = pd.merge(tx_df, customers_df, on="customer_id", how="left")
    
    return master_df

def extract_features_for_baskets(df):
    features = df.groupby(['customer_id', 'category']).agg(
        tx_count=('tx_amount_eur', 'count'),
        avg_amount=('tx_amount_eur', 'mean'),
        std_amount=('tx_amount_eur', 'std'),
        max_amount=('tx_amount_eur', 'min') # Note: Expenses are negative, so 'min' gets the largest expense
    ).reset_index()
    
    # Fill NaN standard deviations for single-transaction categories
    features['std_amount'] = features['std_amount'].fillna(0)
    
    # Convert amounts to absolute values for easier ML math
    features['avg_amount_abs'] = features['avg_amount'].abs()
    features['max_amount_abs'] = features['max_amount'].abs()
    
    # NEW FEATURE: Coefficient of Variation (Volatility of the spending)
    # Avoid division by zero by replacing 0s with 1s just for the denominator
    features['cv_amount'] = features['std_amount'] / features['avg_amount_abs'].replace(0, 1)
    
    return features

def extract_features_for_twins(df, basket_features_df):
    """
    Creates a single-row summary for every user for the Financial Twin model.
    Combines age, postal code, and their total spending.
    """
    # 1. Get user demographics (take the first row per user since age doesn't change)
    user_demographics = df.groupby('customer_id').agg(
        age=('customer_age', 'first'),
        postal_code=('customer_postal_code', 'first')
    ).reset_index()
    
    # 2. Calculate their total spending from the basket features
    # tx_count * avg_amount_abs gives us the total money spent per category
    basket_features_df['total_spent'] = basket_features_df['tx_count'] * basket_features_df['avg_amount_abs']
    
    user_spending = basket_features_df.groupby('customer_id').agg(
        total_volume=('total_spent', 'sum'),
        total_transactions=('tx_count', 'sum')
    ).reset_index()
    
    # 3. Merge them together
    twin_features = pd.merge(user_demographics, user_spending, on="customer_id")
    
    # Drop anyone missing an age to prevent math errors
    return twin_features.dropna(subset=['age'])

if __name__ == "__main__":
    print("Loading and merging S-Bank datasets...")
    master_data = load_and_merge_data()
    
    print(f"Master dataset ready: {len(master_data)} transactions loaded.")
    
    print("Extracting mathematical features for KNN classification...")
    basket_features = extract_features_for_baskets(master_data)
    
    print(f"Feature extraction complete! {len(basket_features)} customer-category pairs ready for scikit-learn.")
    # Print a preview
    print(basket_features.head())