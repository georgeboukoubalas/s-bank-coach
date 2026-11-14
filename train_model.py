import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

from data_loader import load_and_merge_data, extract_features_for_baskets

def train_basket_classifier():
    print("Loading data for training...")
    master_data = load_and_merge_data()
    features_df = extract_features_for_baskets(master_data)
    
    category_mapping = {
        'Housing': 'Mandatory Regular', 'Utilities': 'Mandatory Regular',
        'Loans': 'Mandatory Regular', 'Insurance': 'Mandatory Regular',
        'Health': 'Mandatory Irregular', 'Home & Tech': 'Mandatory Irregular',
        'Vehicle': 'Mandatory Irregular', 'Maintenance': 'Mandatory Irregular',
        'Groceries': 'Everyday', 'Entertainment': 'Everyday',
        'Restaurants': 'Everyday', 'Transport': 'Everyday',
        'Shopping': 'Everyday', 'Travel': 'Everyday'
    }
    
    features_df['basket'] = features_df['category'].map(category_mapping)
    train_df = features_df.dropna(subset=['basket']).copy()
    
    # 1. ADD OUR NEW FEATURES TO 'X'
    X = train_df[['tx_count', 'avg_amount_abs', 'std_amount', 'max_amount_abs', 'cv_amount']]
    y = train_df['basket']
    
    # 2. Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Define the base pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])
    
    # 4. Define the hyperparameter grid to search
    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9, 11, 15],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan']
    }
    
    print("Running Grid Search to find optimal hyperparameters (this may take a minute)...")
    # 5. Run the search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    # 6. Extract the best model and evaluate
    best_model = grid_search.best_estimator_
    accuracy = best_model.score(X_test, y_test)
    
    print(f"\nBest parameters found: {grid_search.best_params_}")
    print(f"Optimized Validation Accuracy: {accuracy * 100:.2f}%")
    
    joblib.dump(best_model, 'basket_classifier.pkl')
    print("Optimized model successfully saved to 'basket_classifier.pkl'")

if __name__ == "__main__":
    train_basket_classifier()