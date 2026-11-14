import joblib
from data_loader import load_and_merge_data, extract_features_for_baskets

print("Loading raw CSV data on local machine...")
master_data = load_and_merge_data()

print("Calculating math features...")
all_user_features = extract_features_for_baskets(master_data)

print("Saving features to a lightweight file...")
joblib.dump(all_user_features, "user_features.pkl")

print("Done! user_features.pkl is ready for the server.")