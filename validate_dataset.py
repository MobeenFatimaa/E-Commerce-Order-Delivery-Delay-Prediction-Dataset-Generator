import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

DATA_FILE = "ecommerce_delivery_predictions.csv"

def run_validation():
    print("--- 1. Verification of File Structure ---")
    df = pl.read_csv(DATA_FILE)
    
    assert df.shape[0] == 100_000, f"Expected 100,000 rows, got {df.shape[0]}"
    assert df.shape[1] == 40, f"Expected 40 columns, got {df.shape[1]}"
    print(f"[PASS] Exact dimensions satisfied: {df.shape}")

    print("\n--- 2. Checking Nulls & Target Distribution ---")
    null_counts = df.null_count().sum_horizontal()[0]
    assert null_counts == 0, f"Found {null_counts} missing values in dataset!"
    print("[PASS] Zero missing values detected.")

    late_ratio = df["late_delivery"].mean()
    print(f"[INFO] Class balance (Late Delivery): {late_ratio:.2%}")
    assert 0.20 <= late_ratio <= 0.35, f"Class imbalance out of target range: {late_ratio}"
    print("[PASS] Class balance resides within healthy 20%-35% target bounds.")

    print("\n--- 3. Predictive Validation (Random Forest Baseline) ---")
    numeric_features = [
        "warehouse_distance_km", "package_weight_kg", "processing_time_hours",
        "dispatch_delay_hours", "seller_reliability_score", "shipping_risk_score",
        "seller_performance_score", "delivery_efficiency_score", "holiday_period", "peak_season"
    ]
    
    X = df.select(numeric_features).to_pandas()
    y = df["late_delivery"].to_pandas()

    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X, y)
    
    preds = rf.predict(X)
    probs = rf.predict_proba(X)[:, 1]
    
    auc = roc_auc_score(y, probs)
    print(f"[PASS] Baseline Model AUC Score: {auc:.4f}")
    
    print("\nTop Predictors Identified by Model:")
    importances = sorted(zip(numeric_features, rf.feature_importances_), key=lambda x: x[1], reverse=True)
    for feat, imp in importances[:5]:
        print(f" - {feat}: {imp:.4f}")

    print("\nClassification Report:")
    print(classification_report(y, preds))

if __name__ == "__main__":
    run_validation()