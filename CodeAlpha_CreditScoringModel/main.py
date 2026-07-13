"""
main.py
Location: CodeAlpha_CreditScoringModel/main.py

Step 5: Run the full pipeline end to end.

Usage:
    python main.py
"""

import os
from data_loader import load_data
from preprocess import preprocess
from train_models import (
    train_logistic_regression,
    train_decision_tree,
    train_random_forest,
)
from evaluate import evaluate_model, plot_roc_curves, plot_feature_importance


def main():
    os.makedirs("outputs", exist_ok=True)

    # Step 1: Load data
    df = load_data()
    print("Data loaded:", df.shape)

    # Step 2: Preprocess
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = preprocess(df)

    # Step 3: Train models
    log_reg = train_logistic_regression(X_train_scaled, y_train)
    dt = train_decision_tree(X_train, y_train)
    rf = train_random_forest(X_train, y_train)

    # Predictions
    log_pred = log_reg.predict(X_test_scaled)
    log_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

    dt_pred = dt.predict(X_test)
    dt_proba = dt.predict_proba(X_test)[:, 1]

    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    # Step 4: Evaluate
    evaluate_model("Logistic Regression", y_test, log_pred, log_proba)
    evaluate_model("Decision Tree", y_test, dt_pred, dt_proba)
    evaluate_model("Random Forest", y_test, rf_pred, rf_proba)

    plot_roc_curves(y_test, {
        "Logistic Regression": log_proba,
        "Decision Tree": dt_proba,
        "Random Forest": rf_proba,
    })

    plot_feature_importance(rf, X_train.columns)

    print("Pipeline complete. Plots saved in the 'outputs/' folder.")


if __name__ == "__main__":
    main()
