"""
data_loader.py
Location: CodeAlpha_CreditScoringModel/data_loader.py

Step 1: Load the German Credit dataset (UCI Statlog).
"""

import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

COLUMNS = [
    'checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings', 'employment', 'installment_rate', 'personal_status', 'other_debtors',
    'residence_since', 'property', 'age', 'other_installment', 'housing',
    'existing_credits', 'job', 'num_dependents', 'telephone', 'foreign_worker', 'target'
]


def load_data(path_or_url: str = DATA_URL) -> pd.DataFrame:
    """
    Loads the German Credit dataset from a URL or local file path.
    If you downloaded the file manually, save it to:
        CodeAlpha_CreditScoringModel/data/german.data
    and call load_data("data/german.data")
    """
    df = pd.read_csv(path_or_url, sep=' ', header=None, names=COLUMNS)
    # Original target: 1 = good credit, 2 = bad credit -> remap to 1/0
    df['target'] = df['target'].map({1: 1, 2: 0})
    return df


if __name__ == "__main__":
    df = load_data()
    print("Shape:", df.shape)
    print(df.head())
