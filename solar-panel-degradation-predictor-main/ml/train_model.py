"""
Train and save regression models for solar panel efficiency prediction.

Trains 3 regression models on the cleaned dataset:
1. Linear Regression — baseline, interpretable
2. Random Forest Regressor — captures non-linearity, robust
3. Gradient Boosting Regressor — sequential error reduction, often best-in-class

Models are saved to models/ for later loading by the web app.
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def train_models(data_path='data/train_cleaned.csv', models_dir='models'):
    """
    Load cleaned dataset, split into train/test, train 3 regression models.
    Save all models and return train/test splits for evaluation.
    
    Args:
        data_path (str): Path to cleaned CSV
        models_dir (str): Directory to save trained models
        
    Returns:
        dict: {
            'X_train': training features,
            'X_test': test features,
            'y_train': training target,
            'y_test': test target,
            'models': {'linear_regression': model, 'random_forest': model, 'gradient_boosting': model}
        }
    """
    
    data_path = Path(data_path)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Load cleaned data
    df = pd.read_csv(data_path)
    print(f"\n{'='*70}")
    print(f"TRAINING PHASE 1 — ML CORE")
    print(f"{'='*70}")
    print(f"\n[1/3] Loaded cleaned dataset: {df.shape}")
    
    # Separate features (X) and target (y)
    X = df.drop(columns=['efficiency'])
    y = df['efficiency']
    
    print(f"Features (X): {X.shape[1]} columns")
    print(f"Target (y): {y.name}, range [{y.min():.3f}, {y.max():.3f}]")
    
    # Train/test split: 80/20, random_state=42 for reproducibility
    # Rationale for 80/20: 16k training samples sufficient for 3 complex models,
    # 4k test set large enough for stable evaluation metrics. 
    # Fixed random_state=42 ensures reproducible splits across runs (important for viva).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42
    )
    
    print(f"\nTrain/test split (80/20, random_state=42):")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set:     {X_test.shape[0]} samples")
    
    # Dictionary to store trained models
    models = {}
    
    # Model 1: Linear Regression
    # Rationale: Baseline, interpretable, assumes linear relationship between features and efficiency.
    # Fast to train. Multicollinearity (irradiance-temperature correlation of 0.90) could inflate
    # feature coefficients, but acceptable for baseline comparison.
    print(f"\n[2/3] Training 3 models...")
    print(f"      [1] Linear Regression (baseline, interpretable)")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['linear_regression'] = lr
    joblib.dump(lr, models_dir / 'linear_regression.joblib')
    
    # Model 2: Random Forest Regressor
    # Rationale: Captures non-linear relationships, handles multicollinearity naturally (feature selection),
    # robust to outliers. n_estimators=100 standard; max_depth=20 prevents overfitting on 16k samples.
    # random_state=42 ensures reproducibility.
    print(f"      [2] Random Forest (100 trees, max_depth=20)")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    joblib.dump(rf, models_dir / 'random_forest.joblib')
    
    # Model 3: Gradient Boosting Regressor
    # Rationale: Sequential error correction, often achieves best RMSE/R² in practice.
    # n_estimators=200 for adequate boosting depth; learning_rate=0.05 prevents overfitting.
    # max_depth=5 keeps individual trees shallow (weak learners). random_state=42 reproducibility.
    print(f"      [3] Gradient Boosting (200 trees, learning_rate=0.05, max_depth=5)")
    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    gb.fit(X_train, y_train)
    models['gradient_boosting'] = gb
    joblib.dump(gb, models_dir / 'gradient_boosting.joblib')

    # Save the ordered feature column list alongside the models.
    # Rationale: the model's internal structure only records tree splits — it has
    # no memory of column *names* or *order*. If we derive feature names from the
    # CSV at inference time instead, a retrain on a different feature set would
    # let old CSVs silently feed the wrong columns. Saving here makes the model
    # folder self-contained: one copy of the column list, created at training time,
    # guaranteed to match the model that was actually trained.
    feature_columns = list(X.columns)
    joblib.dump(feature_columns, models_dir / 'feature_columns.joblib')

    print(f"\n[3/3] Models saved to {models_dir}/")
    print(f"      OK linear_regression.joblib")
    print(f"      OK random_forest.joblib")
    print(f"      OK gradient_boosting.joblib")
    print(f"      OK feature_columns.joblib  ({len(feature_columns)} columns)")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'models': models,
        'feature_names': feature_columns
    }


if __name__ == '__main__':
    result = train_models()
    print(f"\nTraining complete. Ready for evaluation.")
