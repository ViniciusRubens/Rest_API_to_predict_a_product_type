import os
import pickle
import json
import numpy as np
import cudf 
import cuml
from cuml.ensemble import RandomForestClassifier
from cuml.metrics import accuracy_score, confusion_matrix
from cuml.model_selection import GridSearchCV

# conda install -c rapidsai -c nvidia -c conda-forge cuml cudf

# --- Config ---
DATA_DIR = '../../pre_processing/data/'
X_TRAIN_PATH = os.path.join(DATA_DIR, 'X_train.parquet')
X_TEST_PATH = os.path.join(DATA_DIR, 'X_test.parquet')
Y_TRAIN_PATH = os.path.join(DATA_DIR, 'y_train.parquet')
Y_TEST_PATH = os.path.join(DATA_DIR, 'y_test.parquet')

ARTIFACTS_DIR = '../artifacts/'
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'model.pkl')
METRICS_PATH = os.path.join(ARTIFACTS_DIR, 'model_best_metrics.json')
# --------------------

# --- Hiperparams ---
PARAM_GRID = {
    'max_depth': [6, 8, 10, 12],
    'n_estimators': [100, 200, 300],
    'min_samples_leaf': [2, 3, 4, 5]
    # Total combinations: 4 * 3 * 4 = 48
}

# GridSearchCV config
CV_FOLDS = 5 # "Cross-Validations"
# Total trained models: 48 * 5 folds = 240
# --------------------

def run_grid_search():

    # --- 1. Load Data ---
    print("Loading data directly into GPU memory (VRAM)...")
    X_train = cudf.read_parquet(X_TRAIN_PATH)
    X_test = cudf.read_parquet(X_TEST_PATH)
    y_train = cudf.read_parquet(Y_TRAIN_PATH)
    y_test = cudf.read_parquet(Y_TEST_PATH)
    print("Data loading complete.\n")

    # --- 2. Grid Search Setup ---
    print("Initializing cuML RandomForestClassifier and GridSearchCV...")
    
    model_base = RandomForestClassifier() 
    
    grid_search = GridSearchCV(
        estimator = model_base,
        param_grid = PARAM_GRID,
        cv = CV_FOLDS,
        verbose = 2
    )

    print("--- Starting Hyperparameter GridSearch on GPU ---")
    print(f"Combinations: {len(PARAM_GRID['max_depth']) * len(PARAM_GRID['n_estimators']) * len(PARAM_GRID['min_samples_leaf'])}")
    print(f"CV Folds: {CV_FOLDS}")
    print(f"Total models to fit: {len(PARAM_GRID['max_depth']) * len(PARAM_GRID['n_estimators']) * len(PARAM_GRID['min_samples_leaf']) * CV_FOLDS}")
    print("This may take several time...\n")

    # --- 3. Model Training ---
    grid_search.fit(X_train, y_train['product_type'].values.get()) # GridSearchCV in cuML needs y in 1D array
    
    print("\n--- GridSearch Complete ---")

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"Best parameters found: {best_params}")
    print("------------------------------------------\n")

    # --- 4. Prediction & Evaluation ---
    print("Predicting on test set with best model...")
    y_pred = best_model.predict(X_test)

    print("Calculating final metrics on test set...")
    y_test_values = y_test['product_type'].values

    accuracy = accuracy_score(y_test_values, y_pred)

    cm_cupy = confusion_matrix(y_test_values, y_pred)
    cm_numpy = cm_cupy.get() # Convert CuPy array -> NumPy array
    cm_list = cm_numpy.tolist() # Convert NumPy array -> Python list
    
    metrics = {
        "best_parameters": best_params,
        "test_metrics": {
            "accuracy": round(float(accuracy), 4),
            "confusion_matrix": cm_list
        }
    }

    # --- 5. Print & Save Artifacts ---
    print(f"\n--- Best Model Metrics ---")
    print(f"Accuracy: {metrics['test_metrics']['accuracy']}")
    print(f"Confusion Matrix:\n{np.array(cm_list)}")
    print("------------------------------------------\n")

    os.makedirs(ARTIFACTS_DIR, exist_ok = True)

    print(f"Saving best model artifact to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)

    print(f"Saving metrics and best params to {METRICS_PATH}...")
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)

    print("GridSearch script complete.")

if __name__ == "__main__":
    run_grid_search()