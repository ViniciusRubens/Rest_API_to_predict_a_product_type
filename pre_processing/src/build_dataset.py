import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, List, Any

# --- Config ---
RAW_DATA_PATH = '../../dataset/synthetic_shipping_data.csv'
X_TRAIN_PATH = '../data/X_train.parquet'
X_TEST_PATH = '../data/X_test.parquet'
Y_TRAIN_PATH = '../data/y_train.parquet'
Y_TEST_PATH = '../data/y_test.parquet'
ARTIFACTS_PATH = '../data/artifacts/'

TEST_SIZE = 0.2
RANDOM_STATE = 42


# --- Load data ---
def load_data(path: str) -> pd.DataFrame:

    df_products = pd.read_csv(path)
    df_products = df_products[['package_weight_gr', 'package_size', 'product_type']]
    return df_products


# --- Train and test split
def split_data(data: pd.DataFrame) -> List[Any]:

    X = data[['package_weight_gr', 'package_size']]
    y = data['product_type']

    print(f"Splitting data... Test size: {TEST_SIZE}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

    return X_train, X_test, y_train, y_test 


def label_encoder_data(
    x_train: pd.DataFrame, x_test: pd.DataFrame, 
    y_train: pd.Series, y_test: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder, LabelEncoder]:
    
    """
    Encodes categorical features for train and test sets.
    
    Applies LabelEncoder to 'package_size' in X and to the
    target variable y. 
    """
    
    # Create copies to safely modify data
    x_train_encoded = x_train.copy()
    x_test_encoded = x_test.copy()

    # Initialize encoders
    label_package_size = LabelEncoder()
    label_product_type = LabelEncoder()

    # Fit 'package_size' encoder ONLY on training data
    label_package_size.fit(x_train_encoded['package_size'])
    
    # Fit 'product_type' encoder ONLY on training data
    label_product_type.fit(y_train) # y_train is 1D

    # Apply transform to 'package_size' on both train and test
    x_train_encoded['package_size'] = label_package_size.transform(x_train_encoded['package_size'])
    x_test_encoded['package_size'] = label_package_size.transform(x_test_encoded['package_size'])

    # Apply transform to target variable y (returns numpy arrays)
    y_train_encoded = label_product_type.transform(y_train)
    y_test_encoded = label_product_type.transform(y_test)

    return (
        x_train_encoded, x_test_encoded, 
        y_train_encoded, y_test_encoded, 
        label_package_size, label_product_type
    )


def scale_features(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    
    """
    Do StandardScaler in 'package_weight_gr' column.
    """
    
    # Create copies to safely modify data
    x_train_scaled = x_train.copy()
    x_test_scaled = x_test.copy()

    scaler = StandardScaler()

    # --- Just fit the train ---
    scaler.fit(x_train_scaled[['package_weight_gr']])

    # --- TRANSFORM both train and test ---
    x_train_scaled['package_weight_gr'] = scaler.transform(x_train_scaled[['package_weight_gr']])
    x_test_scaled['package_weight_gr'] = scaler.transform(x_test_scaled[['package_weight_gr']])
    
    return x_train_scaled, x_test_scaled, scaler


if __name__ == "__main__":

    print("--- 1. Loading Data ---")
    
    data_frame = load_data(RAW_DATA_PATH)

    print("--- 2. Splitting Data ---")
    X_train, X_test, y_train, y_test = split_data(data_frame)

    print(f"X Train set shape: {X_train.shape}")
    print(f"X Test set shape: {X_test.shape}")
    print(f"y Train set shape: {y_train.shape}")
    print(f"y Test set shape: {y_test.shape}")

    print("--- 3. Encoding Data ---")
    (
        x_train_encoded, x_test_encoded, 
        y_train_encoded, y_test_encoded, 
        package_size_encoder, product_type_encoder

    ) = label_encoder_data(X_train, X_test, y_train, y_test)

    # print("--- 4. Scaling Data ---")
    # (
    #     x_train_final, x_test_final, 
    #     weight_scaler

    # ) = scale_features(x_train_encoded, x_test_encoded)
    
    print("--- 5. Saving Artifacts ---")
    joblib.dump(package_size_encoder, f'{ARTIFACTS_PATH}/package_size_encoder.pkl')
    joblib.dump(product_type_encoder, f'{ARTIFACTS_PATH}/product_type_encoder.pkl')
    # joblib.dump(weight_scaler, f'{ARTIFACTS_PATH}/weight_scaler.pkl')

    print("--- 6. Saving Data ---")
    x_train_encoded.to_parquet(X_TRAIN_PATH, index = False)
    x_test_encoded.to_parquet(X_TEST_PATH, index = False)

    # Save target as DataFrame to keep column name
    pd.Series(y_train_encoded).to_frame(name = 'product_type').to_parquet(Y_TRAIN_PATH, index = False)
    pd.Series(y_test_encoded).to_frame(name = 'product_type').to_parquet(Y_TEST_PATH, index = False)


    print("\nPre-processing complete.")
    print("Final training data ready:")
    
