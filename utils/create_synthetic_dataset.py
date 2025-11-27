import pandas as pd
import numpy as np

# 1. CONSTANTS DEFINITION
N_ROWS_TOTAL = 5000
PRODUCTS = ['Smartphone', 'Tablet']
N_PRODUCTS = len(PRODUCTS)
N_PER_PRODUCT = N_ROWS_TOTAL // N_PRODUCTS # 1250 rows per product

# Noise settings
LABEL_NOISE_PERCENT = 0.1
WEIGHT_NOISE_STD_PERCENT = 0.25

np.random.seed(42)

def get_probabilistic_size(product, base_size):
    if product == 'Tablet' and base_size == 'Small Package':
        # 80% chance of being 'Small', 20% of being 'Large'
        return np.random.choice(
            ['Small Package', 'Large Package'], 
            size=N_PER_PRODUCT, 
            p=[0.8, 0.2]
        )
    elif product == 'Notebook' and base_size == 'Large Package':
        # 95% chance of being 'Large', 5% of being 'Small' (e.g., netbook)
        return np.random.choice(
            ['Large Package', 'Small Package'], 
            size=N_PER_PRODUCT, 
            p=[0.95, 0.05]
        )
    else:
        # Keep the default for others
        return [base_size] * N_PER_PRODUCT

# 2. PATTERN DEFINITION (BUSINESS RULES)
# Structure: { 'product': ('package_size', base_mean_weight) }
patterns = {
    'Smartphone': ('Small Package', 220),
    'Tablet':     ('Large Package', 550)
}

print(f"Starting generation of {N_ROWS_TOTAL} rows...")
print(f"Classes: {PRODUCTS} ({N_PER_PRODUCT} rows per class).")
print(f"Label Noise: {LABEL_NOISE_PERCENT*100}%")
print(f"Weight Noise (Std Dev): {WEIGHT_NOISE_STD_PERCENT*100}% of the mean")

generated_dfs = []

# 3. GENERATING "CLEAN" DATA (WITH WEIGHT NOISE)
print("Generating base data (with Gaussian weight noise)...")
for product, (pkg_size, mean_weight) in patterns.items():
    
    # Generate ground truth labels
    true_labels = [product] * N_PER_PRODUCT
    
    # Generate package size (probabilistic/deterministic)
    package_sizes = get_probabilistic_size(product, pkg_size)
    
    # Calculate standard deviation (percentage of the mean)
    weight_std_dev = mean_weight * WEIGHT_NOISE_STD_PERCENT
    
    # Generate weights with Gaussian noise (normal distribution)
    # loc = mean (mean_weight), scale = standard deviation (weight_std_dev)
    weights = np.random.normal(
        loc=mean_weight,
        scale=weight_std_dev,
        size=N_PER_PRODUCT
    )
    
    # Round weights to 2 decimal places
    weights = np.round(weights, 2)
    
    # Ensure no weight is negative (unlikely, but good practice)
    weights[weights < 0] = mean_weight 
    
    # Create DataFrame for this class
    df_product = pd.DataFrame({
        'true_product_type': true_labels, # Helper column for label noise
        'package_size': package_sizes,
        'package_weight_gr': weights
    })
    
    generated_dfs.append(df_product)

# Combine all product dataframes
df = pd.concat(generated_dfs, ignore_index=True)

# 4. APPLYING LABEL NOISE
# Copy the true label column to the final column
df['product_type'] = df['true_product_type']

# Calculate how many labels should be corrupted
n_rows_to_noise = int(N_ROWS_TOTAL * LABEL_NOISE_PERCENT) 
print(f"Applying label noise to {n_rows_to_noise} random rows...")

# Select indices of rows to have their label flipped
noise_indices = np.random.choice(df.index, n_rows_to_noise, replace=False)

# Apply noise
for i in noise_indices:
    # Get current true label
    current_true_label = df.loc[i, 'true_product_type']
    
    # Define false label options (anything except the true one)
    possible_new_labels = [p for p in PRODUCTS if p != current_true_label]
    
    # Randomly choose a new false label
    new_label = np.random.choice(possible_new_labels)
    
    # Overwrite the label in 'product_type' column
    df.loc[i, 'product_type'] = new_label

# 5. FINALIZATION AND EXPORT

print("Cleaning and shuffling the dataset...")
# Shuffle the final dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Add ID column
df['id'] = range(1, len(df) + 1)

# Select and order final columns
final_columns = ['id', 'package_weight_gr', 'package_size', 'product_type']
df_final = df[final_columns]

# Save to CSV
FILE_NAME = 'synthetic_shipping_data.csv'
df_final.to_csv(FILE_NAME, index=False)

print("\n--- SCRIPT COMPLETED ---")
print(f"File '{FILE_NAME}' generated successfully!")
print("\n--- Final DataFrame Head ---")
print(df_final.head())
print("\n--- Target Column Distribution (product_type) ---")
print(df_final['product_type'].value_counts())