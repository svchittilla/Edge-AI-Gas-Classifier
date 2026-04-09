import pandas as pd
import numpy as np

# --- Configuration ---
CSV_FILE = 'chemicals_in_wind_tunnel_fault.csv'
LABEL_COL = 'Chemical'  # Make sure this matches your CSV
HEADER_FILE = 'include/wokwi_test_vectors.h'
N_SAMPLES = 5

# 1. Load the dataset
print(f"Loading data from {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)

print(f"Dropping the first column: '{df.columns[0]}'")
df = df.drop(columns=[df.columns[0]])

# Separate features (12 columns) and labels
X = df.drop(columns=[LABEL_COL]).values
y = df[LABEL_COL].values

# 2. Identify indices for each category 
# Alert / Fault
fault_idx = np.where(y == 'Sensor_Fault')[0]

# Hazardous Gas
hazardous_labels = ['Ammonia_10000', 'CO_1000', 'CO_4000']
hazard_idx = np.where(np.isin(y, hazardous_labels))[0]

# Safe Gas (Everything that is NOT fault and NOT hazardous)
safe_idx = np.where(~np.isin(y, ['Sensor_Fault'] + hazardous_labels))[0]

# 3. Perform stratified sampling (2 Alert, 1 Hazardous, 2 Safe)
rng = np.random.default_rng()
selected_indices = []

# -> Pick 2 Alerts (Faults)
selected_indices.extend(rng.choice(fault_idx, size=2, replace=False))

# -> Pick 1 Hazardous
selected_indices.extend(rng.choice(hazard_idx, size=1, replace=False))

# -> Pick 2 strictly Safe
selected_indices.extend(rng.choice(safe_idx, size=2, replace=False))

# Shuffle the final 5 samples
rng.shuffle(selected_indices)

# 4. Extract the finalized RAW vectors (Notice we use X, not X_scaled)
final_vectors = X[selected_indices].astype(np.float32)
final_labels = y[selected_indices]
n_features = X.shape[1]

# 5. Build the C++ header file
lines = []
for row_i, (vec, lbl) in enumerate(zip(final_vectors, final_labels)):
    vals = ', '.join(f'{v:.6f}f' for v in vec)
    comma = "," if row_i < len(final_vectors) - 1 else ""
    lines.append(f"  {{ {vals} }}{comma}  // [{row_i}] true={lbl}")

array_block = '\n'.join(lines)

header_content = f"""// {HEADER_FILE} — {N_SAMPLES} RAW test vectors for Wokwi simulation
// Shape: [{N_SAMPLES}][{n_features}]
// These are RAW values. They MUST be scaled using scaler_params.h before inference.

#ifndef VECTOR_SAMPLES_H
#define VECTOR_SAMPLES_H

#include <pgmspace.h>

#define N_TEST_VECTORS {N_SAMPLES}
#define N_FEATURES     {n_features}

static const char* TRUE_LABELS[{N_SAMPLES}] = {{
  {', '.join(f'"{l}"' for l in final_labels)}
}};

static const float TEST_VECTORS[{N_SAMPLES}][{n_features}] PROGMEM = {{
{array_block}
}};

#endif // VECTOR_SAMPLES_H
"""

# Write to file
with open(HEADER_FILE, 'w') as f:
    f.write(header_content)

print(f"\n✅ {HEADER_FILE} written successfully!")
print(f"Shape: {N_SAMPLES} vectors x {n_features} features")
print("\nSampled Labels in order:")
for i, lbl in enumerate(final_labels):
    print(f"  [{i}] {lbl}")