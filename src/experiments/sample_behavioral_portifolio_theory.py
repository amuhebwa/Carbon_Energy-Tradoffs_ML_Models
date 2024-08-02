import code

import numpy as np
import pandas as pd

# Define utility functions for each objective
def utility_accuracy(acc):
    return np.log(1 + acc)  # Concave function for accuracy

def utility_carbon_emissions(ce):
    return -np.sqrt(ce)  # Convex function for carbon emissions

def utility_energy_consumed(ec):
    return -np.sqrt(ec)  # Convex function for energy consumed

# Assign weights to each objective
weights = {
    'accuracy': 0.5,
    'carbon_emissions': 0.25,
    'energy_consumed': 0.25
}

# Sample data for illustration
data = {
    'full_model_name': [
        'Model_1', 'Model_2', 'Model_3', 'Model_4', 'Model_5'
    ],
    'accuracy': [
        0.95, 0.90, 0.85, 0.80, 0.75
    ],
    'carbon_emissions': [
        0.02, 0.03, 0.04, 0.05, 0.06
    ],
    'energy_consumed': [
        0.10, 0.15, 0.20, 0.25, 0.30
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)
# Calculate overall utility for each model
df['utility'] = (
    weights['accuracy'] * df['accuracy'].apply(utility_accuracy) +
    weights['carbon_emissions'] * df['carbon_emissions'].apply(utility_carbon_emissions) +
    weights['energy_consumed'] * df['energy_consumed'].apply(utility_energy_consumed)
)

# Select the model(s) with the highest overall utility
best_models_bpt = df.loc[df['utility'].idxmax()]
print("Best Model(s) Based on BPT:")
print(best_models_bpt)
