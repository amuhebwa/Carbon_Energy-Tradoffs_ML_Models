import numpy as np
import pandas as pd
import code

# Define parameters for CPT
alpha = 0.88  # Curvature parameter for accuracy (gains)
beta = 0.88   # Curvature parameter for carbon emissions and energy consumption (losses)
lambda_ = 2.25  # Loss aversion coefficient
gamma = 0.61  # Probability weighting parameter (less relevant in deterministic context)

# Reference points (set to 0 for simplicity)
x_ref = 0

# Define value functions
def value_function(x, ref, alpha, beta, lambda_):
    if x >= ref:
        return (x - ref) ** alpha
    else:
        return -lambda_ * (ref - x) ** beta

# Define probability weighting function
def probability_weighting(p, gamma):
    return p ** gamma / (p ** gamma + (1 - p) ** gamma) ** (1 / gamma)

# Define the overall CPT value function
def cpt_value(accuracy, carbon_emissions, energy_consumed, weights, alpha, beta, lambda_, gamma):
    # Combine the objectives into a single outcome using weights
    outcome = (
        weights['accuracy'] * value_function(accuracy, x_ref, alpha, beta, lambda_) +
        weights['carbon_emissions'] * value_function(-carbon_emissions, x_ref, alpha, beta, lambda_) +
        weights['energy_consumed'] * value_function(-energy_consumed, x_ref, alpha, beta, lambda_)
    )
    return outcome

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

# Calculate the CPT value for each model
df['cpt_value'] = df.apply(
    lambda row: cpt_value(
        row['accuracy'],
        row['carbon_emissions'],
        row['energy_consumed'],
        weights,
        alpha,
        beta,
        lambda_,
        gamma
    ),
    axis=1
)

code.interact(local=locals())
# Select the model(s) with the highest CPT value
best_models_cpt = df.loc[df['cpt_value'].idxmax()]
print("Best Model(s) Based on CPT:")
print(best_models_cpt)
