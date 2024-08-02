import numpy as np
import pandas as pd
import tensorflow as tf
from utils import weight_combinations, priorities
import code
# turn off warning
pd.options.mode.chained_assignment = None  # default='warn'

# seed
np.random.seed(42)
tf.random.set_seed(42)

# Define parameters for CPT
alpha = 0.88  # Curvature parameter for accuracy (gains)
beta = 0.88   # Curvature parameter for carbon emissions and energy consumption (losses)
lambda_ = 2.0 # Loss aversion coefficient
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
        weights['mcc'] * value_function(accuracy, x_ref, alpha, beta, lambda_) +
        weights['carbon_emissions'] * value_function(-carbon_emissions, x_ref, alpha, beta, lambda_) +
        weights['energy_consumed'] * value_function(-energy_consumed, x_ref, alpha, beta, lambda_)
    )
    return outcome

if __name__ == '__main__':
    dataset = pd.read_csv('../results/merged_metrics.csv')
    experiment_names = dataset['experiment_name'].unique()
    model_sizes = dataset['model_size'].unique()

    best_models_prioritized = []

    print("We are running computations for only 'Small' models")
    dataset = dataset[dataset['model_size'] == 'Small']

    for experiment_name in experiment_names:
        current_df = dataset[(dataset['experiment_name'] == experiment_name)]
        # Transform MCC to [0, 1] range
        current_df['mcc'] = (current_df['mcc'] + 1) / 2
        # Normalize carbon emissions and energy consumption using a small offset to avoid exact zeros
        offset = 1e-8  # Small constant to avoid exact zeros
        current_df['carbon_emissions'] = (current_df['carbon_emissions'] - current_df['carbon_emissions'].min() + offset) / (current_df['carbon_emissions'].max() - current_df['carbon_emissions'].min() + offset)
        current_df['energy_consumed'] = (current_df['energy_consumed'] - current_df['energy_consumed'].min() + offset) / (current_df['energy_consumed'].max() - current_df['energy_consumed'].min() + offset)

        for selected_priority in priorities:
            temp_df = current_df.copy()
            temp_df['Priority'] = selected_priority
            current_weights = weight_combinations[selected_priority]
            temp_df['cpt_value'] = temp_df.apply(
                lambda row: cpt_value(
                    row['mcc'], row['carbon_emissions'], row['energy_consumed'],
                    current_weights, alpha, beta, lambda_, gamma
                ),
                axis=1
            )

            # Select the model(s) with the highest CPT value
            best_models_df = temp_df.sort_values(by='cpt_value', ascending=False)
            # best_model = best_models_df.loc[best_models_df['cpt_value'].idxmax()]
            # best_model = best_models_df.head(1)
            best_models_prioritized.append(best_models_df.head(1))
            # print("Best Model(s) Based on BPT for experiment: ", experiment_name)
            # print(best_model)
            # print("-" * 50)
    best_models_prioritized_df = pd.concat(best_models_prioritized)
    best_models_prioritized_df['Theory'] = 'CPT'
    print(best_models_prioritized_df)
    best_models_prioritized_df.to_csv('../results/best_models_cpt.csv', index=False)
