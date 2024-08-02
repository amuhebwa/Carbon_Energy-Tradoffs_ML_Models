import code

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
# Define utility functions for each objective
def utility_accuracy(acc):
    return np.log(1 + acc)  # Concave function for accuracy

def utility_carbon_emissions(ce):
    return -np.sqrt(ce)  # Convex function for carbon emissions

def utility_energy_consumed(ec):
    return -np.sqrt(ec)  # Convex function for energy consumed




if __name__ == '__main__':

    dataset = pd.read_csv('../results/merged_metrics.csv')
    experiment_names = dataset['experiment_name'].unique()

    best_models_prioritized = []

    print("We are running computations for only 'Small' models")
    dataset = dataset[dataset['model_size'] == 'Small']

    for experiment_name in experiment_names:
        current_df = dataset[dataset['experiment_name'] == experiment_name]
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
            temp_df['utility'] = (
                    current_weights['mcc'] * current_df['mcc'].apply(utility_accuracy) +
                    current_weights['carbon_emissions'] * current_df['carbon_emissions'].apply(utility_carbon_emissions) +
                    current_weights['energy_consumed'] * current_df['energy_consumed'].apply(utility_energy_consumed)
            )

            # Select the model(s) with the highest CPT value
            best_models_df = temp_df.sort_values(by='utility', ascending=False)
            # best_model = temp_df.loc[current_df['utility'].idxmax()]
            # best_model = best_models_df.head(1)
            best_models_prioritized.append(best_models_df.head(1))
            # print("Best Model(s) Based on BPT for experiment: ", experiment_name)
            # print(best_model)
            # print("-" * 50)

    best_models_prioritized_df = pd.concat(best_models_prioritized)
    best_models_prioritized_df['Theory'] = 'BPT'
    print(best_models_prioritized_df)
    best_models_prioritized_df.to_csv('../results/best_models_bpt.csv', index=False)