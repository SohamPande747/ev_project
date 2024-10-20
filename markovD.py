import os
import logging
import numpy as np  # Replace CuPy with NumPy
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix  # Use SciPy's sparse CSR matrix

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def load_and_preprocess_csv(file_path):
    try:
        logger.info(f"[INFO] Attempting to load CSV file: {file_path}")
        data = pd.read_csv(file_path)  # Still using pandas as it's CPU-based
        logger.info(f"[INFO] Successfully loaded CSV file: {file_path} with {len(data)} rows.")
        return data
    except Exception as e:
        logger.error(f"[ERROR] Failed to load CSV file: {file_path}, {str(e)}")
        raise

def compute_reward(row):
    # Example reward function: reward based on energy charge and discharge
    reward = row['EnergyCharge_W_h'] - row['EnergyDischarge_W_h']
    logger.debug(f"[DEBUG] Computed reward for row: {row['cycleNumber']} = {reward}")
    return reward

def plot_data(data, plot_file):
    logger.info(f"[INFO] Plotting data and saving to: {plot_file}")
    plt.figure(figsize=(10, 5))
    plt.plot(data['cycleNumber'], data['EnergyCharge_W_h'], label='Energy Charge (W*h)', color='blue')
    plt.plot(data['cycleNumber'], data['EnergyDischarge_W_h'], label='Energy Discharge (W*h)', color='red')
    plt.xlabel('Cycle Number')
    plt.ylabel('Energy (W*h)')
    plt.title('Energy Charge and Discharge Over Cycles')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_file)
    plt.close()
    logger.info(f"[INFO] Successfully saved plot to: {plot_file}")

def process_csv_file(file_path):
    logger.info(f"[INFO] Starting processing for file: {file_path}")
    output_file = f'mdp_results_{os.path.basename(file_path)}.txt'
    plot_file = f'plot_{os.path.basename(file_path).replace(".csv", ".png")}'

    # Load the CSV data
    data = load_and_preprocess_csv(file_path)

    # Create states and rewards
    states = []
    action_count = 3  # Number of actions (charge, discharge, maintain)
    rewards = []
    logger.info(f"[INFO] Number of actions: {action_count}")

    # Populate states and rewards
    for index, row in data.iterrows():
        state = (
            row['Ecell_V'],
            row['I_mA'],
            row['EnergyCharge_W_h'],
            row['QCharge_mA_h'],
            row['EnergyDischarge_W_h'],
            row['QDischarge_mA_h'],
            row['Temperature__C'],
            row['cycleNumber'],
            row['Ns']
        )
        states.append(state)
        reward_value = compute_reward(row)
        rewards.append([reward_value] * action_count)  # Append the same reward for each action
        logger.debug(f"[DEBUG] Added state {index}: {state} with rewards: {rewards[-1]}")

    # Convert states and rewards to NumPy arrays for CPU computation
    states = np.array(states)
    rewards = np.array(rewards)

    # Create transition probabilities (sparse representation with SciPy)
    transitions = [csr_matrix((len(states), len(states)), dtype=np.float32) for _ in range(action_count)]
    logger.info(f"[INFO] Creating transition probabilities for {len(states)} states.")
    for current_state in range(len(states)):
        for action_index in range(action_count):
            for next_state in range(len(states)):
                # Uniformly distribute transitions for simplicity
                transitions[action_index][current_state, next_state] = 1.0 / len(states)
                logger.debug(f"[DEBUG] Transition from state {current_state} to {next_state} for action {action_index} set to {transitions[action_index][current_state, next_state]}.")

    # MDP Value Iteration on CPU
    discount_factor = 0.9  # Discount factor
    epochs = 10  # Set lower epochs here
    logger.info(f"[INFO] Starting MDP Value Iteration with discount factor: {discount_factor} and epochs: {epochs}.")
    value_function = np.zeros(len(states))  # Value function on CPU
    policy = np.zeros(len(states), dtype=np.int32)  # Policy on CPU

    for iteration in range(epochs):  # Iterate for a defined number of epochs
        new_value_function = np.copy(value_function)
        logger.debug(f"[DEBUG] Iteration {iteration + 1}: current value function: {value_function}")
        for state in range(len(states)):
            action_values = np.zeros(action_count)
            for action in range(action_count):
                action_values[action] = np.sum(transitions[action][state].toarray() * 
                    (rewards[state][action] + discount_factor * value_function))
            new_value_function[state] = np.max(action_values)
            policy[state] = np.argmax(action_values)
            logger.debug(f"[DEBUG] State {state}: Action values = {action_values}, Optimal Action = {policy[state]}, Value = {new_value_function[state]}")
        value_function = new_value_function

    logger.info("[INFO] MDP run complete. Writing results to file.")
    # Save the output
    with open(output_file, 'a') as f:
        f.write(f"\nResults for {os.path.basename(file_path)}:\n")
        for i, state in enumerate(states):
            f.write(f"State {state}: Optimal Action = {policy[i]}, Value = {value_function[i]}\n")
            logger.debug(f"[DEBUG] Wrote to file: State {state}, Optimal Action = {policy[i]}, Value = {value_function[i]}")

    # Plot the data comparison and save it as a PNG
    plot_data(data, plot_file)

    logger.info(f"[INFO] Processing complete for {file_path}. Results saved to {output_file} and plot saved to {plot_file}.")
    return output_file, plot_file  # Return the output files for downloading later

# Main function to process multiple CSV files
def process_multiple_csv_files(file_paths):
    logger.info("[INFO] Starting to process multiple CSV files...")
    for file_path in file_paths:
        logger.info(f"[INFO] Submitting file for processing: {file_path}")
        process_csv_file(file_path)

# Example usage
if __name__ == "__main__":
    csv_files = ['VAH05.csv']  # Adjust path if necessary
    process_multiple_csv_files(csv_files)
