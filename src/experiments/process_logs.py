# Open the text file
import code
import pandas as pd
import matplotlib.pyplot as plt
import ast

if __name__ == "__main__":
    filepath = "/Users/amuhebwa/Documents/Stanford/Research_Code/CarbonMeasure_Responsibility/src/results/logs/log1.txt"

    train_accuracies, train_losses, eval_accuracies, eval_losses = [], [], [], []
    with open(filepath, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            if 'eval_accuracy' in line:
                current_dict = ast.literal_eval(line)
                eval_loss = current_dict['eval_loss']
                eval_accuracy = current_dict['eval_accuracy']
                eval_accuracies.append(eval_accuracy)
                eval_losses.append(eval_loss)

            elif 'train_accuracy' in line:
                current_dict = ast.literal_eval(line)
                train_loss = current_dict['train_loss']
                train_accuracy = current_dict['train_accuracy']
                train_accuracies.append(train_accuracy)
                train_losses.append(train_loss)
            else:
                continue

    # create a pandas dataframe
    #data = {'train_accuracy': train_accuracies, 'train_loss': train_losses, 'eval_accuracy': eval_accuracies, 'eval_loss': eval_losses}
    #df = pd.DataFrame(data)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='train_loss')
    plt.plot(eval_losses, label='eval_loss')
    plt.plot(train_accuracies, label='train_accuracy', linestyle='--')
    plt.plot(eval_accuracies, label='eval_accuracy', linestyle='-.')
    plt.show()