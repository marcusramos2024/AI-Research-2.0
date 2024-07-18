import pandas as pd
import matplotlib.pyplot as plt

def plot_mse_vs_layers(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.plot(df['Layer'], df['MSE'], marker='o', linestyle='-')
    plt.xlabel('ESM2 Layers')
    plt.ylabel('MSE')
    plt.title('Melting Temp Prediction')
    plt.grid(True)
    plt.show()

# Example 
plot_mse_vs_layers('Results/esm2_t33_650M_UR50D/2.0/Layers.csv')