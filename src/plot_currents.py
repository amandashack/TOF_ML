import pandas as pd
import matplotlib.pyplot as plt

def load_and_plot_data(filepath):
    # Load the data
    data = pd.read_csv(filepath, sep=' ', engine='python')
    print("Data Loaded:\n", data.head())

    # Visual inspection of data
    print("Data Description:\n", data.describe())

    # Plotting
    fig, ax = plt.subplots()
    scatter = ax.scatter(data['Front_Voltage'], data['Back_Voltage'], c=data['Max_Val'], cmap='viridis')
    colorbar = plt.colorbar(scatter, ax=ax)
    colorbar.set_label('Max Value')

    ax.set_xlabel('Front Voltage (V)')
    ax.set_ylabel('Back Voltage (V)')
    ax.set_title('Front vs Back Voltage and Max Values')
    plt.show()

if __name__ == '__main__':
    # Replace 'path_to_output_file.txt' with the path to your simulation results file
    load_and_plot_data('C:/Users/proxi/Documents/coding/TOF_ML/src/simulation_results.txt')