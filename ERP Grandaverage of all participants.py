"""
ERP Grandaverage of all participants

Functionality:
- The script identifies CSV files within a user-specified folder, each expected to contain ERP data for events like stimulus presentation (64) or response registration (65).
- It processes each file to extract and average ERP data across all participants for each time point, allowing for detailed examination of ERP responses throughout the time series.
- Results are saved in both graphical and spreadsheet formats for further analysis and reporting.

Input Requirements:
- A directory containing CSV files named in the format 'evoked_{eventnumber}_{participantnumber}.csv', example filename: 'evoked_64_1.csv'.
- Each CSV file should have columns representing different EEG channels and rows representing time points. Please check the columnanmes for capitalizations and spelling as any discrepency will cause errors.

Outputs:
- Average ERP data for each event will be saved as Excel files in the selected directory.
- Plots of the average ERP data, showing both mean amplitudes and standard deviations, will be saved as PNG files in a subdirectory for each event.

How to Use:
1. Run the script and follow the prompts to select the folder containing the ERP CSV files.
2. The script will automatically process all suitable files in the directory, calculate averages, generate plots, and save the results.
3. Check the selected output directory for Excel files and subdirectories containing plots.

The script is equipped with a logging system to track its progress and capture any issues encountered during execution, ensuring transparency and ease of troubleshooting.

Author: Soumya Kaim
Date: 8th August 2024
Version: Final
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import logging
import datetime

def select_folder(title="Select Folder"):
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title=title)
    root.destroy()
    return folder_path

def setup_logging(output_dir, folder_path):
    current_datetime = datetime.datetime.now()
    formatted_date_time = current_datetime.strftime("%Y-%m-%d_%Hhr-%Mmins-%Ssec")
    
    log_file_path = os.path.join(output_dir, f'{os.path.basename(folder_path)}_processing_{formatted_date_time}.log')
    
    file_handler = logging.FileHandler(log_file_path, mode='w')
    console_handler = logging.StreamHandler()
    logger = logging.getLogger()
    
    logger.setLevel(logging.INFO)
    
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levellevel)s - %(message)s'))
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levellevel)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return log_file_path

def load_data_files(folder_path):
    data_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            data_files.append(os.path.join(folder_path, file_name))
    return data_files

def extract_erp_data_from_csv(data_files):
    """
    Extracts ERP data from a list of CSV files and organizes it by event number.

    Parameters:
        data_files (List[str]): A list of file paths to the CSV files containing the ERP data.

    Returns:
        Dict[str, List[pd.DataFrame]]: A dictionary where the keys are event numbers as strings and the values are lists of pandas DataFrames containing the ERP data for each event number.

    Raises:
        IndexError: If the filename format does not match the expected format.
        ValueError: If an unexpected event number is encountered.

    Notes:
        - The function assumes that the CSV files have a specific filename format: 'evoked_{eventnumber}_{participantnumber}.csv'.
        - The function logs information about the processing of each file using the logging module.
        - If a file does not match the expected filename format, a warning is logged and the file is skipped.
        - If an unexpected event number is encountered, a warning is logged and the file is skipped.
    """

    erp_data = {'64': [], '65': []}
    
    for file in data_files:
        df = pd.read_csv(file)
        basename = os.path.basename(file)
        
        logging.info(f"Processing file: {basename}")
        
        # Handle filename parsing to extract event number
        filename_parts = basename.replace('evoked_', '').replace('.csv', '').split('_')
        
        # Assume that the filename format is like 'evoked_{eventnumber}_{participantnumber}.csv'
        try:
            event = filename_parts[0]  # event number
        except IndexError:
            logging.error(f"The filename does not match the expected format: {basename}")
            continue

        if event not in erp_data:
            logging.warning(f"Unexpected event number {event} in file {basename}")
            continue
        
        erp_data[event].append(df)
        logging.info(f"File {basename} processed with {df.shape[1]-1} channels and {df.shape[0]} time points.")
    
    return erp_data

def compute_average_erp(erp_data):
    """
    Compute the average event-related potential (ERP) data for each event in the input ERP data.
    
    Parameters:
    erp_data (dict): A dictionary containing event numbers as keys and lists of DataFrames as values.
    
    Returns:
    dict: A dictionary containing event numbers as keys and DataFrames of average ERP data as values.
    """

    avg_erp_data = {}
    
    for event, dataframes in erp_data.items():
        if not dataframes:
            logging.warning(f"No data found for event {event}")
            continue
        
        # Concatenate all dataframes along the columns (axis=0)
        combined_df = pd.concat(dataframes, axis=0)
        
        # Group by 'time' and compute the mean for each group
        avg_df = combined_df.groupby('time').mean().reset_index()
        
        avg_erp_data[event] = avg_df
        logging.info(f"Computed average ERP for event {event} with shape {avg_df.shape}.")
    
    return avg_erp_data

def save_to_excel(avg_erp_data, folder_path):
    folder_name = os.path.basename(folder_path)
    for event, df in avg_erp_data.items():
        output_file = os.path.join(folder_path, f'average_erp_data_{folder_name}_event_{event}.xlsx')
        df.to_excel(output_file, index=False)
        logging.info(f"Average ERP data for event {event} saved to {output_file}")

def plot_erp_data(folder_path, avg_erp_data):
    folder_name = os.path.basename(folder_path)
    for event, df in avg_erp_data.items():
        time = df['time'].values
        output_dir = os.path.join(folder_path, f'plots_{folder_name}_event_{event}')
        os.makedirs(output_dir, exist_ok=True)
        
        for column in df.columns[1:]:
            mean = df[column].values
            std_dev = df[column].std()
            
            plt.figure()
            plt.plot(time, mean, label='Mean')
            plt.fill_between(time, mean - std_dev, mean + std_dev, alpha=0.2, label='Standard Deviation')
            plt.xlabel('Time (s)')
            plt.ylabel('ERP Amplitude')
            plt.title(f'ERP for {column} (Event {event})')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'{column}_event_{event}_{folder_name}.png'))
            plt.close()
            logging.info(f"Plot for channel {column} of event {event} saved.")

def main():
    print("Select the folder containing evoked files for all conditions")
    folder_path = select_folder("Select Folder Containing Evoked Files")
    
    log_file_path = setup_logging(folder_path, folder_path)
    logging.info(f"Log file created at {log_file_path}")
    
    data_files = load_data_files(folder_path)
    logging.info(f"Found {len(data_files)} data files.")
    
    erp_data = extract_erp_data_from_csv(data_files)
    avg_erp_data = compute_average_erp(erp_data)
    
    save_to_excel(avg_erp_data, folder_path)
    plot_erp_data(folder_path, avg_erp_data)

if __name__ == "__main__":
    main()
