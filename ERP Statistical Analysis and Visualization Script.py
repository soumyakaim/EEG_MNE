"""
ERP Statistical Analysis and Visualization Script

How to Use:
1. Run the script and follow the prompts to select the folder containing the subfolders with ERP Excel files.
2. The script will automatically load the data, perform statistical tests, generate plots, and save all results in the specified output directory.
3. Review the output directory for Excel files containing statistical results and subdirectories containing visualizations of the data.

Logging:
- The script includes a logging system, set up in `setup_logging()`, that tracks the progress of the analysis, records detailed statistical computations, and captures any issues for easy troubleshooting.

Functionality:
- This script processes and analyzes ERP (Event-Related Potential) data stored in Excel files across different experimental conditions: `keypress`, `robot_sync`, and `robot_async`.
- The main functions of the script include:
  - select_folder(): Prompts the user to select a directory containing the subfolders for each condition.
  - setup_logging(): Sets up logging to track the analysis process and records it in a log file within the selected directory.
  - load_excel_files_from_folders(): Loads ERP data from the specified Excel files within each condition's subfolder.
  - perform_statistical_tests(): Conducts one-way ANOVA to compare ERP amplitudes across conditions and logs detailed statistical measures.
  - perform_tukey_hsd(): Performs Tukey's Honest Significant Difference (HSD) test to identify specific pairs of conditions with significant differences.
  - save_statistical_results(): Saves the statistical analysis results to Excel files.
  - summarize_significant_results(): Summarizes significant results from the Tukey HSD test into an Excel file.
  - plot_comparisons(): Generates comparison plots of ERP amplitudes across conditions and saves them as PNG files.
  - plot_facet_grid_by_event(): Creates advanced facet grid plots for visualizing significant differences between conditions, including confidence intervals.

Statistical Analysis:
- One-Way ANOVA (Performed in `perform_statistical_tests()` function):
  - Purpose: To test if there are significant differences in ERP amplitudes between the three conditions (`keypress`, `robot_sync`, and `robot_async`) for each EEG channel and event (`event_64` and `event_65`).
  - F-statistic: The ratio of variance between the group means to the variance within the groups, calculated to determine the significance of differences.
  - p-value: Assesses the significance of the differences; a p-value < 0.05 indicates statistically significant differences between the conditions.

- Detailed Statistical Measures (calculated in `perform_statistical_tests()`):
  - Means: Average ERP amplitude for each condition.
  - Variances: Variability of ERP amplitudes within each condition.
  - Sum of Squares (SS):
    - SS Between: Variability of the condition means from the overall mean.
    - SS Within: Variability within each condition.
  - Degrees of Freedom (df):
    - df Between: Number of conditions minus one.
    - df Within: Total number of observations minus the number of conditions.
  - Mean Squares (MS):
    - MS Between: SS Between divided by df Between.
    - MS Within: SS Within divided by df Within.

- Tukey's Honest Significant Difference (HSD) Test (Performed in `perform_tukey_hsd()` function):
  - Purpose: To determine which specific pairs of conditions have significant differences in mean ERP amplitudes.
  - Significance: Identifies condition pairs with statistically significant differences, providing detailed post-hoc analysis.

- Summarization and Visualization:
  - summarize_significant_results(): Summarizes significant results from Tukey's HSD into an Excel file.
  - plot_comparisons(): Generates and saves comparison plots showing mean and standard deviation of ERP amplitudes across conditions.
  - plot_facet_grid_by_event(): Creates facet grid plots to visualize significant mean differences between conditions, including error bars for confidence intervals.

 Input Requirements:
- Hardcoded:
  - Condition names: `keypress`, `robot_sync`, `robot_async`.
  - Event names: `event_64`, `event_65`.
  - The structure of subfolders and expected file formats (e.g., `.xlsx` files) are assumed to be consistent.

- Variable:
  - Directory paths: The folder containing the subfolders and Excel files is user-specified via the `select_folder()` function.
  - File names: The script looks for Excel files within the subfolders; specific file names are determined by their presence in the subfolders.

 Outputs:
- Statistical Results: 
  - Saved in Excel files using `save_statistical_results()`, with separate sheets for ANOVA and summary statistics for each event (`event_64`, `event_65`).
- Summary of Significant Results:
  - Tukey HSD results summarized and saved in an Excel file using `summarize_significant_results()`.
- Visualizations:
  - PNG files of ERP comparison plots are saved by `plot_comparisons()`.
  - Advanced facet grid plots saved by `plot_facet_grid_by_event()`.

Author: Soumya Kaim
Date: 13th August 2024
Version: Final

"""
import matplotlib
print(matplotlib.__version__)

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
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

def setup_logging(folder_path):
    current_datetime = datetime.datetime.now()
    formatted_date_time = current_datetime.strftime("%Y-%m-%d_%Hhr-%Mmins-%Ssec")
    log_file_path = os.path.join(folder_path, f'{os.path.basename(folder_path)}_comparison_{formatted_date_time}.log')

    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levellevel)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file_path, mode='w'),
                            logging.StreamHandler()
                        ])
    return log_file_path

def load_excel_files_from_folders(folder_path):
    data_files = {}
    for condition in ['keypress', 'robot_sync', 'robot_async']:
        condition_folder = os.path.join(folder_path, condition)
        if os.path.isdir(condition_folder):
            data_files[condition] = {'64': [], '65': []}
            for file_name in os.listdir(condition_folder):
                if file_name.endswith('.xlsx'):
                    if 'event_64' in file_name:
                        data_files[condition]['64'].append(os.path.join(condition_folder, file_name))
                    elif 'event_65' in file_name:
                        data_files[condition]['65'].append(os.path.join(condition_folder, file_name))
    return data_files

def load_grand_average_data(data_files):
    data = {'64': {}, '65': {}}
    for condition, events in data_files.items():
        for event, files in events.items():
            if files:
                data[event][condition] = pd.read_excel(files[0])  # Assuming there is only one grand average file per condition and event
    return data

def perform_statistical_tests(data):
    """
    Performs one-way ANOVA statistical tests on the provided data.

    Args:
        data (dict): A dictionary containing event data, where each event is a dictionary of conditions,
        and each condition is a pandas DataFrame with 'time' as the first column.

    Returns:
        tuple: A tuple of two dictionaries. The first dictionary contains the F-statistic and p-value for each channel in each event.
        The second dictionary contains detailed results for each channel in each event, including means, variances, sample sizes,
        overall mean, sum of squares, degrees of freedom, mean squares, F-statistic, and p-value.
    """

    stats_results = {'64': {}, '65': {}}
    detailed_results = {'64': {}, '65': {}}

    for event in ['64', '65']:
        conditions = list(data[event].keys())
        for ch in data[event][conditions[0]].columns[1:]:  # Skip the 'time' column
            samples = [data[event][cond][ch].values for cond in conditions]
            f_stat, p_val = f_oneway(*samples)

            # Calculate and log detailed steps
            means = [np.mean(sample) for sample in samples]
            variances = [np.var(sample, ddof=1) for sample in samples]
            ns = [len(sample) for sample in samples]
            overall_mean = np.mean([val for sublist in samples for val in sublist])

            logging.info(f"Calculations for Channel {ch} (Event {event}):")
            for cond, mean, var, n in zip(conditions, means, variances, ns):
                logging.info(f"{cond} - Mean: {mean}, Variance: {var}, N: {n}")
            logging.info(f"Overall Mean: {overall_mean}")

            # Sum of squares
            ss_between = sum(n * (mean - overall_mean)**2 for n, mean in zip(ns, means))
            ss_within = sum((n - 1) * var for n, var in zip(ns, variances))
            df_between = len(conditions) - 1
            df_within = sum(ns) - len(conditions)

            ms_between = ss_between / df_between
            ms_within = ss_within / df_within

            logging.info(f"SS Between: {ss_between}, SS Within: {ss_within}")
            logging.info(f"MS Between: {ms_between}, MS Within: {ms_within}")
            logging.info(f"F-statistic: {f_stat}, p-value: {p_val}")

            stats_results[event][ch] = (f_stat, p_val)
            detailed_results[event][ch] = {
                'means': means,
                'variances': variances,
                'ns': ns,
                'overall_mean': overall_mean,
                'ss_between': ss_between,
                'ss_within': ss_within,
                'df_between': df_between,
                'df_within': df_within,
                'ms_between': ms_between,
                'ms_within': ms_within
            }

    return stats_results, detailed_results

def perform_tukey_hsd(data, stats_results):
    """
    Performs Tukey's Honest Significant Difference (HSD) test on the provided data.

    Parameters:
        data (dict): A dictionary containing event data, where each event is a dictionary of conditions,
        and each condition is a pandas DataFrame with 'time' as the first column.
        stats_results (dict): A dictionary containing the results of previous statistical tests.

    Returns:
        dict: A dictionary containing the Tukey HSD test results, where each key represents an event and
        each value is another dictionary with channel names as keys and Tukey HSD test summaries as values.
    """

    tukey_results = {'64': {}, '65': {}}
    for event in ['64', '65']:
        conditions = list(data[event].keys())
        for ch in data[event][conditions[0]].columns[1:]:  # Skip the 'time' column
            df = pd.concat([data[event][cond][['time', ch]].assign(condition=cond) for cond in conditions])
            tukey = pairwise_tukeyhsd(endog=df[ch], groups=df['condition'], alpha=0.05)
            tukey_results[event][ch] = tukey.summary()
    return tukey_results

def save_statistical_results(stats_results, detailed_results, tukey_results, data, output_dir):
    """
    Saves statistical results to an Excel file.

    Args:
        stats_results (dict): A dictionary containing the statistical results for each event and channel.
        detailed_results (dict): A dictionary containing detailed statistical results for each event and channel.
        tukey_results (dict): A dictionary containing Tukey HSD results for each event and channel.
        data (dict): A dictionary containing the data for each event and condition.
        output_dir (str): The directory to save the Excel file.

    Returns:
        None

    This function saves the statistical results and detailed results to separate Excel sheets.
    The ANOVA results are saved to one sheet for each event, and the summary statistics are saved to another sheet.
    The Tukey HSD results are saved to separate sheets for each event, with one sheet for each channel.
    The Excel file is saved in the specified output directory.
    """

    with pd.ExcelWriter(os.path.join(output_dir, 'statistical_results.xlsx'), engine='xlsxwriter') as writer:
        for event in stats_results.keys():
            anova_rows = []
            summary_stats = []
            for ch, (f_stat, p_val) in stats_results[event].items():
                detail = detailed_results[event][ch]
                interpretation = "Significant" if p_val < 0.05 else "Not significant"
                anova_row = {
                    'Channel': ch,
                    'F-statistic': f_stat,
                    'p-value': p_val,
                    'SS Between': detail['ss_between'],
                    'SS Within': detail['ss_within'],
                    'DF Between': detail['df_between'],
                    'DF Within': detail['df_within'],
                    'MS Between': detail['ms_between'],
                    'MS Within': detail['ms_within'],
                    'Interpretation': interpretation
                }
                anova_rows.append(anova_row)

                for condition, mean, var, n in zip(data[event].keys(), detail['means'], detail['variances'], detail['ns']):
                    summary_stats.append({
                        'Event': event,
                        'Condition': condition,
                        'Channel': ch,
                        'Mean': mean,
                        'Std Deviation': np.sqrt(var),
                        'N': n
                    })

            df_anova = pd.DataFrame(anova_rows)
            df_summary = pd.DataFrame(summary_stats)
            df_anova.to_excel(writer, sheet_name=f'Event_{event}_ANOVA', index=False)
            df_summary.to_excel(writer, sheet_name=f'Event_{event}_Summary', index=False)
            logging.info(f'Statistical results for event {event} saved to Excel sheet.')

        # Save Tukey HSD results to Excel
        for event, channels in tukey_results.items():
            for ch, summary in channels.items():
                df_tukey = pd.read_html(summary.as_html())[0]
                df_tukey.to_excel(writer, sheet_name=f'Tukey_{ch}_event_{event}', index=False)

def summarize_significant_results(tukey_results):
    """
    Generate a summary of significant results from the given Tukey results.

    Args:
        tukey_results (dict): A dictionary containing the Tukey results, where each key is an event and each value is another dictionary with channel names as keys and Tukey summary as values.

    Returns:
        pandas.DataFrame: A DataFrame containing the summary of significant results. Each row represents a significant result and includes the following columns:
            - Event: The event name.
            - Channel: The channel name.
            - Group1: The name of the first group.
            - Group2: The name of the second group.
            - Mean Difference: The mean difference between the two groups.
            - p-adj: The adjusted p-value.
            - Lower: The lower bound of the confidence interval.
            - Upper: The upper bound of the confidence interval.
    """

    significant_results = []
    for event, channels in tukey_results.items():
        for ch, summary in channels.items():
            df = pd.read_html(summary.as_html())[0]
            sig_df = df[df['reject'] == True]
            for _, row in sig_df.iterrows():
                significant_results.append({
                    'Event': event,
                    'Channel': ch,
                    'Group1': row['group1'],
                    'Group2': row['group2'],
                    'Mean Difference': row['meandiff'],
                    'p-adj': row['p-adj'],
                    'Lower': row['lower'],
                    'Upper': row['upper']
                })
    return pd.DataFrame(significant_results)

def plot_comparisons(data, stats_results, output_dir):
    """
    Plots comparisons of ERP amplitudes across different conditions for each channel and event.

    Parameters:
        data (dict): A dictionary containing the data to be plotted, 
            where each key represents an event and each value is another dictionary with condition names as keys and 
            pandas DataFrames as values.
        stats_results (dict): A dictionary containing the results of previous statistical tests, 
            where each key represents an event and each value is another dictionary with channel names as keys and 
            statistical results as values.
        output_dir (str): The directory where the plots will be saved.

    Returns:
        None
    """

    for event in ['64', '65']:
        event_output_dir = os.path.join(output_dir, f'event_{event}')
        os.makedirs(event_output_dir, exist_ok=True)
        time = data[event][list(data[event].keys())[0]]['time'].values

        for ch in data[event][list(data[event].keys())[0]].columns[1:]:  # Skip the 'time' column
            plt.figure()
            for cond in data[event].keys():
                mean = data[event][cond][ch].values
                std_dev = data[event][cond][ch].std()
                plt.plot(time, mean, label=f'{cond} Mean')
                plt.fill_between(time, mean - std_dev, mean + std_dev, alpha=0.2, label=f'{cond} SD')
            
            plt.xlabel('Time (s)')
            plt.ylabel('ERP Amplitude')
            plt.title(f'ERP Comparison for {ch} (Event {event})')
            plt.legend()
            plt.savefig(os.path.join(event_output_dir, f'comparison_{ch}_event_{event}.png'))
            plt.close()

            # Print statistical results
            logging.info(f'Statistical results for {ch} (Event {event}):')
            f_stat, p_val = stats_results[event][ch]
            logging.info(f'F-statistic={f_stat:.2f}, p-value={p_val:.5f}')

    
def plot_facet_grid_by_event(significant_results, output_dir):
    """
    Generate a facet grid plot of significant mean differences by channel and event.

    Parameters:
        significant_results (pandas.DataFrame): A DataFrame containing the significant results, including 'Event', 'Channel', 'Group1', 'Group2', 'Mean Difference', 'p-adj', 'Lower', and 'Upper' columns.
        output_dir (str): The directory where the plot will be saved.

    Returns:
        None
    """

    # Convert 'Event' to string for better handling in seaborn
    significant_results['Event'] = significant_results['Event'].astype(str)

    # Set up the facet grid
    g = sns.FacetGrid(significant_results, col="Channel", hue="Group1", col_wrap=4, height=4, aspect=1.5, palette='tab10')

    # Map the barplot onto the grid
    g.map(sns.barplot, "Event", "Mean Difference", order=sorted(significant_results['Event'].unique()), ci=None)

    # Add error bars manually
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(90)
            label.set_ha('right')
        ch = ax.get_title().split(' = ')[-1]
        event_data = significant_results[significant_results['Channel'] == ch]
        for idx, row in event_data.iterrows():
            ax.errorbar(
                x=row['Event'],
                y=row['Mean Difference'],
                yerr=[[row['Mean Difference'] - row['Lower']], [row['Upper'] - row['Mean Difference']]],
                fmt='none',
                c='black'
            )

    # Adjust the plot
    g.add_legend()
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.set_axis_labels("Event", "Mean Difference")
    g.fig.suptitle('Significant Mean Differences and 95% CI by Channel and Event', y=1.02)

    # Save the plot
    plt.tight_layout()
    g.savefig(os.path.join(output_dir, 'facet_grid_significant_mean_differences.png'))
    plt.show()

def main():
    """
    The main function of the program, responsible for executing the main logic of the program.

    It performs the following steps:
    1. Prompts the user to select a folder containing condition folders.
    2. Sets up logging and creates a log file.
    3. Loads the Excel files from the selected folder.
    4. Logs the found data files.
    5. Loads the grand average data from the loaded files.
    6. Performs statistical tests on the loaded data.
    7. Creates a directory for comparison plots.
    8. Plots the comparisons of ERP amplitudes.
    9. Performs Tukey's HSD test.
    10. Saves the statistical results.
    11. Logs the completion of the process.

    Parameters:
        None

    Returns:
        None
    """

    print("Select the folder containing the condition folders (keypress, robot_sync, robot_async)")
    folder_path = select_folder("Select Folder Containing Condition Folders")
    
    log_file_path = setup_logging(folder_path)
    logging.info(f"Log file created at {log_file_path}")
    
    data_files = load_excel_files_from_folders(folder_path)
    logging.info(f"Found data files for conditions: {list(data_files.keys())}")
    
    data = load_grand_average_data(data_files)
    stats_results, detailed_results = perform_statistical_tests(data)
    
    output_dir = os.path.join(folder_path, 'comparison_plots')
    os.makedirs(output_dir, exist_ok=True)
    plot_comparisons(data, stats_results, output_dir)
    
    tukey_results = perform_tukey_hsd(data, stats_results)
    save_statistical_results(stats_results, detailed_results, tukey_results, data, output_dir)

    significant_results = summarize_significant_results(tukey_results)
    significant_results.to_excel(os.path.join(output_dir, 'significant_results_summary.xlsx'), index=False)
    plot_facet_grid_by_event(significant_results, output_dir)
    
    logging.info(f'Comparison plots and statistical results saved to {output_dir}')

if __name__ == "__main__":
    main()

