"""
EEG Data Processing and Analysis Pipeline

This script automates the process of analyzing EEG data from BrainVision system files. 
It provides comprehensive steps for loading, preprocessing, and analyzing EEG signals to identify and characterize neurophysiological responses to stimuli. 
The script is designed for researchers in neuroscience and related fields who need to process EEG data efficiently while maintaining high standards for data integrity and reproducibility.

Inputs:
- A BrainVision EEG file (.vhdr) that contains raw EEG data.
- A directory path where the output (figures, logs, and processed data files) will be saved.

Processing Steps:
1. File Selection: The user selects the EEG file and the output directory through a graphical interface.
2. Logging Setup: Configures logging to track all processing steps and errors, saving logs to a designated output directory.
3. Data Loading and Preprocessing: The raw EEG data is loaded, filtered, and preprocessed to remove artifacts and ensure data quality.
4. Independent Component Analysis (ICA): Applies ICA to identify and remove artifact components from the EEG signals.
5. Epoching: Segments the continuous EEG data into epochs centered around event markers, applying criteria to reject noise and artifact-laden epochs.
6. Analysis: Performs various analyses including time-frequency decomposition, power spectral density estimation, and computation of event-related potentials (ERPs).
7. Visualization: Generates and saves plots of the processed data, ICA components, and analysis results to help in interpreting the EEG responses.

Outputs:
- Log files documenting each step and any issues encountered.
- Cleaned EEG data files in FIF format.
- Figures illustrating the ICA components, ERP responses, and frequency analysis results.
- Excel and CSV files containing detailed analysis outputs such as average ERP measurements and power spectral densities.

How to Run:
Ensure that all required libraries are installed, including MNE-Python and its dependencies. 
Run the script from a command line interface or through an IDE that supports Python execution. 
Follow the prompts to select the EEG file and output directory.

Author: Soumya Kaim
Date: 8th August 2024
Version: final version
"""
import datetime
import mne
import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
import mne
#print(dir(mne)) # check if all the required libraries are installed properly or not 
from mne.viz import plot_topomap
from mne.preprocessing import ICA
from mne.time_frequency import tfr_morlet
from mne_icalabel import label_components  # Ensure this is properly installed and configured
import warnings
'''
import sklearn
from sklearn.decomposition import FastICA
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
'''


# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def select_file():
    """ Open a file dialog to select an EEG file. """
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select EEG File", filetypes=[("BrainVision files", "*.vhdr")])
    root.destroy()
    logging.info(f"Selected EEG file: {file_path}")
    return file_path

def select_output_directory():
    """ Open a file dialog to select the output directory. """
    root = tk.Tk()
    root.withdraw()
    output_dir = filedialog.askdirectory(title="Select Output Directory")
    root.destroy()
    logging.info(f"Selected output directory: {output_dir}")
    return output_dir

def setup_logging(output_dir, eeg_file_path):
    """ Set up logging to file and console. """
    current_datetime = datetime.datetime.now()
    formatted_date_time = current_datetime.strftime("%Y-%m-%d_%Hhr-%Mmins-%Ssec")
    
    # Extract the basename of the EEG file
    eeg_file_basename = os.path.basename(eeg_file_path)
    eeg_file_name, _ = os.path.splitext(eeg_file_basename)
    
    # Construct the log file path using the basename of the EEG file
    log_file_path = os.path.join(output_dir, f'{eeg_file_name}_processing_{formatted_date_time}.log')
    
    file_handler = logging.FileHandler(log_file_path, mode='w')
    console_handler = logging.StreamHandler()
    logger = logging.getLogger()
    
    # Set logging level to INFO to avoid debug statements
    logger.setLevel(logging.INFO)
    
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Redirect MNE logging to the Python logger
    configure_mne_logging(logger)
    
    return log_file_path

def configure_mne_logging(logger):
    """ Redirect MNE logging to the Python logger. """
    class MNEHandler(logging.Handler):
        def emit(self, record):
            # Redirect the record to the given logger
            logger.handle(record)
    
    mne.set_log_level("INFO")
    handler = MNEHandler()
    mne_logger = mne.utils.logger
    mne_logger.addHandler(handler)
    mne_logger.setLevel(logging.INFO)

def load_and_preprocess(file_path):
    logging.info(f"Loading raw data from {file_path}")
    try:
        raw = mne.io.read_raw_brainvision(file_path, preload=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]  # Assign base_name here

        logging.info(f"Loaded raw data with {len(raw.ch_names)} channels and {raw.n_times} time points. Sampling frequency: {raw.info['sfreq']} Hz")
    except Exception as e:
        logging.error(f"Failed to load raw data: {e}")
        raise

     # Crop the raw data to remove the first 200 seconds
    try:
        raw.crop(tmin=200)
        logging.info(f"Cropped the first 200 seconds from the raw data. New start time: {raw.times[0]} seconds.")
    except Exception as e:
        logging.error(f"Failed to crop raw data: {e}")
        raise

    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')
        logging.info(f"Montage set successfully with standard_1020.")
    except Exception as e:
        logging.error(f"Failed to set montage: {e}")
        raise

    try:
        raw.filter(l_freq=1.0, h_freq=45.0)
        raw.notch_filter(freqs=50.0)
        logging.info("Data filtered successfully: high-pass at 1.0 Hz, low-pass at 45.0 Hz, notch filter at 50.0 Hz.")
    except Exception as e:
        logging.error(f"Failed to filter data: {e}")
        raise

    eog_channels = ['A1', 'A2', 'EOGH', 'EOGV']
    channel_types = {ch: 'eog' for ch in eog_channels if ch in raw.ch_names}
    raw.set_channel_types(channel_types)
    logging.info(f"Set EOG channel types: {channel_types}")

    exclude_channels = ['35', '36', '37', '38', '39', '40', '43', '44', '45', '46', '47', '48', '49', '50', 
                        '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64']
    raw.drop_channels([ch for ch in exclude_channels if ch in raw.ch_names])
    logging.info(f"Excluded channels: {exclude_channels}")

    return raw, base_name

def apply_ica(raw, output_dir, base_name):
    """
    Apply ICA (Independent Component Analysis) to the given raw data and 
    save the ICA object and component plots.
    
    Args:
        raw (mne.io.Raw): The raw data to apply ICA to.
        output_dir (str): The directory to save the ICA object and component plots.
        base_name (str): The base name for the saved ICA object and component plots.
    
    Returns:
        mne.io.Raw: The raw data after applying ICA.
    
    Raises:
        Exception: If there is an error during ICA fitting.
        AttributeError: If ICA fitting does not produce the 'components_' attribute.
    """
    
    logging.info("Starting ICA fitting")
    
    ica = ICA(n_components=None, method='fastica', max_iter=800, random_state=42)
    raw_for_ica = raw.copy().filter(l_freq=1, h_freq=None)
    logging.info(f"Filtered raw data for ICA: high-pass at 1 Hz, method {ica.method}")

    try:
        ica.fit(raw_for_ica)
        logging.info(f"ICA fitting completed with {ica.n_components_} components.")
    except Exception as e:
        logging.error(f"Error during ICA fitting: {e}")
        raise

    if not hasattr(ica, 'components_'):
        if hasattr(ica, 'pca_components_'):
            ica.components_ = ica.pca_components_
            logging.info('Using pca_components_ as ICA components.')
        else:
            logging.error("ICA fitting did not produce 'components_' attribute.")
            raise AttributeError("ICA fitting did not produce 'components_' attribute.")
        
    ica_file_path = os.path.join(output_dir, f'{base_name}_ica.fif')
    ica.save(ica_file_path, overwrite=True)
    logging.info(f"ICA object saved at {ica_file_path}")

    ica_figs = ica.plot_components(show=False)
    for i, fig in enumerate(ica_figs):
        fig.savefig(os.path.join(output_dir, f'ica_component_{i}_{base_name}.png'))
        plt.close(fig)
        logging.info(f"ICA component {i} figure saved.")

    for i in range(len(ica.components_)):
        if i >= ica.n_components_:
            logging.warning(f"Skipping plot_properties for component {i} as it exceeds the number of components {ica.n_components_}.")
            continue
        prop_fig = ica.plot_properties(raw, picks=[i], show=False)
        for fig in prop_fig:
            fig.savefig(os.path.join(output_dir, f'ica_properties_component_{i}_{base_name}.png'))
            plt.close(fig)
            logging.info(f"ICA properties for component {i} figure saved.")

    eog_inds, scores = ica.find_bads_eog(raw)
    ica.exclude.extend(eog_inds)
    logging.info(f"Automatically excluded EOG-related components: {eog_inds}")

    eog_scores_fig = ica.plot_scores(scores, show=False)
    eog_scores_fig.savefig(os.path.join(output_dir, f'ica_eog_scores_{base_name}.png'))
    plt.close(eog_scores_fig)
    logging.info(f"EOG scores figure saved.")

    excluded_components = input("Enter the components to exclude (comma-separated, or press Enter to skip): ")
    if excluded_components.strip():
        manual_exclude = [int(x) for x in excluded_components.split(',')]
        ica.exclude.extend(manual_exclude)
        logging.info(f"Manually excluded components: {manual_exclude}")

    raw = ica.apply(raw)
    raw.plot(duration=5, n_channels=32, clipping=None)
    logging.info(f"Applied ICA to raw data.")

    ref_channels = [ch for ch in ['A1', 'A2'] if ch in raw.ch_names]
    if ref_channels:
        raw.set_eeg_reference(ref_channels=ref_channels)
        logging.info(f"Reference channels set to {ref_channels}")

    return raw


def epoch_and_save(raw, base_name, output_dir):
    """
    Creates epochs from raw data, rejects bad epochs based on criteria, 
    and saves the epochs.

    Args:
        raw (mne.io.Raw): The raw data to create epochs from.
        base_name (str): The base name used for saving the epochs.
        output_dir (str): The directory to save the epochs.

    Returns:
        mne.Epochs: The epochs object containing the created epochs.

    """

    logging.info(f"Creating epochs for {base_name}")
    events, event_id = mne.events_from_annotations(raw)
    logging.info(f"Detected {len(events)} events with event IDs: {event_id}")

    epochs = mne.Epochs(raw, events, event_id, tmin=-1, tmax=2, baseline=(-0.3, 0), preload=True)

    # Reject bad epochs based on amplitude and flatline criteria
    reject_criteria = dict(eeg=100e-6)  # 100 µV
    flat_criteria = dict(eeg=1e-6)  # 1 µV
    initial_epochs_count = len(epochs)
    epochs.drop_bad(reject=reject_criteria, flat=flat_criteria)
    dropped_epochs_count = initial_epochs_count - len(epochs)
    logging.info(f"Dropped {dropped_epochs_count} bad epochs. Number of epochs remaining: {len(epochs)}")

    if len(epochs) == 0:
        logging.error("No epochs created. Skipping...")
        return None

    epochs_save_path = os.path.join(output_dir, f'{base_name}_clean-epo.fif')
    epochs.save(epochs_save_path, overwrite=True)
    logging.info(f"Epochs saved at {epochs_save_path}")

    return epochs


def time_frequency_analysis(epochs):
    """
    Perform time-frequency analysis on the given epochs.

    Parameters:
        epochs: The epochs to perform time-frequency analysis on.

    Returns:
        tuple: A tuple containing two objects:
            - power: The power spectrum of the epochs.
            - itc: The inter-trial coherence of the epochs.
    """

    logging.info("Performing time-frequency analysis")
    freqs = np.logspace(*np.log10([4, 30]), num=10)
    n_cycles = freqs / 2.0
    power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True)
    logging.info(f"Time-frequency analysis completed with frequencies: {freqs}")
    return power, itc

def frequency_analysis(epochs_clean, output_dir, base_name):
    """
    Performs frequency analysis on the provided epochs.

    Parameters:
        epochs_clean (mne.Epochs): The clean epochs to perform frequency analysis on.
        output_dir (str): The directory to save the frequency analysis results.
        base_name (str): The base name used for saving the frequency analysis results.

    """

    logging.info("Performing frequency analysis")

    def plot_and_save_psd(freq_band, fmin, fmax, psds, freqs, title, filename):
        fig, ax = plt.subplots(figsize=(10, 8))
        for psd in psds:
            ax.plot(freqs, 10 * np.log10(psd.T), alpha=0.5)
        ax.set_xlim(fmin, fmax)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density (dB/Hz)')
        ax.set_title(title)
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)
        logging.info(f"{title} PSD figure saved at {filename}")

    n_times = len(epochs_clean.times)
    n_fft = min(2048, n_times)
    logging.info(f"Computing PSD using Welch method with n_fft={n_fft}")

    psd_welch = epochs_clean.compute_psd(method='welch', fmin=4, fmax=45, n_fft=n_fft)
    psds, freqs = psd_welch.get_data(return_freqs=True)
    psd_df = pd.DataFrame(psds.mean(axis=0), columns=freqs)
    psd_df.to_csv(os.path.join(output_dir, f'psd_data_{base_name}.csv'), index=False)
    logging.info(f"PSD data saved at {os.path.join(output_dir, f'psd_data_{base_name}.csv')}")

    plot_and_save_psd('Theta', 4, 8, psds, freqs, 'Theta Band (4-8 Hz)', f'theta_psd_{base_name}.png')
    plot_and_save_psd('Alpha', 8, 12, psds, freqs, 'Alpha Band (8-12 Hz)', f'alpha_psd_{base_name}.png')
    plot_and_save_psd('Beta', 12, 30, psds, freqs, 'Beta Band (12-30 Hz)', f'beta_psd_{base_name}.png')

def erp_analysis(epochs_clean, output_dir, base_name):
    """
    Perform ERP analysis on the given clean epochs.

    Args:
        epochs_clean (mne.Epochs): The clean epochs to perform ERP analysis on.
        output_dir (str): The directory to save the ERP analysis results.
        base_name (str): The base name used for saving the ERP analysis results.

    Returns:
        None

    This function calculates the evoked responses for Stimulus 64 and 65 in the given clean epochs.
    It then saves and plots the evoked responses for both stimuli.
    The evoked responses are also saved as CSV files and FIF files.
    Finally, the function logs the completion of ERP analysis and file saving.
    """

    logging.info(f"Calculating ERP for {base_name}")
    evoked_64 = epochs_clean['Stimulus/S 64'].average()
    evoked_65 = epochs_clean['Stimulus/S 65'].average()
    logging.info(f"Evoked responses calculated for Stimulus 64 and 65.")

    save_and_plot_evoked(evoked_64, output_dir, base_name, '64')
    save_and_plot_evoked(evoked_65, output_dir, base_name, '65')

    save_evoked_to_csv(evoked_64, f'evoked_64_{base_name}.csv', output_dir)
    save_evoked_to_csv(evoked_65, f'evoked_65_{base_name}.csv', output_dir)

    evoked_64.save(os.path.join(output_dir, f'evoked_64_{base_name}-ave.fif'), overwrite=True)
    evoked_65.save(os.path.join(output_dir, f'evoked_65_{base_name}-ave.fif'), overwrite=True)
    logging.info("ERP analysis and file saving completed.")

def save_evoked_to_csv(evoked, filename, output_dir):
    evoked_df = evoked.to_data_frame()
    file_path = os.path.join(output_dir, filename)
    evoked_df.to_csv(file_path, index=False)
    logging.info(f"ERP data saved to CSV at {file_path}")

def save_and_plot_evoked(evoked, output_dir, base_name, tag):
    evoked_fig = evoked.plot(show=False)
    fig_path = os.path.join(output_dir, f'{base_name}_{tag}_evoked.png')
    evoked_fig.savefig(fig_path)
    plt.close(evoked_fig)
    logging.info(f"Evoked plot for {tag} saved at {fig_path}")

def segment_and_analyze_epochs(epochs, raw, output_dir, base_name):
    logging.info("Segmenting and analyzing epochs")
    epochs_fig_path = os.path.join(output_dir, f'epochs_{base_name}.png')
    epochs_fig = epochs.plot(n_epochs=4, show=False)
    epochs_fig.savefig(epochs_fig_path)
    plt.close(epochs_fig)
    logging.info(f"Epochs figure saved at {epochs_fig_path}")

def main():
    """
    The main function is the entry point of the EEG analysis pipeline.
    It orchestrates the entire process, from file selection and logging setup 
    to data preprocessing, epoching, and various analyses.

    """

    file_path = select_file()
    output_dir = select_output_directory()
    log_file_path = setup_logging(output_dir, file_path)  # Pass the EEG file path
    raw, base_name = load_and_preprocess(file_path)
    raw = apply_ica(raw, output_dir, base_name)
    
    epochs_clean = epoch_and_save(raw, base_name, output_dir)
    if epochs_clean is not None:
        power, itc = time_frequency_analysis(epochs_clean)
        erp_analysis(epochs_clean, output_dir, base_name)
        frequency_analysis(epochs_clean, output_dir, base_name)
        segment_and_analyze_epochs(epochs_clean, raw, output_dir, base_name)
        logging.info("Processing completed successfully.")
    else:
        logging.error("Epochs cleaning failed.")

if __name__ == "__main__":
    main()
