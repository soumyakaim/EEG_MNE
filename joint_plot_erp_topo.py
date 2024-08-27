import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

def load_erp_data(file_path):
    """Load ERP data from a CSV or Excel file."""
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:  # Assuming Excel file
        data = pd.read_excel(file_path)

    times = data.iloc[:, 0].values
    channels = data.columns[1:]
    erp_data = data.iloc[:, 1:].values.T  # Transpose to get channels x timepoints
    return erp_data, times, channels

def create_mne_evoked(erp_data, times, channels):
    """Create an MNE Evoked object from ERP data."""
    info = mne.create_info(channels.tolist(), sfreq=1/(times[1] - times[0]), ch_types='eeg')
    evoked = mne.EvokedArray(erp_data, info, tmin=times[0])
    montage = mne.channels.make_standard_montage('standard_1020')
    evoked.set_montage(montage)
    return evoked

def find_dynamic_time_points(evoked, time_window=(-0.1, 0.6)):
    """
    This function dynamically finds four time points of interest in an evoked response signal:

    1. Event onset (0 ms)
    2. Peak amplitude within a specified time window
    3. 100 ms before the peak (pre-peak time)
    4. 100 ms after the peak (post-peak time)
    
    It returns these time points in ascending order, suitable for plotting.
        
    Args:
        evoked (mne.Evoked): The evoked response to analyze.
        time_window (tuple): The time window to search for the peak (start, end).
    
    Returns:
        list: Ordered time points for plotting.
    """
    times = evoked.times
    data = evoked.data
    
    # Find the index corresponding to the time window
    start_idx = np.searchsorted(times, time_window[0])
    end_idx = np.searchsorted(times, time_window[1])
    
    # Find peak amplitude within the window
    peak_idx = np.abs(data[:, start_idx:end_idx]).mean(axis=0).argmax() + start_idx
    peak_time = times[peak_idx]
    
    # Define other time points of interest
    onset_time = 0  # Event onset
    pre_peak_time = peak_time - 0.1 if peak_time - 0.1 > times[0] else times[0]  # 100 ms before peak
    post_peak_time = peak_time + 0.1 if peak_time + 0.1 < times[-1] else times[-1]  # 100 ms after peak
    
    # Ensure time points are ordered correctly
    time_points = sorted([pre_peak_time, onset_time, peak_time, post_peak_time])
    
    return time_points


def plot_joint_erp_topomap(evoked, title, output_dir, base_name, event_tag):
    """Plot joint ERP and topomap for dynamically determined time points."""
    times_to_plot = find_dynamic_time_points(evoked)
    fig = evoked.plot_joint(times=times_to_plot, title=title, show=False)
    fig.savefig(os.path.join(output_dir, f'{base_name}_event_{event_tag}_joint_plot.png'))
    plt.close(fig)

def process_condition_event(condition_folder, event_id, output_dir):
    """Process ERP data for a given condition and event."""
    file_name = f'average_erp_data_{os.path.basename(condition_folder)}_event_{event_id}.xlsx'
    file_path = os.path.join(condition_folder, file_name)

    erp_data, times, channels = load_erp_data(file_path)
    evoked = create_mne_evoked(erp_data, times, channels)

    plot_joint_erp_topomap(evoked, f'ERP and Topography - Event {event_id} - {os.path.basename(condition_folder)}', 
                           output_dir, os.path.basename(condition_folder), event_id)

def main():
    main_folder = r'E:\Academics\MajorThesis\DATA\Robotics\EEG_mainexp+pilot'
    output_dir = os.path.join(main_folder, 'output')

    condition_folders = ['keypress', 'robot_sync', 'robot_async']
    event_ids = ['64', '65']

    for condition in condition_folders:
        condition_folder = os.path.join(main_folder, condition)
        for event_id in event_ids:
            process_condition_event(condition_folder, event_id, output_dir)

if __name__ == "__main__":
    main()
