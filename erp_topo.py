import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import mne
import os

# Define your base directory
base_dir = r"E:\Academics\MajorThesis\DATA\Robotics\EEG_mainexp+pilot"

# Define output directory for the red-white-blue color scheme
output_dir_rwb = os.path.join(base_dir, "topo_plots_key_timepoints_rwb")
os.makedirs(output_dir_rwb, exist_ok=True)

# Voltage ranges based on the statistical summary provided (rounded)
voltage_ranges = {
    ('keypress', 64): (-2, 10),
    ('keypress', 65): (-3, 3),
    ('robot_sync', 64): (-2, 6),
    ('robot_sync', 65): (-2, 4),
    ('robot_async', 64): (-2, 9),
    ('robot_async', 65): (-8, 4)
}

# Define the conditions and the corresponding files
conditions = {
    'keypress': [os.path.join(base_dir, 'keypress', 'average_erp_data_keypress_event_64.xlsx'),
                 os.path.join(base_dir, 'keypress', 'average_erp_data_keypress_event_65.xlsx')],
    'robot_sync': [os.path.join(base_dir, 'robot_sync', 'average_erp_data_robot_sync_event_64.xlsx'),
                   os.path.join(base_dir, 'robot_sync', 'average_erp_data_robot_sync_event_65.xlsx')],
    'robot_async': [os.path.join(base_dir, 'robot_async', 'average_erp_data_robot_async_event_64.xlsx'),
                    os.path.join(base_dir, 'robot_async', 'average_erp_data_robot_async_event_65.xlsx')]
}

# Load a standard 10-20 montage for electrode positions
montage = mne.channels.make_standard_montage('standard_1020')

# Time points of interest
baseline_time = -0.3  # 300 ms before event onset
event_onset_time = 0.0  # Event onset
post_peak_time = 0.2  # Example post-peak time (adjust based on your data)

# Process data for each condition and event type
for condition, file_paths in conditions.items():
    for event_idx, file_path in enumerate(file_paths):
        """
        Process event data for each condition and event type.

        Parameters:
        file_paths (list): A list of file paths for the event data.
        conditions (dict): A dictionary containing conditions as keys and lists of file paths as values.
        montage (object): An MNE montage object.
        output_dir_rwb (str): The output directory for the RWB plots.
        voltage_ranges (dict): A dictionary containing voltage ranges for each condition and event.

        Returns:
        None

        Notes:
        This function processes event data for each condition and event type, creates topographic plots for key time points,
        and saves the plots to the output directory.
        """
        # Load data from Excel
        df = pd.read_excel(file_path)
        data = df.values  # Data should be (n_channels, n_samples)
        
        # Adjust the scale if necessary (assuming the data might be in microvolts)
        data = data * 1e-6  # Convert from µV to V if necessary
        
        # Transpose data if necessary
        if data.shape[0] == 1501:
            data = data.T  # Convert to (n_channels, n_samples)
        
        # Create an MNE Info object with correct channel names and types
        ch_names = montage.ch_names[:data.shape[0]]  # Ensure we match the number of channels
        info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg')
        info.set_montage(montage)
        
        # Create the Evoked object
        evoked = mne.EvokedArray(data, info, tmin=-0.3)  # tmin adjusted to match -300 ms baseline
        
        # Find the peak amplitude time point across all channels
        peak_time = evoked.get_peak(ch_type='eeg', mode='pos')[1]  # Time of peak amplitude
        
        # Time points of interest
        time_points = [baseline_time, event_onset_time, peak_time, post_peak_time]
        time_labels = ['Baseline', 'Event Onset', 'Peak Amplitude', 'Post-Peak']
        
        # Determine the voltage range for this condition and event
        voltage_min, voltage_max = voltage_ranges[(condition, 64 + event_idx)]
        
        # Create output subdirectory for the event (event 64 or event 65)
        event_folder_rwb = os.path.join(output_dir_rwb, f"event_{64 + event_idx}")
        os.makedirs(event_folder_rwb, exist_ok=True)
        
        # Plot the topographies for the key time points with RWB color scheme
        fig_rwb, axes_rwb = plt.subplots(1, len(time_points), figsize=(15, 4))
        divider_rwb = make_axes_locatable(axes_rwb[-1])
        cax_rwb = divider_rwb.append_axes("right", size="5%", pad=0.05)
        
        for i, (time, label) in enumerate(zip(time_points, time_labels)):
            im_rwb = evoked.plot_topomap(times=[time], ch_type='eeg', axes=axes_rwb[i], show=False, contours=0, colorbar=False, cmap='RdBu_r', vlim=(voltage_min, voltage_max))
            axes_rwb[i].set_title(f"{label} ({int(time * 1000)} ms)")
        
        # Add colorbar with units (e.g., microvolts)
        cbar = plt.colorbar(axes_rwb[-1].images[-1], cax=cax_rwb)
        cbar.set_label('Amplitude (µV)', rotation=270, labelpad=15)

        # Add overall title with condition and event information
        fig_rwb.suptitle(f"{condition.capitalize()} Condition - Event {64 + event_idx}", fontsize=16)
        
        # Save the figure
        filename_rwb = f"{condition}_event_{64 + event_idx}_key_timepoints.png"
        filepath_rwb = os.path.join(event_folder_rwb, filename_rwb)
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Adjust layout to fit title
        plt.savefig(filepath_rwb)
        plt.close(fig_rwb)
        
        print(f"Saved topomap plot for {condition} in {filepath_rwb}")

print("All plots saved successfully.")











'''
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import mne
import os

# Define your base directory
base_dir = r"E:\Academics\MajorThesis\DATA\Robotics\EEG_mainexp+pilot"

# Define output directories
output_dir = os.path.join(base_dir, "topo_plots_key_timepoints")
os.makedirs(output_dir, exist_ok=True)

# Define the conditions and the corresponding files
conditions = {
    'keypress': [os.path.join(base_dir, 'keypress', 'average_erp_data_keypress_event_64.xlsx'),
                 os.path.join(base_dir, 'keypress', 'average_erp_data_keypress_event_65.xlsx')],
    'robot_sync': [os.path.join(base_dir, 'robot_sync', 'average_erp_data_robot_sync_event_64.xlsx'),
                   os.path.join(base_dir, 'robot_sync', 'average_erp_data_robot_sync_event_65.xlsx')],
    'robot_async': [os.path.join(base_dir, 'robot_async', 'average_erp_data_robot_async_event_64.xlsx'),
                    os.path.join(base_dir, 'robot_async', 'average_erp_data_robot_async_event_65.xlsx')]
}

# Load a standard 10-20 montage for electrode positions
montage = mne.channels.make_standard_montage('standard_1020')

# Time points of interest
baseline_time = -0.3  # 300 ms before event onset
event_onset_time = 0.0  # Event onset
post_peak_time = 0.4  # Example post-peak time (adjust based on your data)

# Process data for each condition and event type
for condition, file_paths in conditions.items():
    for event_idx, file_path in enumerate(file_paths):
        # Load data from Excel
        df = pd.read_excel(file_path)
        data = df.values  # Data should be (n_channels, n_samples)
        
        # Adjust the scale if necessary (assuming the data might be in microvolts)
        data = data * 1e-6  # Convert from µV to V if necessary
        
        # Transpose data if necessary
        if data.shape[0] == 1501:
            data = data.T  # Convert to (n_channels, n_samples)
        
        # Create an MNE Info object with correct channel names and types
        ch_names = montage.ch_names[:data.shape[0]]  # Ensure we match the number of channels
        info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg')
        info.set_montage(montage)
        
        # Create the Evoked object
        evoked = mne.EvokedArray(data, info, tmin=-0.3)  # tmin adjusted to match -300 ms baseline
        
        # Find the peak amplitude time point across all channels
        peak_time = evoked.get_peak(ch_type='eeg', mode='pos')[1]  # Time of peak amplitude
        
        # Time points of interest
        time_points = [baseline_time, event_onset_time, peak_time, post_peak_time]
        time_labels = ['Baseline', 'Event Onset', 'Peak Amplitude', 'Post-Peak']
        
        # Create output subdirectory for the event (event 64 or event 65)
        event_folder = os.path.join(output_dir, f"event_{64 + event_idx}")
        os.makedirs(event_folder, exist_ok=True)
        
        # Plot the topographies for the key time points
        fig, axes = plt.subplots(1, len(time_points), figsize=(15, 4))
        
        # Create a separate axis for the colorbar using a divider
        divider = make_axes_locatable(axes[-1])
        cax = divider.append_axes("right", size="5%", pad=0.05)  # Adjust size and padding as needed
        
        for i, (time, label) in enumerate(zip(time_points, time_labels)):
            im = evoked.plot_topomap(times=[time], ch_type='eeg', axes=axes[i], show=False, contours=0, colorbar=False, vlim=(-10, 10)) #vlim added by me, without it it ranges from -10 to 10
            axes[i].set_title(f"{label} ({int(time * 1000)} ms)")
        
        # Use the image from the last plot for the colorbar
        plt.colorbar(axes[-1].images[-1], cax=cax)

        # Save the figure
        filename = f"{condition}_event_{64 + event_idx}_key_timepoints.png"
        filepath = os.path.join(event_folder, filename)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

        print(f"Saved topomap plot for {condition} in {filepath}")

print("All plots saved successfully.")

'''
