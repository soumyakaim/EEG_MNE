import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from tkinter import filedialog
import tkinter as tk
from matplotlib.animation import FuncAnimation, PillowWriter

def create_info_object(channel_names, sfreq=1000):
    """
    Creates an MNE Info object with a standard montage.

    Parameters:
    - channel_names (list): List of channel names (e.g., ['Cz', 'Pz', 'Fz']).
    - sfreq (float): Sampling frequency of the data.

    Returns:
    - info (mne.Info): The Info object with the standard montage.
    """
    # Ensure channel_names is a list
    if not isinstance(channel_names, list):
        channel_names = list(channel_names)
    
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(channel_names, sfreq, ch_types='eeg')
    info.set_montage(montage)
    return info

def load_grand_average_data(folder_path):
    """
    Loads the grand average ERP data from Excel files within subfolders of the specified directory.

    Parameters:
    - folder_path (str): The path to the main directory containing condition subfolders with Excel files.

    Returns:
    - data (dict): A dictionary containing the grand average data for each event and condition.
    """
    data_files = {}
    for condition in ['keypress', 'robot_sync', 'robot_async']:
        condition_folder = os.path.join(folder_path, condition)
        if os.path.isdir(condition_folder):
            data_files[condition] = {}
            for file_name in os.listdir(condition_folder):
                if file_name.endswith('.xlsx'):
                    if 'event_64' in file_name:
                        data_files[condition]['64'] = pd.read_excel(os.path.join(condition_folder, file_name))
                    elif 'event_65' in file_name:
                        data_files[condition]['65'] = pd.read_excel(os.path.join(condition_folder, file_name))
    return data_files

def animate_topomap(data, info, output_dir, event, condition, times, interval=200):
    """
    Creates an animated topographic map over time for a given event and condition.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the ERP amplitudes for each channel (time x channels).
    - info (mne.Info): The info object containing the montage (channel locations).
    - output_dir (str): The directory to save the topographic plots.
    - event (str): The event identifier (e.g., '64', '65').
    - condition (str): The condition identifier (e.g., 'keypress', 'robot_sync', 'robot_async').
    - times (list): A list of time points at which to plot the topographic maps.
    - interval (int): The interval between frames in milliseconds.
    """
    # Create a directory for the event and condition if it doesn't exist
    event_dir = os.path.join(output_dir, f'topomap_event_{event}')
    os.makedirs(event_dir, exist_ok=True)

    # Initialize the figure
    fig, ax = plt.subplots(figsize=(6, 5))

    # Function to update the topomap for each frame
    def update_topomap(time_point):
        ax.clear()
        time_idx = np.argmin(np.abs(data['time'] - time_point))
        topo_data = data.iloc[time_idx, 1:].values  # Skip the 'time' column
        mne.viz.plot_topomap(topo_data, info, axes=ax, show=False)
        ax.set_title(f'{condition} - Event {event} - Time: {time_point:.2f} s')

    # Create the animation
    anim = FuncAnimation(fig, update_topomap, frames=times, interval=interval)

    # Save the animation as a GIF
    gif_path = os.path.join(event_dir, f'{condition}_event_{event}_topomap_animation.gif')
    anim.save(gif_path, writer=PillowWriter(fps=1000 // interval))

    plt.close()

def plot_topomap_for_events(data, output_dir, times):
    """
    Animates topographic maps over time for each event and condition.

    Parameters:
    - data (dict): A dictionary containing grand average ERP data for each event and condition.
    - output_dir (str): The directory to save the topographic plots.
    - times (list): A list of time points at which to plot the topographic maps.
    """
    # Assume all conditions have the same channels and times
    example_condition = list(data.keys())[0]
    example_event = list(data[example_condition].keys())[0]
    channel_names = data[example_condition][example_event].columns[1:]  # Skip the 'time' column

    # Convert the channel names to a list
    channel_names = list(channel_names)

    # Create the MNE Info object with the channel names
    info = create_info_object(channel_names)

    for condition in data.keys():
        for event in data[condition].keys():
            animate_topomap(data[condition][event], info, output_dir, event, condition, times)

def main():
    print("Select the folder containing the condition folders with ERP data")
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Folder Containing Condition Folders")
    root.destroy()
    
    if not folder_path:
        print("No directory selected. Exiting.")
        return
    
    data_files = load_grand_average_data(folder_path)
    
    if not data_files:
        print("No grand average data found. Exiting.")
        return
    
    # Define the time points at which to create topographic plots (e.g., every 0.1 seconds)
    times = np.arange(-0.1, 0.5, 0.05)  # Adjust these times as needed
    
    plot_topomap_for_events(data_files, folder_path, times)
    print(f'Topographic animations saved in {folder_path}')

if __name__ == "__main__":
    main()
