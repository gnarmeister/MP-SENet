import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pydub import AudioSegment
import wave
import os


def load_audio_waveform(audio_file):
    # Convert m4a to wav (pydub handles it automatically)
    audio = AudioSegment.from_file(audio_file, format="m4a")
    audio.export("temp.wav", format="wav")  # Export to WAV for easier handling
    with wave.open("temp.wav", "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        audio_data = np.frombuffer(wav_file.readframes(n_frames), dtype=np.int16)
        time_array = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))
    os.remove("temp.wav")  # Clean up temporary WAV file
    return time_array, audio_data, sample_rate


def plot_trimmed_comparison(df, audio_time, audio_data):
    """
    Plot the original actuator output and shifted trimmed audio waveform.
    Audio starts from 0 while actuator output remains unchanged.
    """
    # Scale audio data for better visualization (divide amplitude by 15)
    audio_data_scaled = audio_data / 15

    # Plot
    plt.figure(figsize=(10, 6))  # Smaller plot size
    plt.plot(df["timestamp_ms"], df["output[0]"], label="Motor Output (Original)", color="blue")
    plt.plot(audio_time * 1000, audio_data_scaled, label="Audio Waveform (Shifted, Scaled)", color="orange", alpha=0.7)
    plt.title("Trimmed Data Comparison: Motor Output and Shifted Audio")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude / Motor Output (Scaled)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def sync_tool(csv_file, audio_file, output_audio_dir):
    # Load CSV data
    df = pd.read_csv(csv_file)
    df["timestamp_ms"] = df["timestamp"] / 1000  # Convert microseconds to milliseconds

    # Load audio waveform
    audio_time, audio_data, sample_rate = load_audio_waveform(audio_file)
    audio = AudioSegment.from_file(audio_file, format="m4a")

    # Main loop for syncing
    while True:
        # Plot original data for manual start time selection
        print(f"Processing {csv_file} and {audio_file}...")
        print("Zoom and pan in the plot to decide the start time.")
        plt.figure(figsize=(10, 6))
        plt.plot(df["timestamp_ms"], df["output[0]"], label="Motor Output", color="blue")
        plt.plot(audio_time * 1000, audio_data / 15, label="Audio Waveform (Scaled)", color="orange", alpha=0.7)
        plt.title("Original Flight Log and Audio")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude / Motor Output (Scaled)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Ask the user for the start time
        try:
            start_time = float(input("Enter the START time (in ms) for sync: "))
        except ValueError:
            print("Invalid input. Please enter a numeric value for the start time.")
            continue

        # Determine the duration to match the actuator output length
        end_time = start_time + (df["timestamp_ms"].iloc[-1] - df["timestamp_ms"].iloc[0])

        # Trim the audio data to match the actuator output duration
        trimmed_audio = audio[start_time:end_time]

        # Prepare audio time for shifted plot
        trimmed_audio_time = np.linspace(0, (end_time - start_time) / 1000, num=len(trimmed_audio.get_array_of_samples()))

        # Plot the trimmed data for validation
        print("Comparing the trimmed plot...")
        plot_trimmed_comparison(df, trimmed_audio_time, np.array(trimmed_audio.get_array_of_samples()))

        # Ask user whether to proceed
        proceed = input("Do you want to save the synced data? (y/n): ").strip().lower()
        if proceed == "y":
            # Save trimmed audio with proper codec
            audio_basename = os.path.basename(audio_file).replace(".m4a", "_trimmed.m4a")
            trimmed_audio_filename = os.path.join(output_audio_dir, audio_basename)
            trimmed_audio.export(trimmed_audio_filename, format="ipod", codec="aac")

            print(f"Trimmed audio file saved as: {trimmed_audio_filename}")
            break
        else:
            print("Retrying. Please enter a new start time.")


# Directories for input and output
csv_dir = "data/flight_csv_processed"  # Folder containing flight logs
audio_dir = "data/audio"  # Folder containing audio files
output_audio_dir = "data/audio_synced"  # Output folder for synced audio files

# Ensure output directories exist
os.makedirs(output_audio_dir, exist_ok=True)

# Process files one by one
for i in range(1, 16):
    csv_file = os.path.join(csv_dir, f"{str(i).zfill(2)}.csv")
    audio_file = os.path.join(audio_dir, f"{str(i).zfill(2)}.m4a")

    if os.path.exists(csv_file) and os.path.exists(audio_file):
        print(f"\nSyncing {csv_file} with {audio_file}...")
        sync_tool(csv_file, audio_file, output_audio_dir)
        print("\n--- Next File ---")
    else:
        print(f"Missing file: {csv_file} or {audio_file}. Skipping...")
