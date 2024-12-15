from tqdm import tqdm
import os
import numpy as np
from pydub import AudioSegment


noise_m4a_dir = "flightlog_audio_converter/data/audio_synced/"
noise_wav_dir = "flightlog_audio_converter/data/audio_synced_wav/"
noise_flightlog_dir = "flightlog_audio_converter/data/flight_csv_processed/"

clean_dir = "VoiceBank+DEMAND/wavs_clean/"
output_dir = "drone_dataset/wavs_noisy/"
noise_log_dir = "drone_dataset/noise_logs/"

# set numpy random seed
np.random.seed(0)

# m4a to wav
for m4a in tqdm(os.listdir(noise_m4a_dir)):
    if m4a.endswith(".m4a"):
        wav = m4a.replace(".m4a", ".wav")
        AudioSegment.from_file(noise_m4a_dir + m4a).export(noise_wav_dir + wav, format="wav")


def read_noise_log(file):
    with open(file, "r") as f:
        lines = f.readlines()
    output = []

    for line in lines[1:]:
        output.append(float(line.split(",")[1]))
    return output


def combine_wav(clean_file, noise_file, noise_log_file):
    clean = AudioSegment.from_file(clean_file)
    noise = AudioSegment.from_file(noise_file)
    noise_log = read_noise_log(noise_log_file)
    jump = (len(noise) - 1) // (len(noise_log) - 1)
    noise_log = np.array(noise_log)
    # interpolate noise log

    noise_log = np.interp(np.arange(len(noise)), np.linspace(0, len(noise), len(noise_log), dtype=int), noise_log)

    if len(clean) > len(noise):
        noise_start = 0
    else:
        noise_start = np.random.randint(0, len(noise) - len(clean))
    noise = noise[noise_start : noise_start + len(clean)]
    noise_log = noise_log[noise_start : noise_start + len(clean)]

    # increase noise volume
    noise = noise + 20

    # mix clean and noise
    combined = clean.overlay(noise)

    # downsample to 16kHz
    combined = combined.set_frame_rate(16000)

    return combined, noise_log


n_noise = len(os.listdir(noise_wav_dir))

for clean_file in tqdm(os.listdir(clean_dir)):
    if not clean_file.endswith(".wav"):
        continue

    # choose noise randomly and combine
    noise_idx = np.random.randint(0, n_noise)
    noise_file = os.listdir(noise_wav_dir)[noise_idx]
    noise_log_file = os.listdir(noise_flightlog_dir)[noise_idx]
    combined, noise_log = combine_wav(clean_dir + clean_file, noise_wav_dir + noise_file, noise_flightlog_dir + noise_log_file)
    combined.export(output_dir + clean_file, format="wav")
    np.save(noise_log_dir + clean_file.replace(".wav", ".npy"), noise_log)
