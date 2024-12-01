import os
import pandas as pd
from pyulog.ulog2csv import convert_ulog2csv

# Directories and file settings
ulogfilepath = 'data/flightlog_raw'
output_dir = 'data/flight_csv'
processed_dir = 'data/flight_csv_processed'  # Directory to store processed CSV files
messages_type = ['actuator_outputs']

# Ensure directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

# Loop through files from 01.ulg to 15.ulg
for i in range(1, 16):
    ulogfilename = f"{str(i).zfill(2)}.ulg"
    ulog_full_path = os.path.join(ulogfilepath, ulogfilename)
    print(f"Processing {ulogfilename}...")

    try:
        # Convert .ulog file to .csv
        convert_ulog2csv(
            ulog_file_name=ulog_full_path,
            messages=','.join(messages_type),
            output=output_dir,
            delimiter=',',
            time_s=0,
            time_e=0,
            disable_str_exceptions=False
        )
    except Exception as e:
        print(f"Error converting {ulogfilename}: {e}")
        continue

    # Process extracted actuator_outputs CSV
    csv_filename = f"{ulogfilename[:-4]}_{messages_type[0]}_0.csv"
    csv_full_path = os.path.join(output_dir, csv_filename)

    if os.path.exists(csv_full_path):
        try:
            # Load data
            df = pd.read_csv(csv_full_path)
            if 'timestamp' not in df or 'output[0]' not in df:
                print(f"Required columns not found in {csv_filename}. Skipping...")
                continue

            # Keep only necessary columns
            df = df[['timestamp', 'output[0]']]

            # Remove rows after `output[0]` reaches 1000 (flight end)
            cutoff_index = df[df['output[0]'] == 1000].index.min()
            if pd.notna(cutoff_index):  # If a cutoff point exists
                df = df[:cutoff_index]  # Keep all rows until the cutoff index (exclusive)

            # Normalize timestamp to start from 0 and convert to milliseconds
            df['timestamp'] = df['timestamp'] - df['timestamp'].iloc[0]

            # Save processed data as a new CSV
            processed_filename = f"{str(i).zfill(2)}.csv"
            processed_full_path = os.path.join(processed_dir, processed_filename)
            df.to_csv(processed_full_path, index=False)
            print(f"Processed file saved as {processed_full_path}")
        except Exception as e:
            print(f"Error processing {csv_filename}: {e}")
    else:
        print(f"{csv_filename} not found in {output_dir}. Skipping...")
