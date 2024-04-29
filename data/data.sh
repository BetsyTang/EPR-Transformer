#!/bin/bash
# Run the Python script and redirect all output (stdout and stderr) to a log file
mode="token"
output_data="data/data_files/data_${mode}.npz"
csv_file="/home/smg/v-jtbetsy/DATA/ATEPP-s2a/ATEPP-s2a.csv"
max_len=256
performance_folder="/home/smg/v-jtbetsy/DATA/ATEPP-s2a/"
score_folder=""  # Ensure this is supposed to be empty or set appropriately
others="-s -C -A" #Options include -A alignment, -S use score, -s split, -T transcribed score, -P not padding, -C save in a compact file

# Check if the necessary directories exist or not
mkdir -p "$(dirname "$logger_file")"  # Create the log file directory if it doesn't exist
if [ ! -d "$performance_folder" ]; then
    echo "Performance folder does not exist: $performance_folder" >&2
    exit 1
fi

if [ -f "${logger_file}" ]; then
    # Rename the existing logger file
    mv "${logger_file}" "${logger_file}.history"
fi

# Run the Python script
python data/expression_dataset.py \
    -c "${csv_file}" \
    -o "${output_data}" \
    -m "${mode}" \
    -ml ${max_len} \
    -d "${performance_folder}" "${score_folder}" \
    ${others} 2>&1 | tee "${logger_file}"