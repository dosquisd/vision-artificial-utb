#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 -n NOTEBOOK_FILE -m FORMAT [-v]"
    echo "  -n    Path to the Jupyter notebook file (.ipynb)"
    echo "  -f    Export format (pdf or html)"
    echo "  -v    Verbose mode (optional)"
    exit 1
}

# Default values
VERBOSE=false

# Parse command-line options
while getopts ":n:f:v" opt; do
    case ${opt} in
        n) NOTEBOOK_FILE=$OPTARG ;;
        f) FORMAT=$OPTARG ;;
        v) VERBOSE=true ;;
        *) usage ;;
    esac
done

# Validate input arguments
if [[ -z "$NOTEBOOK_FILE" || -z "$FORMAT" ]]; then
    usage
fi

if [[ ! -f "$NOTEBOOK_FILE" ]]; then
    echo "Error: Notebook file '$NOTEBOOK_FILE' does not exist."
    exit 1
fi

if [[ "$FORMAT" != "pdf" && "$FORMAT" != "html" ]]; then
    echo "Error: Format must be 'pdf' or 'html'."
    exit 1
fi

# Get filename without extension
FILENAME=$(basename -- "$NOTEBOOK_FILE")
EXTENSION="${FILENAME##*.}"
NAME="${FILENAME%.*}"

if [[ "$EXTENSION" != "ipynb" ]]; then
    echo "Error: Only .ipynb files are supported."
    exit 1
fi

OUTPUT_FILE="${NAME}.${FORMAT}"

if $VERBOSE; then
    echo "Starting export..."
fi

# Capture start time
START_TIME=$SECONDS

# Run nbconvert command
jupyter nbconvert --to $FORMAT $NOTEBOOK_FILE

# Capture end time
ELAPSED_TIME=$(( SECONDS - START_TIME ))

if $VERBOSE; then
    echo "Export complete: $OUTPUT_FILE"
    echo "Total execution time: ${ELAPSED_TIME} seconds"
fi
