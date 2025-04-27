#!/bin/bash

set -e

if ! command -v enscript &> /dev/null; then
    echo "enscript is not installed. Try: sudo apt install enscript OR brew install enscript"
    exit 1
fi

if ! command -v ps2pdf &> /dev/null; then
    echo "ps2pdf (Ghostscript) is not installed. Try: sudo apt install ghostscript OR brew install ghostscript"
    exit 1
fi

PS_FILE="output.ps"
PDF_FILE="output.pdf"

PY_FILES=(./neural_networks/*.py train_conv.py train_ffnn.py)

# Check if files exist
if [ ${#PY_FILES[@]} -eq 0 ]; then
    echo "No Python files found to include."
    exit 1
fi

echo "Generate PostScript with enscript..."
enscript -Epython --color -o "$PS_FILE" "${PY_FILES[@]}"

# convert to PDF
ps2pdf "$PS_FILE" "$PDF_FILE"

rm "$PS_FILE"

echo "Done! Successfully created: $PDF_FILE"
