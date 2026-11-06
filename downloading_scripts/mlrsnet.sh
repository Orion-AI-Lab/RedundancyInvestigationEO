#!/bin/bash

if [ -z "$1" ]; then
  echo "Error: No root path provided."
  exit 1
fi

root_folder_path=$1
download_url="https://data.mendeley.com/public-api/zip/7j9bv9vwsx/download/2" #${2:-""}
full_path="$root_folder_path/MLRSNet"

mkdir -p "$full_path"
archive_path="$full_path/MLRSNet_V1.zip"

# --- 1. DOWNLOAD ---
if [ ! -f "$archive_path" ]; then
    echo "Downloading MLRSNet..."
    wget -O "$archive_path" "$download_url"
    if [ $? -ne 0 ]; then
        echo "Download failed."
        rm "$archive_path"
        exit 1
    fi
fi

# --- 2. EXTRACT ---
# Only extract if we don't see the target folders yet
if [ ! -d "$full_path/Images" ]; then
    echo "Unzipping data..."
    unzip -q "$archive_path" -d "$full_path"
    
    # --- 3. NORMALIZE (Fixed Logic) ---
    # Case 1: If unzip created a wrapper folder (e.g. MLRSNet_V1/Images), move contents up.
    # We check if 'Images' is NOT in root, but IS in a subfolder.
    if [ ! -d "$full_path/Images" ]; then
        inner_dir=$(find "$full_path" -mindepth 1 -maxdepth 1 -type d | head -n 1)
        if [ -n "$inner_dir" ] && [ -d "$inner_dir/Images" ]; then
            echo "Fixing nested directory structure from: $inner_dir"
            mv "$inner_dir"/* "$full_path"/
            rmdir "$inner_dir"
        fi
    fi
fi

# --- 4. PROJECT-LOCAL UNRAR SETUP (NO SUDO REQUIRED) ---
echo "Checking for local RAR extraction tool..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_BIN_DIR="$SCRIPT_DIR/bin"
UNRAR_CMD="$LOCAL_BIN_DIR/unrar"

# If local unrar doesn't exist, download it automatically into the project
if [ ! -x "$UNRAR_CMD" ]; then
    echo "Local unrar not found. Downloading precompiled binary locally..."
    mkdir -p "$LOCAL_BIN_DIR"
    temp_dir=$(mktemp -d)
    
    cd "$temp_dir"
    wget -q https://www.rarlab.com/rar/rarlinux-x64-700.tar.gz
    if [ $? -eq 0 ]; then
        tar -xzf rarlinux-x64-700.tar.gz
        cp rar/unrar "$UNRAR_CMD"
        chmod 755 "$UNRAR_CMD"
        cd - > /dev/null
        rm -rf "$temp_dir"
        echo "Successfully installed unrar locally to: $UNRAR_CMD"
    else
        echo "Error: Failed to download unrar. Check internet connection."
        cd - > /dev/null
        rm -rf "$temp_dir"
        exit 1
    fi
else
    echo "Using local unrar found at: $UNRAR_CMD"
fi

# # --- 4. INSTALL PROPER UNRAR TOOL ---
# echo "Checking for proper RAR extraction tool..."
# UNRAR_CMD=""

# # Check if official unrar is already installed
# if command -v unrar &> /dev/null; then
#     # Test if it's the full version (not unrar-free)
#     if unrar 2>&1 | grep -q "freeware"; then
#         UNRAR_CMD="unrar"
#         echo "Found official unrar tool."
#     fi
# fi

# # If not found, install it
# if [ -z "$UNRAR_CMD" ]; then
#     echo "Installing official unrar tool..."
    
#     # Try apt install first
#     if command -v apt &> /dev/null; then
#         sudo apt update
#         sudo apt install -y unrar
#         if [ $? -eq 0 ]; then
#             UNRAR_CMD="unrar"
#         fi
#     fi
    
#     # If apt install failed or not available, download official binary
#     if [ -z "$UNRAR_CMD" ]; then
#         echo "Downloading official unrar from rarlab.com..."
#         temp_dir=$(mktemp -d)
#         cd "$temp_dir"
#         wget https://www.rarlab.com/rar/rarlinux-x64-700.tar.gz
#         if [ $? -eq 0 ]; then
#             tar -xzf rarlinux-x64-700.tar.gz
#             sudo cp rar/unrar /usr/local/bin/
#             sudo chmod 755 /usr/local/bin/unrar
#             UNRAR_CMD="/usr/local/bin/unrar"
#             cd - > /dev/null
#             rm -rf "$temp_dir"
#             echo "Official unrar installed to /usr/local/bin/unrar"
#         else
#             echo "Error: Could not download unrar. Please install manually."
#             cd - > /dev/null
#             rm -rf "$temp_dir"
#             exit 1
#         fi
#     fi
# fi

# --- 5. CLEAN UP CORRUPTED EXTRACTIONS ---
if [ -d "$full_path/Images" ]; then
    echo "Checking for corrupted image folders..."
    cd "$full_path/Images"
    
    # Remove folders that have empty (0-byte) jpg files
    for dir in */; do
        if [ -d "$dir" ]; then
            zero_count=$(find "$dir" -name "*.jpg" -size 0 2>/dev/null | wc -l)
            if [ $zero_count -gt 0 ]; then
                echo "Removing corrupted folder with $zero_count empty files: $dir"
                rm -rf "$dir"
            fi
        fi
    done
fi

# --- 6. EXTRACT RAR ARCHIVES IN IMAGES FOLDER ---

# --- 6. EXTRACT RAR ARCHIVES IN IMAGES FOLDER (QUIET MODE) ---
if [ -d "$full_path/Images" ]; then
    cd "$full_path/Images"
    
    # Check if there are any .rar files to extract
    if ls *.rar 1> /dev/null 2>&1; then
        echo "Extracting category archives from Images folder..."
        
        for f in *.rar; do
            # Extract quietly (-inul suppresses unrar's internal spam/file lists)
            $UNRAR_CMD x -inul -o+ "$f"
            
            if [ $? -eq 0 ]; then
                # Verify extraction by checking for non-empty jpg files
                category_name="${f%.rar}"
                if [ -d "$category_name" ]; then
                    zero_count=$(find "$category_name" -name "*.jpg" -size 0 2>/dev/null | wc -l)
                    total_count=$(find "$category_name" -name "*.jpg" 2>/dev/null | wc -l)
                    
                    if [ $zero_count -eq 0 ] && [ $total_count -gt 0 ]; then
                        # Successfully extracted; delete .rar silently without printing per-file OK
                        rm -f "$f"
                    else
                        echo "✗ Warning: $f extraction may be incomplete ($zero_count empty files out of $total_count)"
                    fi
                else
                    # Successfully extracted; delete .rar silently
                    rm -f "$f"
                fi
            else
                echo "✗ Failed to extract $f"
            fi
        done
    else
        echo "No .rar files found in Images folder."
    fi
    
    cd - > /dev/null
fi
# if [ -d "$full_path/Images" ]; then
#     cd "$full_path/Images"
    
#     # Check if there are any .rar files to extract
#     if ls *.rar 1> /dev/null 2>&1; then
#         echo "Extracting category archives from Images folder using $UNRAR_CMD..."
        
#         for f in *.rar; do
#             echo "Extracting $f"
#             $UNRAR_CMD x -o+ "$f"
            
#             if [ $? -eq 0 ]; then
#                 # Verify extraction by checking for non-empty jpg files
#                 category_name="${f%.rar}"
#                 if [ -d "$category_name" ]; then
#                     zero_count=$(find "$category_name" -name "*.jpg" -size 0 2>/dev/null | wc -l)
#                     total_count=$(find "$category_name" -name "*.jpg" 2>/dev/null | wc -l)
                    
#                     if [ $zero_count -eq 0 ] && [ $total_count -gt 0 ]; then
#                         echo "✓ Successfully extracted $f ($total_count files)"
#                         rm -f "$f"
#                     else
#                         echo "✗ Warning: $f extraction may be incomplete ($zero_count empty files out of $total_count)"
#                     fi
#                 else
#                     echo "✓ Extracted $f"
#                     rm -f "$f"
#                 fi
#             else
#                 echo "✗ Failed to extract $f"
#             fi
#         done
#     else
#         echo "No .rar files found in Images folder."
#     fi
    
#     cd - > /dev/null
# fi

# --- 7. STANDARDIZE LABELS FOLDER ---
# Ensure the folder is named "labels" (lowercase) to match Dataloader
if [ -d "$full_path/Labels" ] && [ ! -d "$full_path/labels" ]; then
    echo "Renaming 'Labels' to 'labels' for consistency..."
    mv "$full_path/Labels" "$full_path/labels"
fi

# --- 7b. APPLY ACTUAL LABEL NAMES TO CSVs (for visualization) ---
# If Categories_names.xlsx exists in the dataset folder, extract the 60 label names from Sheet2
# and set CSV column headers so the visualization script can load real names from labels/*.csv.
# Categories_names.xlsx is usually included in the MLRSNet Mendeley ZIP; if missing, download from
# https://github.com/cugbrs/MLRSNet or the Mendeley dataset page and place it in $full_path.
if [ -d "$full_path/labels" ] && [ -f "$full_path/Categories_names.xlsx" ]; then
    echo "Applying actual label names from Categories_names.xlsx to label CSVs..."
    export MLRSNET_FULL_PATH="$full_path"
    python3 << 'PYTHON_SCRIPT'
import os
import sys

full_path = os.environ.get("MLRSNET_FULL_PATH")
if not full_path or not os.path.isdir(os.path.join(full_path, "labels")):
    sys.exit(0)

labels_dir = os.path.join(full_path, "labels")
xlsx_path = os.path.join(full_path, "Categories_names.xlsx")
if not os.path.isfile(xlsx_path):
    sys.exit(0)

try:
    import pandas as pd
except ImportError:
    print("  (pandas not installed; skipping label name update)")
    sys.exit(0)

try:
    sheet2 = pd.read_excel(xlsx_path, sheet_name=1, header=None, engine="openpyxl")
except Exception:
    try:
        sheet2 = pd.read_excel(xlsx_path, sheet_name=1, header=None)
    except Exception as e:
        print("  Could not read Sheet2 (install openpyxl: pip install openpyxl):", e)
        sys.exit(0)

label_names = None
if sheet2.shape[1] >= 60:
    first_row = sheet2.iloc[0].astype(str).tolist()
    if len(first_row) >= 60 and not first_row[0].isdigit():
        label_names = first_row[:60]
elif sheet2.shape[0] >= 60:
    col0 = sheet2.iloc[:60, 0].astype(str).tolist()
    if not col0[0].isdigit():
        label_names = col0[:60]
if label_names is None and sheet2.shape[1] >= 61:
    header = sheet2.iloc[0].tolist()
    if len(header) >= 61:
        label_names = [str(h) for h in header[1:61]]

if label_names is None:
    print("  Could not infer 60 label names from Categories_names.xlsx; leaving CSV headers as-is.")
    sys.exit(0)

label_names = label_names[:60]
if len(label_names) < 60:
    label_names = label_names + [f"label_{i}" for i in range(len(label_names), 60)]

csv_files = [f for f in os.listdir(labels_dir) if f.endswith(".csv")]
updated = 0
for fname in csv_files:
    path = os.path.join(labels_dir, fname)
    try:
        df = pd.read_csv(path, nrows=1)
    except Exception:
        continue
    cols = df.columns.tolist()
    if len(cols) != 61:
        continue
    new_header = ["image"] + label_names
    df_full = pd.read_csv(path)
    df_full.columns = new_header
    df_full.to_csv(path, index=False)
    updated += 1

if updated > 0:
    print("  Updated headers in", updated, "CSV(s) with actual label names.")
PYTHON_SCRIPT
fi

# --- 8. FINAL VERIFICATION ---
echo ""
echo "=== Extraction Complete ==="
echo "Data located at: $full_path"
echo ""
echo "Verifying extraction:"

if [ -d "$full_path/Images" ]; then
    category_count=$(find "$full_path/Images" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "- Found $category_count category folders in Images/"
    
    # Check for any remaining issues
    remaining_rar=$(find "$full_path/Images" -name "*.rar" 2>/dev/null | wc -l)
    if [ $remaining_rar -gt 0 ]; then
        echo "⚠ Warning: $remaining_rar .rar files still remain (extraction may have failed)"
    fi
    
    zero_files=$(find "$full_path/Images" -name "*.jpg" -size 0 2>/dev/null | wc -l)
    if [ $zero_files -gt 0 ]; then
        echo "⚠ Warning: Found $zero_files empty .jpg files (possible extraction errors)"
    else
        echo "✓ No empty image files detected"
    fi
fi

if [ -d "$full_path/labels" ]; then
    label_count=$(ls "$full_path/labels"/*.csv 2>/dev/null | wc -l)
    echo "- Found $label_count CSV label files in labels/"
    if [ -f "$full_path/Categories_names.xlsx" ]; then
        echo "  (Categories_names.xlsx present; CSV headers updated with actual label names for visualization)"
    fi
fi

echo ""
echo "Done!"