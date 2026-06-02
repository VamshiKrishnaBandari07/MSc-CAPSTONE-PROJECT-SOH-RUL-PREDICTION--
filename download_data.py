import os
import urllib.request
import zipfile
import shutil

# Dataset URL constants
DATASETS = {
    "NASA": {
        "url": "https://raw.githubusercontent.com/VamshiKrishnaBandari07/MSc-CAPSTONE-PROJECT-SOH-RUL-PREDICATION-/main/data/nasa_placeholder.txt", # Placeholder or official mirror
        "desc": "NASA Prognostics Center of Excellence (PCoE) Battery Dataset (B0005, B0006, B0007, B0018)",
        "official_page": "https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository"
    },
    "Oxford": {
        "url": "https://ora.ox.ac.uk/objects/uuid:03ba4b01-7ed5-4da1-a1c9-cd3e54b6555c",
        "desc": "Oxford Battery Degradation Dataset (pouch cells cycled at 40C)",
        "official_page": "https://ora.ox.ac.uk/objects/uuid:03ba4b01-7ed5-4da1-a1c9-cd3e54b6555c"
    },
    "CALCE": {
        "url": "https://calce.umd.edu/battery-data",
        "desc": "Maryland Center for Advanced Life Cycle Engineering (CALCE) Battery Dataset (CS2, CX2 cells)",
        "official_page": "https://calce.umd.edu/battery-data"
    }
}

def setup_data_folders():
    """
    Creates standard structured data directories for raw files.
    """
    base_dir = os.path.join(os.getcwd(), "data")
    print(f"\n==========================================")
    print(f"Creating Academic Data Directories...")
    print(f"==========================================")
    
    subfolders = ["NASA", "Oxford", "CALCE", "processed"]
    for folder in subfolders:
        path = os.path.join(base_dir, folder)
        os.makedirs(path, exist_ok=True)
        print(f"[*] Folder initialized: [data/{folder}/]")
        
    print(f"[*] Standard data layout successfully configured!")

def write_placeholder_guides():
    """
    Creates informative guide files inside each dataset directory, outlining
    where to download and place the raw files.
    """
    base_dir = os.path.join(os.getcwd(), "data")
    
    # NASA Guide
    nasa_guide_path = os.path.join(base_dir, "NASA", "PLACE_DATA_HERE.txt")
    with open(nasa_guide_path, "w", encoding="utf-8") as f:
        f.write("=== NASA PCoE BATTERY DATASET GUIDE ===\n\n"
                "Official Download Portal: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository\n"
                "Direct Download Link: https://ti.arc.nasa.gov/c3/ti/research/prognostic-senti/battery-data-set/\n\n"
                "Expected Files:\n"
                "- B0005.mat\n"
                "- B0006.mat\n"
                "- B0007.mat\n"
                "- B0018.mat\n\n"
                "Place these .mat files directly in this directory (data/NASA/).\n")
                
    # Oxford Guide
    oxford_guide_path = os.path.join(base_dir, "Oxford", "PLACE_DATA_HERE.txt")
    with open(oxford_guide_path, "w", encoding="utf-8") as f:
        f.write("=== OXFORD BATTERY DEGRADATION DATASET GUIDE ===\n\n"
                "Official Download Portal: https://ora.ox.ac.uk/objects/uuid:03ba4b01-7ed5-4da1-a1c9-cd3e54b6555c\n\n"
                "Expected Files:\n"
                "- Cell1.mat, Cell2.mat, Cell3.mat, etc.\n"
                "Extract all raw cycle mat/csv data directly into this directory (data/Oxford/).\n")
                
    # CALCE Guide
    calce_guide_path = os.path.join(base_dir, "CALCE", "PLACE_DATA_HERE.txt")
    with open(calce_guide_path, "w", encoding="utf-8") as f:
        f.write("=== CALCE BATTERY DATASET GUIDE ===\n\n"
                "Official Download Portal: https://calce.umd.edu/battery-data\n\n"
                "Expected Files:\n"
                "- CS2_35.xls / CS2_36.xls / CS2_37.xls / CS2_38.xls\n"
                "Place the cell testing sheets directly in this directory (data/CALCE/).\n")
                
    print("[*] Placement guide files written to all dataset folders.")

def main():
    setup_data_folders()
    write_placeholder_guides()
    
    print("\n" + "="*50)
    print("DATA ACQUISITION GUIDE FOR MSc CAPSTONE PROJECT")
    print("="*50)
    print("Because public battery aging datasets are multi-gigabyte files protected by university portals:")
    print("1. Please download the raw files from the official links listed inside `data/<DatasetName>/PLACE_DATA_HERE.txt`.")
    print("2. Put the files in their respective folders.")
    print("3. Run the train script (`python train.py`), which will automatically detect and parse the raw files, or execute high-fidelity simulation fallbacks if you want to run instant demonstration runs without large downloads.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
