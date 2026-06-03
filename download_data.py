import os
import shutil
import tempfile
import urllib.request
import zipfile

from experiments.dataset_downloads import download_calce_cells, download_oxford

NASA_ZIP_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
NASA_CORE_CELLS = ("B0005.mat", "B0006.mat", "B0007.mat", "B0018.mat")


def setup_data_folders():
    base_dir = os.path.join(os.getcwd(), "data")
    print("\n==========================================")
    print("Creating Academic Data Directories...")
    print("==========================================")

    for folder in ("NASA", "Oxford", "CALCE", "processed"):
        path = os.path.join(base_dir, folder)
        os.makedirs(path, exist_ok=True)
        print(f"[*] Folder initialized: [data/{folder}/]")

    print("[*] Standard data layout successfully configured!")


def write_placeholder_guides():
    base_dir = os.path.join(os.getcwd(), "data")

    guides = {
        "NASA/PLACE_DATA_HERE.txt": (
            "=== NASA PCoE BATTERY DATASET GUIDE ===\n\n"
            "Official: https://data.nasa.gov/dataset/li-ion-battery-aging-datasets\n"
            "Auto-download: python download_data.py --nasa\n\n"
            "Expected files: B0005.mat, B0006.mat, B0007.mat, B0018.mat\n"
        ),
        "Oxford/PLACE_DATA_HERE.txt": (
            "=== OXFORD BATTERY DEGRADATION DATASET GUIDE ===\n\n"
            "Official: https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac\n"
            "Auto-download: python download_data.py --oxford\n\n"
            "Expected file: Oxford_Battery_Degradation_Dataset_1.mat\n"
        ),
        "CALCE/PLACE_DATA_HERE.txt": (
            "=== CALCE BATTERY DATASET GUIDE ===\n\n"
            "Official: https://calce.umd.edu/battery-data\n"
            "Auto-download: python download_data.py --calce\n\n"
            "Installs CS2_33, CS2_35, CS2_36 cell folders under data/CALCE/\n"
        ),
    }

    for rel_path, content in guides.items():
        with open(os.path.join(base_dir, rel_path), "w", encoding="utf-8") as handle:
            handle.write(content)

    print("[*] Placement guide files written to all dataset folders.")


def _find_mat_files(root_dir, filenames):
    found = {}
    for dirpath, _, files in os.walk(root_dir):
        for name in files:
            if name in filenames and name not in found:
                found[name] = os.path.join(dirpath, name)
    return found


def download_nasa_mat_files(cells=NASA_CORE_CELLS, target_dir=None):
    if target_dir is None:
        target_dir = os.path.join(os.getcwd(), "data", "NASA")
    os.makedirs(target_dir, exist_ok=True)

    existing = [f for f in cells if os.path.exists(os.path.join(target_dir, f))]
    if len(existing) == len(cells):
        print(f"[NASA] All core .mat files already present in {target_dir}")
        return target_dir

    print("[NASA] Downloading dataset zip (~200 MB) from PHM S3 mirror...")
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "nasa_battery.zip")
        urllib.request.urlretrieve(NASA_ZIP_URL, zip_path)

        with zipfile.ZipFile(zip_path, "r") as outer:
            outer.extractall(tmp)

        found = _find_mat_files(tmp, set(cells))
        if len(found) < len(cells):
            for zpath, _, files in os.walk(tmp):
                for fname in files:
                    if not fname.lower().endswith(".zip"):
                        continue
                    try:
                        with zipfile.ZipFile(os.path.join(zpath, fname), "r") as inner:
                            inner.extractall(zpath)
                    except zipfile.BadZipFile:
                        continue
            found = _find_mat_files(tmp, set(cells))

        if not found:
            raise RuntimeError("Could not locate NASA B0005.mat in downloaded archive.")

        for name, src in found.items():
            shutil.copy2(src, os.path.join(target_dir, name))
            print(f"[NASA] Installed: {name}")

    print(f"[NASA] Ready: {len(found)} file(s) in {target_dir}")
    return target_dir


def download_all_datasets():
    download_nasa_mat_files()
    download_oxford()
    download_calce_cells()
    print("\n[OK] NASA, Oxford, and CALCE datasets are ready under data/")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download NASA, Oxford, and CALCE battery datasets.")
    parser.add_argument("--nasa", action="store_true", help="Download NASA B0005-B0018 .mat files")
    parser.add_argument("--oxford", action="store_true", help="Download Oxford degradation .mat file (~266 MB)")
    parser.add_argument("--calce", action="store_true", help="Download CALCE CS2_33/35/36 cell archives")
    parser.add_argument("--all", action="store_true", help="Download all three public datasets")
    args = parser.parse_args()

    setup_data_folders()
    write_placeholder_guides()

    if args.all:
        download_all_datasets()
    else:
        if args.nasa:
            download_nasa_mat_files()
        if args.oxford:
            download_oxford()
        if args.calce:
            download_calce_cells()

    print("\n" + "=" * 50)
    print("DATA ACQUISITION")
    print("=" * 50)
    print("All datasets:   python download_data.py --all")
    print("Then run:       python run_experiments.py")
    print("Then plots:     python generate_figures.py")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
