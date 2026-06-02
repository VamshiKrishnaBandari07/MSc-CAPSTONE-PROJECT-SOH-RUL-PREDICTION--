import os
import shutil
import tempfile
import urllib.request
import zipfile

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
            "Place .mat files in this directory (data/NASA/).\n"
        ),
        "Oxford/PLACE_DATA_HERE.txt": (
            "=== OXFORD BATTERY DEGRADATION DATASET GUIDE ===\n\n"
            "Official: https://ora.ox.ac.uk/objects/uuid:03ba4b01-7ed5-4da1-a1c9-cd3e54b6555c\n\n"
            "Extract raw cycle data into this directory (data/Oxford/).\n"
        ),
        "CALCE/PLACE_DATA_HERE.txt": (
            "=== CALCE BATTERY DATASET GUIDE ===\n\n"
            "Official: https://calce.umd.edu/battery-data\n\n"
            "Place CS2_*.xls files in this directory (data/CALCE/).\n"
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
    """
    Download NASA PCoE battery .mat files from the public PHM S3 mirror.
    Extracts only the core cells (B0005-B0018) used in this project.
    """
    if target_dir is None:
        target_dir = os.path.join(os.getcwd(), "data", "NASA")
    os.makedirs(target_dir, exist_ok=True)

    existing = [f for f in cells if os.path.exists(os.path.join(target_dir, f))]
    if len(existing) == len(cells):
        print(f"[NASA] All core .mat files already present in {target_dir}")
        return target_dir

    print(f"[NASA] Downloading dataset zip (~200 MB) from PHM S3 mirror...")
    print(f"       URL: {NASA_ZIP_URL}")

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "nasa_battery.zip")
        urllib.request.urlretrieve(NASA_ZIP_URL, zip_path)
        print("[NASA] Download complete. Extracting .mat files (this may take a minute)...")

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
            raise RuntimeError(
                "Could not locate B0005.mat in the downloaded archive. "
                "Download manually from https://data.nasa.gov/dataset/li-ion-battery-aging-datasets"
            )

        for name, src in found.items():
            dst = os.path.join(target_dir, name)
            shutil.copy2(src, dst)
            print(f"[NASA] Installed: {name}")

    print(f"[NASA] Ready: {len(found)} file(s) in {target_dir}")
    return target_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Set up battery dataset folders and optional NASA download.")
    parser.add_argument("--nasa", action="store_true", help="Download NASA B0005-B0018 .mat files automatically")
    args = parser.parse_args()

    setup_data_folders()
    write_placeholder_guides()

    if args.nasa:
        download_nasa_mat_files()

    print("\n" + "=" * 50)
    print("DATA ACQUISITION GUIDE FOR MSc CAPSTONE PROJECT")
    print("=" * 50)
    print("NASA (auto):  python download_data.py --nasa")
    print("Then run:       python run_experiments.py")
    print("Then plots:     python generate_figures.py")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
