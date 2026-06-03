"""Download Oxford, CALCE, and NASA battery datasets."""

import os
import shutil
import zipfile
import urllib.request

OXFORD_URL = (
    "https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac/"
    "files/m5ac36a1e2073852e4f1f7dee647909a7"
)
OXFORD_REFERER = "https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac"
CALCE_CELLS = ("CS2_33", "CS2_35", "CS2_36")
CALCE_BASE = "https://web.calce.umd.edu/batteries/data"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BatterySOH/1.0"


def _stream_download(url, dest, headers=None, min_bytes=0):
    headers = headers or {"User-Agent": USER_AGENT}
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) >= min_bytes:
        return dest

    print(f"Downloading: {url}")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=600) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024
        with open(dest, "wb") as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100.0 * downloaded / total
                    print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.1f}%)", end="")
        print()
    return dest


def download_oxford(target_dir=None):
    if target_dir is None:
        target_dir = os.path.join(os.getcwd(), "data", "Oxford")
    dest = os.path.join(target_dir, "Oxford_Battery_Degradation_Dataset_1.mat")
    return _stream_download(
        OXFORD_URL,
        dest,
        headers={"User-Agent": USER_AGENT, "Referer": OXFORD_REFERER},
        min_bytes=200_000_000,
    )


def download_calce_cells(cells=CALCE_CELLS, target_dir=None):
    if target_dir is None:
        target_dir = os.path.join(os.getcwd(), "data", "CALCE")

    installed = []
    for cell in cells:
        cell_dir = os.path.join(target_dir, cell)
        nested = os.path.join(cell_dir, cell)
        if os.path.isdir(nested) and any(
            f.lower().endswith((".xls", ".xlsx")) for f in os.listdir(nested)
        ):
            installed.append(cell)
            continue

        zip_path = os.path.join(target_dir, f"{cell}.zip")
        url = f"{CALCE_BASE}/{cell}.zip"
        _stream_download(url, zip_path, min_bytes=1000)

        os.makedirs(cell_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(cell_dir)
        os.remove(zip_path)
        installed.append(cell)
        print(f"[CALCE] Installed {cell}")

    return installed
