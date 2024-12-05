"""
This module provides a utility function to download Kaggle datasets or competition data.

The function automatically detects whether it is running in a Google Colab environment, a local Linux/Mac environment, or a Windows environment, and sets up the Kaggle API accordingly.

Requirements:
    - Kaggle API installed (`pip install kaggle`)
    - Kaggle API key (`kaggle.json`) with appropriate permissions.

Environment Detection:
    - Google Colab: Uses `/root/.config/kaggle/kaggle.json`.
    - Local Linux/Mac: Uses `~/.kaggle/kaggle.json`.
    - Windows: Uses `C:\\Users\\<Username>\\.kaggle\\kaggle.json`.

Functions:
    get_kaggle_data(json_path: str, data_name: str, is_competition: bool = False, output_dir: str = "data/raw") -> str
"""

import os
import zipfile
import sys
import shutil
import platform

def get_kaggle_data(json_path: str, data_name: str, is_competition: bool = False, output_dir: str = "data/raw") -> str:
    """
    Downloads a Kaggle dataset or competition data using the Kaggle API in Google Colab, local Linux/Mac, or Windows environment.

    Parameters:
        json_path (str): Path to your 'kaggle.json' file.
        data_name (str): Kaggle dataset or competition name (e.g., 'paultimothymooney/chest-xray-pneumonia' or 'house-prices-advanced-regression-techniques').
        is_competition (bool): Set to True if downloading competition data. Default is False (for datasets).
        output_dir (str): Directory to save and extract the data. Default is 'data'.

    Returns:
        str: Path to the extracted dataset folder.

    Raises:
        OSError: If 'kaggle.json' is not found or cannot be copied.
        Exception: If there is an error during download or extraction.

    Example of Usage:
        # For downloading a standard dataset
        dataset_path = get_kaggle_data("kaggle.json", "paultimothymooney/chest-xray-pneumonia")
        print(f"Dataset is available at: {dataset_path}")

        # For downloading competition data
        competition_path = get_kaggle_data("kaggle.json", "house-prices-advanced-regression-techniques", is_competition=True)
        print(f"Competition data is available at: {competition_path}")
    """
    # Detect environment (Colab, local Linux/Mac, or Windows)
    is_colab = "google.colab" in sys.modules
    is_windows = platform.system() == "Windows"

    # Step 1: Setup Kaggle API credentials
    try:
        if is_colab:
            config_dir = "/root/.config/kaggle"
            os.makedirs(config_dir, exist_ok=True)
            print("Setting up Kaggle API credentials for Colab environment.")
            shutil.copy(json_path, os.path.join(config_dir, "kaggle.json"))
            os.chmod(os.path.join(config_dir, "kaggle.json"), 0o600)
        else:
            # For both local Linux/Mac and Windows, use the home directory
            config_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
            os.makedirs(config_dir, exist_ok=True)
            print("Setting up Kaggle API credentials for local environment.")
            kaggle_json_dest = os.path.join(config_dir, "kaggle.json")
            if not os.path.exists(kaggle_json_dest):
                shutil.copy(json_path, kaggle_json_dest)
                if not is_windows:
                    os.chmod(kaggle_json_dest, 0o600)
    except Exception as e:
        raise OSError(f"Could not set up Kaggle API credentials: {e}")

    # Step 2: Create output directory
    dataset_dir = os.path.join(output_dir, data_name.split('/')[-1])
    os.makedirs(dataset_dir, exist_ok=True)
    original_dir = os.getcwd()
    os.chdir(dataset_dir)

    # Step 3: Download the dataset or competition data
    try:
        if is_competition:
            print(f"Downloading competition data: {data_name}")
            cmd = f"kaggle competitions download -c {data_name}"
        else:
            print(f"Downloading dataset: {data_name}")
            cmd = f"kaggle datasets download -d {data_name}"
        os.system(cmd)
    except Exception as e:
        print(f"Error during download: {e}")
        os.chdir(original_dir)
        return None

    # Step 4: Unzip all downloaded files
    zip_files = [f for f in os.listdir() if f.endswith(".zip")]
    if not zip_files:
        print("No zip files found. Please check the dataset or competition name.")
        os.chdir(original_dir)
        return None

    for zip_file in zip_files:
        try:
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall()
            print(f"Extracted: {zip_file}")
            os.remove(zip_file)
        except Exception as e:
            print(f"Error extracting {zip_file}: {e}")

    # Step 5: Navigate back to the original directory
    os.chdir(original_dir)

    return dataset_dir
