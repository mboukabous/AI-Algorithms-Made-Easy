# Datasets Utilities

This folder contains utility scripts for handling datasets, including downloading data from Kaggle.

## 📄 Scripts

### `kaggle_data.py`

- **Description**: A Python script to download Kaggle datasets or competition data seamlessly, supporting Google Colab, local Linux/Mac, and Windows environments.
- **Path**: [`data/datasets/kaggle_data.py`](kaggle_data.py)
- **Key Function**: `get_kaggle_data(json_path, data_name, is_competition=False, output_dir='data')`
- **Example**:

  ```python
  from kaggle_data import get_kaggle_data

  # Download a standard Kaggle dataset
  dataset_path = get_kaggle_data("kaggle.json", "paultimothymooney/chest-xray-pneumonia")

  # Download competition data
  competition_path = get_kaggle_data("kaggle.json", "house-prices-advanced-regression-techniques", is_competition=True)