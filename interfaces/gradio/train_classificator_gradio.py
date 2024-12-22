
"""
Gradio Interface for Training Classification Models

This script provides a Gradio-based user interface to train classification models using various datasets
and algorithms. It allows users to select models, preprocess data, specify hyperparameters, and visualize
results through an intuitive web interface.

Features:
- **Model Selection**: Choose from classification algorithms in `models/supervised/classification`.
- **Dataset Input Options**:
  - Upload a local CSV file.
  - Specify a path to a dataset.
  - Download datasets from Kaggle by uploading `kaggle.json` and specifying a competition name.
- **Hyperparameter Customization**: Modify parameters such as test size, random state, CV folds, and scoring metric.
- **Visualizations**: If enabled, generate classification metrics charts and confusion matrices after training.
- **Interactive Training**: Outputs training metrics, best hyperparameters, and paths to saved models.

Usage:
- Place this script in `interfaces/gradio/`.
- Ensure proper project structure and availability of `train_classification_model.py` and classification model modules.
- Run the script. A Gradio interface will launch for interactive model training.

Requirements:
- Python 3.7 or higher
- Required Python libraries as specified in `requirements.txt`
- Properly structured project with `train_classification_model.py` and classification modules.
"""

import gradio as gr
import pandas as pd
import os
import subprocess
import sys
import glob
import re

# Add the project root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)

def get_classification_model_modules():
    # Get the list of available classification model modules
    models_dir = os.path.join(project_root, 'models', 'supervised', 'classification')
    model_files = glob.glob(os.path.join(models_dir, '*.py'))

    print(f"Looking for model files in: {models_dir}")
    print(f"Found model files: {model_files}")

    models = [os.path.splitext(os.path.basename(f))[0] for f in model_files if not f.endswith('__init__.py')]
    model_modules = [f"{model}" for model in models]
    return model_modules

def download_kaggle_data(json_path, competition_name):
    # Import the get_kaggle_data function
    from data.datasets.kaggle_data import get_kaggle_data
    data_path = get_kaggle_data(json_path=json_path, data_name=competition_name, is_competition=True)
    return data_path

def train_model(model_module, data_option, data_file, data_path, data_name_kaggle, kaggle_json_file, competition_name,
                target_variable, drop_columns, test_size, random_state, cv_folds,
                scoring_metric, model_save_path, results_save_path, visualize):

    # Determine data_path
    if data_option == 'Upload Data File':
        if data_file is None:
            return "Please upload a data file.", None
        data_path = data_file  # data_file is the path to the uploaded file
    elif data_option == 'Provide Data Path':
        if not os.path.exists(data_path):
            return "Provided data path does not exist.", None
    elif data_option == 'Download from Kaggle':
        if kaggle_json_file is None:
            return "Please upload your kaggle.json file.", None
        else:
            # Save the kaggle.json file to ~/.kaggle/kaggle.json
            import shutil
            kaggle_config_dir = os.path.expanduser('~/.kaggle')
            os.makedirs(kaggle_config_dir, exist_ok=True)
            kaggle_json_path = os.path.join(kaggle_config_dir, 'kaggle.json')
            shutil.copy(kaggle_json_file.name, kaggle_json_path)
            os.chmod(kaggle_json_path, 0o600)
        data_dir = download_kaggle_data(json_path=kaggle_json_path, competition_name=competition_name)
        if data_dir is None:
            return "Failed to download data from Kaggle.", None
        # Use the specified data_name_kaggle
        data_path = os.path.join(data_dir, data_name_kaggle)
        if not os.path.exists(data_path):
            return f"{data_name_kaggle} not found in the downloaded Kaggle data.", None
    else:
        return "Invalid data option selected.", None

    # Prepare command-line arguments for train_classification_model.py
    cmd = [sys.executable, os.path.join(project_root, 'scripts', 'train_classification_model.py')]
    cmd.extend(['--model_module', model_module])
    cmd.extend(['--data_path', data_path])
    cmd.extend(['--target_variable', target_variable])

    if drop_columns:
        cmd.extend(['--drop_columns', ','.join(drop_columns)])
    if test_size != 0.2:
        cmd.extend(['--test_size', str(test_size)])
    if random_state != 42:
        cmd.extend(['--random_state', str(int(random_state))])
    if cv_folds != 5:
        cmd.extend(['--cv_folds', str(int(cv_folds))])
    if scoring_metric:
        cmd.extend(['--scoring_metric', scoring_metric])
    if model_save_path:
        cmd.extend(['--model_path', model_save_path])
    if results_save_path:
        cmd.extend(['--results_path', results_save_path])
    if visualize:
        cmd.append('--visualize')

    print(f"Executing command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout
        errors = result.stderr
        if result.returncode != 0:
            return f"Error during training:\n{errors}", None
        else:
            # Clean up output
            output = re.sub(r"Figure\(\d+x\d+\)", "", output).strip()

            # Attempt to find confusion_matrix.png if visualize is True
            plot_image_path = None
            if results_save_path:
                # Showing the confusion matrix
                plot_image_path = os.path.join(results_save_path, 'confusion_matrix.png')
            else:
                # Default path if results_save_path is not provided
                plot_image_path = output.split('Confusion matrix saved to ')[1].strip()
            return f"Training completed successfully.\n\n{output}", plot_image_path
    except Exception as e:
        return f"An error occurred:\n{str(e)}", None

def get_columns_from_data(data_option, data_file, data_path, data_name_kaggle, kaggle_json_file, competition_name):
    # Determine data_path
    if data_option == 'Upload Data File':
        if data_file is None:
            return []
        data_path = data_file
    elif data_option == 'Provide Data Path':
        if not os.path.exists(data_path):
            return []
    elif data_option == 'Download from Kaggle':
        if kaggle_json_file is None:
            return []
        else:
            import shutil
            kaggle_config_dir = os.path.expanduser('~/.kaggle')
            os.makedirs(kaggle_config_dir, exist_ok=True)
            kaggle_json_path = os.path.join(kaggle_config_dir, 'kaggle.json')
            shutil.copy(kaggle_json_file.name, kaggle_json_path)
            os.chmod(kaggle_json_path, 0o600)
        data_dir = download_kaggle_data(json_path=kaggle_json_path, competition_name=competition_name)
        if data_dir is None:
            return []
        data_path = os.path.join(data_dir, data_name_kaggle)
        if not os.path.exists(data_path):
            return []
    else:
        return []

    try:
        data = pd.read_csv(data_path)
        columns = data.columns.tolist()
        return columns
    except Exception as e:
        print(f"Error reading data file: {e}")
        return []

def update_columns(data_option, data_file, data_path, data_name_kaggle, kaggle_json_file, competition_name):
    columns = get_columns_from_data(data_option, data_file, data_path, data_name_kaggle, kaggle_json_file, competition_name)
    if not columns:
        return gr.update(choices=[]), gr.update(choices=[])
    else:
        return gr.update(choices=columns), gr.update(choices=columns)

model_modules = get_classification_model_modules()

if not model_modules:
    print("No classification model modules found. Check 'models/supervised/classification' directory.")

with gr.Blocks() as demo:
    gr.Markdown("# Train a Classification Model")

    with gr.Row():
        model_module_input = gr.Dropdown(choices=model_modules, label="Select Classification Model Module")
        scoring_metric_input = gr.Textbox(value='accuracy', label="Scoring Metric (e.g., accuracy, f1, roc_auc)")

    with gr.Row():
        test_size_input = gr.Slider(minimum=0.1, maximum=0.5, step=0.05, value=0.2, label="Test Size")
        random_state_input = gr.Number(value=42, label="Random State")
        cv_folds_input = gr.Number(value=5, label="CV Folds", precision=0)

    visualize_input = gr.Checkbox(label="Generate Visualizations (metrics & confusion matrix)", value=True)

    with gr.Row():
        model_save_path_input = gr.Textbox(value='', label="Model Save Path (optional)")
        results_save_path_input = gr.Textbox(value='', label="Results Save Path (optional)")

    with gr.Tab("Data Input"):
        data_option_input = gr.Radio(choices=['Upload Data File', 'Provide Data Path', 'Download from Kaggle'], label="Data Input Option", value='Upload Data File')

        upload_data_col = gr.Column(visible=True)
        with upload_data_col:
            data_file_input = gr.File(label="Upload CSV Data File", type="filepath")

        data_path_col = gr.Column(visible=False)
        with data_path_col:
            data_path_input = gr.Textbox(value='', label="Data File Path")

        kaggle_data_col = gr.Column(visible=False)
        with kaggle_data_col:
            kaggle_json_file_input = gr.File(label="Upload kaggle.json File", type="filepath")
            competition_name_input = gr.Textbox(value='', label="Kaggle Competition Name")
            data_name_kaggle_input = gr.Textbox(value='train.csv', label="Data File Name (in Kaggle dataset)")

    def toggle_data_input(option):
        if option == 'Upload Data File':
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
        elif option == 'Provide Data Path':
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        elif option == 'Download from Kaggle':
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

    data_option_input.change(
        fn=toggle_data_input,
        inputs=[data_option_input],
        outputs=[upload_data_col, data_path_col, kaggle_data_col]
    )

    update_cols_btn = gr.Button("Update Columns")

    target_variable_input = gr.Dropdown(choices=[], label="Select Target Variable")
    drop_columns_input = gr.CheckboxGroup(choices=[], label="Columns to Drop")

    update_cols_btn.click(
        fn=update_columns,
        inputs=[data_option_input, data_file_input, data_path_input, data_name_kaggle_input, kaggle_json_file_input, competition_name_input],
        outputs=[target_variable_input, drop_columns_input]
    )

    train_btn = gr.Button("Train Model")
    output_display = gr.Textbox(label="Output")
    image_display = gr.Image(label="Visualization", visible=True)

    def run_training(*args):
        output_text, plot_image_path = train_model(*args)
        if plot_image_path and os.path.exists(plot_image_path):
            return output_text, plot_image_path
        else:
            return output_text, None

    train_btn.click(
        fn=run_training,
        inputs=[
            model_module_input, data_option_input, data_file_input, data_path_input,
            data_name_kaggle_input, kaggle_json_file_input, competition_name_input,
            target_variable_input, drop_columns_input, test_size_input, random_state_input, cv_folds_input,
            scoring_metric_input, model_save_path_input, results_save_path_input, visualize_input
        ],
        outputs=[output_display, image_display]
    )

if __name__ == "__main__":
    demo.launch(share=True)
