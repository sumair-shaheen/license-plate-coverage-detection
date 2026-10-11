# License Plate Coverage Detection

## Project Overview

This project detects potentially covered or obstructed vehicle license plates using image processing and computer vision techniques.

The system analyzes license plate candidate regions and uses image features such as edge density, contrast, and threshold-based pixel patterns to estimate whether a plate is covered.

## Features

- License plate candidate detection using OpenCV.
- Image preprocessing and edge detection.
- Coverage scoring based on image characteristics.
- Binary classification of images as normal or covered.
- Generation of prediction results and performance visualizations.
- Confusion matrix and accuracy, precision, recall, and F1-score evaluation.

## Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook / Google Colab

## Project Structure

```text
license-plate-coverage-detection/
├── main.py
├── Project_Sumaira_Shaheen_402722144.ipynb
├── output.txt
├── covered.zip
├── IRCP_dataset_640X480.zip
├── test_images.zip
├── results.zip
├── README.md
├── requirements.txt
└── .gitignore
```

The dataset and results archives are included only if they are committed to the repository.

## Installation

1. Install Python 3.10 or a compatible Python version.
2. Clone this repository:

   ```bash
   git clone https://github.com/sumair-shaheen/license-plate-coverage-detection.git
   cd license-plate-coverage-detection
   ```

3. Install the required packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

## Running the Project

Extract your test images into a folder, such as `test_images`, then run:

```bash
python main.py --input test_images --output output.txt
```

The script generates predictions in `output.txt` and saves sample images and evaluation plots in a `results` folder.

## Prediction Labels

- `0` — Normal / not detected as covered.
- `1` — Detected as covered.

## Important Notes

This project uses image-processing heuristics rather than a trained deep-learning model. Predictions depend on the selected thresholds and the quality of the input images. The results should be evaluated against correctly labeled ground-truth data.

## Author

Sumaira Shaheen