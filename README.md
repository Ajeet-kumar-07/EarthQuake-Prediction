# Earthquake Prediction Model with Machine Learning

This project demonstrates how to build a neural network model to predict earthquake magnitude and depth using historical earthquake data and Python machine learning libraries.

## Features
- Loads and preprocesses earthquake data from a CSV file
- Visualizes earthquake locations on a world map
- Trains a neural network to predict magnitude and depth
- Evaluates and displays model performance

## Prerequisites
- **Python 3.9 or 3.10** (TensorFlow and scikit-learn are not yet compatible with Python 3.13)
- pip (Python package manager)

## Installation

1. **Clone or download this repository** and place your `database.csv` file in the project directory.

2. **(Recommended) Create a virtual environment:**
   ```sh
   # For Python 3.10 (replace with your installed version if needed)
   python3.10 -m venv ml-env
   ml-env\Scripts\activate  # On Windows
   # source ml-env/bin/activate  # On Mac/Linux
   ```

3. **Install required packages:**
   ```sh
   pip install numpy pandas matplotlib scikit-learn tensorflow
   ```

   > **Note:** If you see errors about package compatibility, ensure you are using Python 3.10 or 3.9.

## Usage

1. Ensure your `database.csv` file is in the project directory and contains the following columns:
   - `Date`, `Time`, `Latitude`, `Longitude`, `Depth`, `Magnitude`

2. Run the main script:
   ```sh
   python earthquake_prediction.py
   ```

3. The script will:
   - Preprocess the data
   - Visualize earthquake locations
   - Train a neural network
   - Print evaluation results and sample predictions

## Troubleshooting

- **TensorFlow or scikit-learn not installing?**
  - Make sure you are using Python 3.10 or 3.9. These libraries do not yet support Python 3.13.
  - You can check your Python version with:
    ```sh
    python --version
    ```
- **Missing columns in CSV?**
  - Ensure your `database.csv` has the required columns as listed above.

## License
This project is for educational purposes.
