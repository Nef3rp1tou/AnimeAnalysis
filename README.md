
# AnimeAnalysis

AnimeAnalysis is a Python project designed for analyzing and predicting anime ratings based on data from Kaggle's anime datasets. 
This project includes data cleaning, exploratory data analysis (EDA), and machine learning (ML) models for recommendations and predictions.

## Features

- **Data Preprocessing**: Handles missing values, duplicates, and inconsistent data formatting.
- **Exploratory Data Analysis (EDA)**: Generates visualizations for user rating distributions, correlations, and seasonal trends.
- **Machine Learning**:
  - Predicts anime average ratings using a Random Forest model.
  - Content-based recommendation system using genre similarity.
- **Modular Design**: Separated functionality into specific modules for better maintainability.

## Directory Structure

```
AnimeAnalysis/
├── .venv/                 # Python virtual environment (optional for development)
├── data/                  # Contains dataset files and scripts
│   ├── anime.csv          
│   └── rating.csv  
├── anime_data_processor.py # Handles data loading and cleaning
├── anime_eda.py            # Exploratory Data Analysis functionality
├── anime_ml.py             # Machine Learning models and features
└── main.py                 # Entry point for running the project
```

## Datasets

The project uses Kaggle's [anime.csv](https://www.kaggle.com/datasets) and [rating.csv](https://www.kaggle.com/datasets). Ensure these files are placed in the `data/` directory.

## Installation

1. Clone the repository or download the source code.
2. Ensure Python 3.8 or later is installed.
3. Install the required packages using pip:

```bash
pip install -r requirements.txt
```

4. Place the `anime.csv` and `rating.csv` datasets in the `data/` directory.

## Usage

Run the `main.py` script to perform the following tasks:
- Load and clean the data.
- Generate visualizations for EDA.
- Train and evaluate machine learning models.

Example command to run the project:

```bash
python data/main.py
```

## Requirements

The project depends on the following Python libraries (also available in `requirements.txt`):
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Future Improvements

- Add collaborative filtering to the recommendation system.
- Enhance EDA visualizations with interactive tools.
- Implement deployment features for easier accessibility.

