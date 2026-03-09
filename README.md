# MSIN0097 Predictive Analytics 25-26 — Individual Coursework

## Project Overview

This project develops an end-to-end predictive analytics solution for estimating animal adoption outcomes at the Austin Animal Center. The task is framed as a binary classification problem (Adopted vs Not Adopted) using structured shelter intake and outcome records.

The final model is a tuned XGBoost classifier achieving a test F1-score of 0.703 and ROC-AUC of 0.797.

## Repository Structure

```
Predictive Analytics/
├── MSIN0097 Individual Assignment.ipynb   # Main notebook (Sections 1-6)
├── requirements.txt                       # Python package dependencies
├── README.md                              # This file
```

## Dataset

- **Source:** Austin Animal Center Intakes and Outcomes (public dataset)
- **Records:** ~131,000 intake-outcome pairs
- **Target variable:** Binary adoption outcome (Adopted = 1, Not Adopted = 0)

### How to obtain the data

1. Download from [Kaggle](https://www.kaggle.com/datasets/aaronschlegel/austin-animal-center-shelter-intakes-and-outcomes).
3. Place it in the **same directory** as the notebook.

The notebook loads the data using a relative path:
```python
```
The notebook will fail immediately if the file is missing or misnamed.

## Setup and Reproducibility

### Prerequisites

- **Python 3.13.5** (other 3.13.x versions should work; earlier versions are untested and may not be compatible)

### Installation

1. Clone or download this repository.

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   # or
   venv\Scripts\activate           # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook "MSIN0097 Individual Assignment.ipynb"
   ```

5. Run all cells sequentially from top to bottom.

### Notes on Reproducibility

- A fixed random seed (`RANDOM_STATE = 42`) is used throughout for reproducible results.
- `GroupShuffleSplit` is used for the train-test split to prevent data leakage from repeated animals.
- `GroupKFold` (k=5) is used for all cross-validation to maintain group integrity.
- High-cardinality category groupings are fitted on training data only and applied to both splits.
- The preprocessing pipeline is cloned within each cross-validation fold to prevent information leakage.

## Notebook Structure

| Section | Title | Description |
|:--------|:------|:------------|
| 1 | Obtain a Dataset and Frame the Predictive Problem | Dataset loading, problem definition, target variable creation |
| 2 | Explore the Data to Gain Insights | EDA — distributions, correlations, adoption patterns |
| 3 | Prepare the Data | Feature engineering, train-test split, preprocessing pipeline |
| 4 | Explore Different Models and Shortlist the Best Ones | 5 models evaluated via GroupKFold CV, shortlisting |
| 5 | Fine-Tune and Evaluate | Hyperparameter tuning, final test evaluation, error analysis |
| 6 | Present the Final Solution | Final model summary, key findings, limitations, future work |

## Models Explored

| Model | CV F1 (Default) | CV F1 (Tuned) |
|:------|:-----------------|:--------------|
| Logistic Regression | 0.615 | — |
| Random Forest | 0.666 | 0.687 |
| XGBoost | 0.695 | **0.698** |
| LightGBM | 0.695 | 0.698 |
| MLP Neural Network | 0.657 | — |

## Key Dependencies

| Package | Version |
|:--------|:--------|
| pandas | 2.2.3 |
| numpy | 2.1.3 |
| scikit-learn | 1.6.1 |
| xgboost | 3.2.0 |
| lightgbm | 4.6.0 |
| matplotlib | 3.10.0 |
| seaborn | 0.13.2 |
| scipy | 1.15.3 |
