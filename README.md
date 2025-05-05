# Linear Regression with Regularization

## Overview

This project implements linear regression with gradient descent from scratch, including both L2 (RIDGE) and L1 (LASSO) regularization techniques. The implementation is applied to predict house prices using the King County House Sales dataset from Kaggle.

## Learning Objectives

- Implement gradient descent for multiple linear regression from scratch
- Apply regularization techniques (RIDGE and LASSO) to prevent overfitting
- Understand feature selection through LASSO regularization
- Work with real-world data from Kaggle
- Visualize and interpret the results of different regularization methods

## Dataset

The dataset used in this project is the King County House Sales dataset from Kaggle, which contains house sale prices along with various features like number of bedrooms, bathrooms, square footage, etc.

Dataset link: [House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)

## Project Structure

```markdown
linear_regression_regularization/
├── README.md                                # Project documentation
├── regularization_starter.py                # Main implementation
├── linear-regression-regularization.ipynb   # Jupyter notebook with analysis
├── kc_house_data.csv                        # Dataset (not tracked by git)
└── .gitignore                               # Git ignore file
```

- `regularization_starter.py`: Main implementation file containing all functions for data preprocessing, model training, and visualization
- `linear_regression_regularization.ipynb`: Jupyter notebook with complete analysis and visualization
- `kc_house_data.csv`: Dataset file (not included in the repository, needs to be downloaded from Kaggle)
- `README.md`: Project documentation

## Implementation Details

The project includes the following key components:

1. **Data Preprocessing**
   - Loading data using pandas
   - Feature selection
   - Handling missing values
   - Feature normalization
   - Train-test split

2. **Basic Linear Regression with Gradient Descent**
   - Cost function (Mean Squared Error)
   - Gradient calculation
   - Parameter optimization using gradient descent

3. **RIDGE Regression (L2 Regularization)**
   - Modified cost function with L2 penalty term
   - Gradient calculation with regularization

4. **LASSO Regression (L1 Regularization)**
   - Modified cost function with L1 penalty term
   - Gradient calculation with regularization

5. **Visualization and Analysis**
   - Cost history plotting
   - Coefficient comparison
   - Predictions visualization
   - Feature selection analysis

## Usage

### Prerequisites

- Python 3.x
- NumPy
- Matplotlib
- Pandas

### Setup

1. Clone the repository:
   ```bash
      git clone https://github.com/NMsby/linear-regression-regularization.git
      cd linear-regression-regularization
   ```

2. Download the dataset from Kaggle and place it in the project directory.

3. Run the Python script:
   ```bash
    python regularization_starter.py
   ```

4. For detailed analysis, open the Jupyter notebook:
   ```bash
    jupyter notebook linear-regression-regularization.ipynb
   ```

## Results

The implementation successfully demonstrates how RIDGE and LASSO regularization techniques affect model performance and feature selection. Key findings include:

1. Both RIDGE and LASSO regularization help prevent overfitting compared to basic linear regression.
2. LASSO regression performs automatic feature selection by driving some coefficients to zero.
3. The regularization parameter (λ) significantly impacts model performance.
4. Square footage of living space, grade, and view are identified as the most important features for predicting house prices.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project was completed as part of a machine learning course assignment
- Thanks to Kaggle for providing the King County Housing dataset
