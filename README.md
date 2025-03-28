
# ResiPrice-GradientDescent Price Prediction

![GitHub repo size](https://img.shields.io/github/repo-size/ameer22l7555/ResiPrice-GradientDescent)
![GitHub last commit](https://img.shields.io/github/last-commit/ameer22l7555/ResiPrice-GradientDescent)
![License](https://img.shields.io/github/license/ameer22l7555/ResiPrice-GradientDescent)

## Overview

This project explores the ** House Sales dataset** to analyze and predict house prices using a linear regression model. The dataset, sourced from Kaggle, contains detailed information about houses sold in King County, Washington, between May 2014 and May 2015. The goal is to build a predictive model that leverages features such as square footage, number of bedrooms, bathrooms, and location to estimate house prices, providing insights into the real estate market in this region.

This repository includes a Jupyter Notebook with Python code for data downloading, preprocessing, exploratory data analysis (EDA), and model training. The project is designed to be reproducible and serves as a foundation for further enhancements, such as feature engineering or advanced machine learning techniques.

---

## Table of Contents

1. [Dataset](#dataset)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Methodology](#methodology)
6. [Results](#results)
7. [Future Work](#future-work)
8. [Contributing](#contributing)
9. [License](#license)
10. [Acknowledgments](#acknowledgments)

---

## Dataset

The dataset used in this project is the **King County House Sales dataset** (`kc_house_data.csv`), available on Kaggle under the identifier `harlfoxem/housesalesprediction`. It contains 21,613 records and 21 features, including:

- **id**: Unique identifier for each house sale.
- **date**: Date of the house sale.
- **price**: Sale price of the house (target variable).
- **bedrooms**: Number of bedrooms.
- **bathrooms**: Number of bathrooms.
- **sqft_living**: Square footage of the living space.
- **sqft_lot**: Square footage of the lot.
- **floors**: Number of floors.
- **waterfront**: Whether the house has a waterfront view (0 or 1).
- **view**: Quality of the view (0-4).
- **condition**: Condition of the house (1-5).
- **grade**: Overall grade given to the housing unit (1-13).
- **sqft_above**: Square footage above ground.
- **sqft_basement**: Square footage of the basement.
- **yr_built**: Year the house was built.
- **yr_renovated**: Year of the last renovation (0 if never renovated).
- **zipcode**: ZIP code of the house location.
- **lat**: Latitude coordinate.
- **long**: Longitude coordinate.
- **sqft_living15**: Living space square footage of the nearest 15 neighbors.
- **sqft_lot15**: Lot square footage of the nearest 15 neighbors.

The dataset is downloaded programmatically using the `kagglehub` library in the notebook.

---

## Project Structure

```
ResiPrice-GradientDescent/
│
├── LR_KC_House_scratch_lib.ipynb    # Main Jupyter Notebook with code
├── README.md                    # Project documentation (this file)
├── requirement.txt             # Python dependencies
└── LICENSE                      # License file (e.g., MIT)
```

- **LR_KC_House_scratch_lib.ipynb**: Contains the full workflow, including data downloading, preprocessing, EDA, and linear regression modeling.
- **requirement.txt**: Lists the required Python libraries.

---

## Installation

To run this project locally, follow these steps:

### Prerequisites

- Python 3.8 or higher
- Git
- Kaggle account (for API access to download the dataset)

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ameer22l7555/ResiPrice-GradientDescent.git
   cd ResiPrice-GradientDescent
   ```

2. **Set Up a Virtual Environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirement.txt
   ```

   The `requirement.txt` file should include:
   ```
   kagglehub
   pandas
   matplotlib
   seaborn
   numpy
   scikit-learn
   jupyter
   ```

4. **Configure Kaggle API**
   - Obtain your Kaggle API key from your Kaggle account settings.
   - Place the `kaggle.json` file in `~/.kaggle/` (Linux/Mac) or `%USERPROFILE%\.kaggle\` (Windows).
   - Ensure proper permissions:
     ```bash
     chmod 600 ~/.kaggle/kaggle.json  # Linux/Mac only
     ```

5. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
   Open `LR_KC_House_scratch_lib.ipynb` in your browser.

---

## Usage

1. **Run the Notebook**
   - Open `LR_KC_House_scratch_lib.ipynb` in Jupyter Notebook.
   - Execute the cells sequentially to:
     - Download the dataset using `kagglehub`.
     - Load and inspect the data with pandas.
     - Perform EDA with matplotlib and seaborn (if extended).
     - Preprocess the data (e.g., scaling with MinMaxScaler).
     - Train a linear regression model and evaluate its performance.

2. **Modify the Code**
   - Adjust features used in the model by modifying the input variables in the preprocessing step.
   - Experiment with different algorithms by replacing `LinearRegression` with other scikit-learn models.

3. **Expected Output**
   - The notebook prints the dataset path after downloading.
   - Displays the raw dataset as a pandas DataFrame.
   - (If extended) Outputs visualizations and model performance metrics like MSE and R².

---

## Methodology

### Data Acquisition
- The dataset is downloaded using `kagglehub.dataset_download()` from Kaggle.

### Data Preprocessing
- Loaded into a pandas DataFrame.
- Potential steps (to be implemented):
  - Handle missing values (if any).
  - Drop irrelevant columns (e.g., `id`, `date`).
  - Scale numerical features using `MinMaxScaler`.

### Exploratory Data Analysis (EDA)
- (Suggested) Visualize relationships between features (e.g., `sqft_living` vs. `price`) using scatter plots and heatmaps.

### Modeling
- A linear regression model (`LinearRegression`) is trained using scikit-learn.
- Features: Likely a subset of numerical columns (e.g., `sqft_living`, `bedrooms`, `bathrooms`).
- Target: `price`.

### Evaluation
- Metrics: Mean Squared Error (MSE) and R² score from `sklearn.metrics`.

---

## Results

- **Model Performance**: The linear regression model from Library achieved an R² score of 0.492 and an MSE of 0.0011 on the test set and the linear regression model from scratch achieved an R² score of 0.4886 and an MSE of 0.0011
- **Key Insights**: Square footage (`sqft_living`) and house grade (`grade`) were the strongest predictors of price.

---

## Future Work

- **Feature Engineering**: Incorporate interaction terms or derive new features (e.g., age of house = current year - `yr_built`).
- **Advanced Models**: Experiment with Random Forest, Gradient Boosting, or neural networks.
- **Geospatial Analysis**: Use `lat` and `long` for location-based clustering or visualization.
- **Hyperparameter Tuning**: Optimize the model using GridSearchCV or RandomizedSearchCV.
- **Deployment**: Convert the model into a web app using Flask or Streamlit.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

Please ensure your code follows PEP 8 style guidelines and includes appropriate comments.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Kaggle**: For providing the King County House Sales dataset.
- **xAI**: For creating Grok, which assisted in generating this README.
- **Open Source Community**: For the amazing libraries used (pandas, scikit-learn, etc.).

---
