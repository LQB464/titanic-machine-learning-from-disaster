# Titanic - Machine Learning from Disaster

A Kaggle classification project that predicts which passengers survived the sinking of the Titanic using passenger information such as age, sex, ticket class, fare, and engineered features.

Competition link: https://www.kaggle.com/competitions/titanic

> Lưu ý cho người xem tiếng Việt: Đây là project Kaggle Titanic được triển khai bằng Jupyter Notebook, tập trung vào EDA, feature engineering và so sánh nhiều mô hình machine learning.

---

## Team

Members list:

- Nguyễn Thị Huyền Trang - 23280092 (Leader)  
- Nguyễn Quang Lập - 23280007  
- Trương Minh Tiền - 23280088  
- Lương Quốc Bình - 23280042  
- Võ Hoàng - 23280060  

---

## Repository structure

```text
titanic-machine-learning-from-disaster/
├── dataset/      # Titanic CSV data (train.csv, test.csv, etc. from Kaggle)
├── notebook/     # Main Jupyter Notebook for analysis and modeling
└── document/     # Reports or supporting documents for the project
````

Main notebook:

* `notebook/titanic-machine-learning-from-disaster.ipynb`

This notebook contains the full workflow from data validation, EDA, feature engineering, preprocessing, model training to final evaluation.

---

## Problem description

The goal of the Kaggle competition is to:

> Predict whether a passenger survived the Titanic disaster based on passenger data (name, age, gender, socio-economic class, etc.).

We are given:

* `train.csv`: includes passenger features and the target `Survived`
* `test.csv`: same structure but without `Survived` (the target to predict)

The notebook uses the train set to:

1. Explore and clean the data
2. Engineer useful features
3. Train and evaluate several models
4. Select the best performing approach
5. Generate predictions for Kaggle submission (if desired)

---

## Approach

The project is organized in several main steps, reflected in the notebook sections.

### 1. Data loading and validation

* Load `train.csv` into a Pandas DataFrame.
* Perform data validation and checks:

  * Missing values
  * Duplicated rows
  * Data types
  * Number of unique values per column
  * Identify numerical vs categorical features
  * Check target imbalance for `Survived`

This step ensures that the dataset is clean enough and guides how to handle missing values and outliers.

### 2. Exploratory Data Analysis (EDA)

The notebook performs EDA on both the original and feature engineered datasets, including:

* Distribution of numerical features (Age, Fare, etc.)
* Distribution of categorical features (Sex, Pclass, Embarked, etc.)
* Correlation analysis between features and with the target
* Visualizations to understand survival patterns across different groups

Some key findings are summarized in the notebook, for example:

* Higher ticket class (Pclass) and higher fares are associated with higher survival rates
* Sex is a strong predictor of survival
* Family related features (siblings, parents, children) add useful information

### 3. Feature engineering

Several new features are created to improve model performance, including for example:

* **Title** extracted from the passenger name
* **FamilySize** combining `SibSp` and `Parch`
* **FamilySize_Bucketized** grouping passengers into categories such as Alone, Medium, Large

These features help the models capture more meaningful patterns from the data, such as social status and group size.

### 4. Data preprocessing

A preprocessing pipeline is built using `scikit-learn`, including:

* Handling missing values with `SimpleImputer`
* Scaling numerical features using `StandardScaler`
* Encoding categorical variables using `OneHotEncoder` or `OrdinalEncoder`
* Using `ColumnTransformer` and `Pipeline` to combine preprocessing steps and models in a clean and reproducible way

This pipeline is later reused across multiple models, which makes experimentation easier and less error prone.

### 5. Model selection and experiments

The notebook explores multiple approaches.

#### Approach 1: Baseline models and PCA

* Train several baseline models without heavy tuning, such as:

  * Logistic Regression
  * Random Forest
  * XGBoost
  * KNN
  * SVM
* Apply PCA to reduce dimensionality when there are many features
* Compare:

  * Models on the original feature space
  * Models on PCA transformed features

Observations:

* PCA can reduce the number of features while retaining most of the variance.
* However, model performance slightly decreases after PCA in this project.
* Therefore, the original feature space is kept for later steps.

#### Approach 2: Feature importance and feature selection

To simplify the model and focus on the most informative features, the notebook:

1. Trains models such as Random Forest and XGBoost.
2. Extracts `feature_importances_` from each model.
3. Ranks features and combines scores from different models.
4. Selects the strongest features for further experiments.

Scenarios evaluated:

* **Scenario 1**: Use all available features
* **Scenario 2**: Remove redundant categorical features and keep only the top features

The notebook then compares performance between using all features and using only the strongest ones.

#### Model optimization using Voting Classifier

The project also experiments with an ensemble Voting Classifier combining:

* KNN
* Random Forest
* XGBoost

This configuration is tested to see whether combining several strong models improves performance. In this specific case, the Voting Classifier does not outperform the best standalone model.

---

## Final model and results

Based on the experiments, the notebook concludes that:

* The most effective configuration is to keep the 5 strongest features and use a **KNN classifier with tuned hyperparameters**.
* This KNN model achieves approximately:

  * **Accuracy**: 85.3%
  * **F1 score**: 0.792

This solution is preferred because it offers:

* High performance on the validation set
* Low complexity, easy to implement and deploy
* Reasonable interpretability

---

## Technologies and libraries

The project mainly uses:

* Python
* Jupyter Notebook
* NumPy, Pandas
* Matplotlib, Seaborn, Plotly
* Scikit-learn
* XGBoost
* SciPy
* Statsmodels

---

## How to run the notebook

1. **Clone the repository**

   ```bash
   git clone https://github.com/LQB464/titanic-machine-learning-from-disaster.git
   cd titanic-machine-learning-from-disaster
   ```

2. **Set up a Python environment**

   Create and activate a virtual environment if you wish:

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # on Linux or macOS
   .venv\Scripts\activate         # on Windows
   ```

3. **Install required packages**

   Install the core libraries used in the notebook, for example:

   ```bash
   pip install numpy pandas matplotlib seaborn plotly scikit-learn xgboost statsmodels scipy
   ```

4. **Download Kaggle data**

   * Go to the competition page: [https://www.kaggle.com/competitions/titanic](https://www.kaggle.com/competitions/titanic)
   * Download `train.csv` (and `test.csv` if needed)
   * Place them inside the `dataset/` folder or adjust the paths in the notebook to match your setup

5. **Open the notebook**

   ```bash
   jupyter notebook
   ```

   Then open:

   ```text
   notebook/titanic-machine-learning-from-disaster.ipynb
   ```

6. **Run all cells**

   Run the notebook from top to bottom to reproduce the full analysis and model training. You can also modify parts of the code to test additional models or feature engineering ideas.

---

## Future work

Some possible extensions for this project:

* Try more advanced feature engineering (for example target encoding, interaction features)
* Explore other model families (LightGBM, CatBoost, neural networks)
* Perform more systematic hyperparameter optimization
* Use cross validation more extensively instead of a single train test split
* Track and compare Kaggle leaderboard scores for different model versions

---

## Acknowledgements

* Kaggle Titanic Machine Learning from Disaster competition and dataset
* Open source libraries in the Python data science ecosystem

```
