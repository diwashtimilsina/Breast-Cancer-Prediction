
# Breast Cancer Prediction using Machine Learning

This project aims to classify breast cancer cases as malignant or benign using machine learning techniques. The primary model used for this classification is Logistic Regression.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Collection and Preprocessing](#data-collection-and-preprocessing)
  - [Handling Missing Data](#handling-missing-data)
  - [Feature Selection](#feature-selection)
  - [Data Normalization](#data-normalization)
- [Model Training and Testing](#model-training-and-testing)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Conclusion](#conclusion)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)

## Project Overview

Breast cancer is one of the most common types of cancer among women. Early detection through predictive models can help in timely treatment and increase survival rates. This project uses machine learning to predict whether a given case of breast cancer is malignant or benign based on various features extracted from cell nuclei present in the breast mass.

## Data Collection and Preprocessing

### Handling Missing Data

The dataset used for this project is the Breast Cancer Wisconsin dataset, which was preprocessed to handle missing data. The target variable `diagnosis` was mapped to binary values where `M` (Malignant) was mapped to `1` and `B` (Benign) was mapped to `0`.

```python
breast_cancer_dataset['diagnosis'] = breast_cancer_dataset['diagnosis'].map({'M': 1, 'B': 0})
```

### Feature Selection

The dataset initially included 32 features. The `id` column was dropped as it does not contribute to the prediction. The target variable `diagnosis` was separated from the feature set.

```python
y = breast_cancer_dataset['diagnosis']
x = breast_cancer_dataset.drop(['diagnosis', 'id'], axis=1)
```

### Data Normalization

The features were normalized to bring all values within the range of 0 to 1, which is essential for the efficient training of the model.

```python
df = np.array([x])
min_df = np.min(df)
max_df = np.max(df)
normalized_df = (df - min_df) / (max_df - min_df)
x_new = normalized_df
y_new = y
```

## Model Training and Testing

The dataset was split into training and testing sets with an 80-20 split. Logistic Regression was used to train the model.

```python
x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, train_size=0.8, random_state=100)
model = LogisticRegression()
model.fit(x_train, y_train)
```

## Evaluation Metrics

The model's performance was evaluated using the following metrics:

- **Accuracy Score**
- **Mean Squared Error (MSE)**
- **R-squared (R2)**
- **Confusion Matrix**
- **Classification Report**

```python
accuracy = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
```

## Results

- **Accuracy:** `...`
- **MSE:** `...`
- **R2 Score:** `...`
- **Confusion Matrix:**

```
[[... ...]
 [... ...]]
```

- **Classification Report:**

```
              precision    recall  f1-score   support
           0       ...       ...       ...       ...
           1       ...       ...       ...       ...

    accuracy                           ...      ...
   macro avg       ...       ...       ...       ...
weighted avg       ...       ...       ...       ...
```

## Conclusion

The Logistic Regression model provides a reliable prediction of whether a breast cancer case is malignant or benign based on the dataset features. The performance metrics show that the model is well-suited for this binary classification task.

## Dependencies

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

You can install the required dependencies using pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## How to Run

1. Clone the repository:

```bash
git clone <repository-url>
```

2. Navigate to the project directory:

```bash
cd BreastCancerPrediction
```

3. Run the script:

```bash
python breast_cancer_prediction.py
```
