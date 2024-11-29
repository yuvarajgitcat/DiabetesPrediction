# DiabetesPrediction
## Project Overview
The **Diabetes Prediction project** leverages machine learning techniques to develop a predictive model capable of identifying individuals at risk of diabetes. By utilizing health-related features, this project aims to address challenges such as delayed detection, lack of personalized prevention, and suboptimal resource allocation in diabetes care.

The ultimate goal is to provide a scalable and accurate tool for early diabetes risk assessment, contributing to improved public health outcomes and reduced healthcare costs.

The project was completed under the guidance of **_VOIS and Vodafone Idea Foundation’s _VOIS for Tech Program** .
<p align="center">
  <img src="https://github.com/yuvarajgitcat/DiabetesPrediction/blob/main/image.png" alt="Brand Logo" width="300" height="200"  position="center"/>
</p>

---

## Dataset
The dataset used in this project is attached in the repository as `diabetes.csv`. It contains the following features:
- **Pregnancies**: Number of times the patient has been pregnant.
- **Glucose Levels**: Plasma glucose concentration.
- **Blood Pressure**: Diastolic blood pressure (mm Hg).
- **Skin Thickness**: Triceps skinfold thickness (mm).
- **Insulin Levels**: 2-Hour serum insulin (mu U/ml).
- **BMI**: Body Mass Index (weight in kg/(height in m)^2).
- **Diabetes Pedigree Function**: A function that represents a summary of the family history of diabetes.
- **Age**: Patient's age (years).
- **Outcome**: Binary value (1 for diabetes, 0 for no diabetes).

### Instructions for Using the Dataset:
1. Download the repository files, including the dataset `diabetes.csv`.
2. Place the dataset in the same directory as the Python script or Jupyter Notebook.
3. Ensure the code correctly loads the dataset using:
   ```python
   import pandas as pd
   data = pd.read_csv('diabetes.csv')
   ```

---

## Key Components

### 1. **Data Analysis and Exploration**
- Comprehensive analysis of the dataset to understand its structure, statistical properties, and distribution of diabetes outcomes (positive/negative).
- Visualizations like **pair plots** and **correlation matrices** are used to examine relationships between features and identify potential dependencies.

### 2. **Data Standardization**
- Ensures uniformity and comparability by applying the **Standard Scaler** from the scikit-learn library.
- This is crucial for models like **Support Vector Machines (SVM)**, which are sensitive to feature scales.

### 3. **Model Training**
Two machine learning models are utilized:
- **Support Vector Machine (SVM)** with a linear kernel.
- **Random Forest Classifier** with 100 decision trees.

### 4. **Model Evaluation**
- Models are evaluated using metrics such as:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**
- Classification reports provide detailed insights into the models' capabilities in correctly predicting diabetes cases.

### 5. **Prediction on New Data**
- A real-world application is demonstrated by predicting diabetes risks for new data points.
- The project includes code snippets showcasing how the model can be used for specific feature inputs.

---

## Tools and Technologies
- **Programming Language**: Python
- **Libraries**:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

---

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the dataset `diabetes.csv` is in the project directory.
4. Run the Jupyter notebook or Python script:
   ```bash
   jupyter notebook Diabetes_Prediction.ipynb
   ```
   or
   ```bash
   python diabetes_prediction.py
   ```

---

## Internship Acknowledgment
This project was completed during my internship with **_VOIS and Vodafone Idea Foundation’s _VOIS for Tech Program** as part of their initiative to build skills in **Machine Learning** and **Artificial Intelligence**. The experience has helped me gain valuable insights into data preprocessing, model development, and real-world applications of predictive analytics.

---
