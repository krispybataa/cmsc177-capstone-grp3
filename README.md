# Gallstone Disease Prediction with Explainable Artificial Intelligence

## Group Members 

- Augustus Clark Raphael P. Rodriguez
- James Angelo R. Dela Cruz
- Harry William R. Acosta II 
- Jasper Anthony G. Perillo

## Project Overview

This project focuses on developing and evaluating machine learning models to predict gallstone disease using various patient health metrics. The models are enhanced with explainable AI techniques to provide interpretable insights into the prediction process, making the results more valuable for clinical applications.

## Dataset

The project uses a gallstone disease dataset with the following characteristics:
- 319 patient records
- 39 features including demographic information, body composition measurements, and blood test results
- Target variable: Gallstone Status (presence or absence of gallstones)

Key features include:
- Demographic: Age, Gender
- Comorbidities: Coronary Artery Disease, Hypothyroidism, Hyperlipidemia, Diabetes Mellitus
- Body Composition: Height, Weight, BMI, Total Body Water, Body Fat Ratio, Lean Mass, etc.
- Blood Tests: Glucose, Cholesterol, Liver enzymes, etc.

## Methodology

The project follows a comprehensive machine learning workflow:
1. **Data Preprocessing**: Handling missing values, feature scaling, and encoding categorical variables
2. **Exploratory Data Analysis**: Statistical analysis and visualization of feature distributions and relationships
3. **Feature Selection**: Identifying the most relevant features for prediction
4. **Model Development**: Training and evaluating multiple classification models
5. **Model Evaluation**: Using metrics such as accuracy, precision, recall, F1-score, and ROC AUC
6. **Explainable AI**: Implementing SHAP (SHapley Additive exPlanations) to interpret model predictions

## Models Implemented

- Random Forest Classifier
- Gradient Boosting Classifier
- AdaBoost Classifier
- Support Vector Machine (SVM)
- Logistic Regression
- Bagging Classifier
- Decision Tree Classifier

## Requirements

The project requires the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imblearn
- shap
- jupyter
- xgboost

You can install all dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository
2. Install the required dependencies
3. Open and run the Jupyter notebook `capstone-notebook.ipynb`

## Results

The project evaluates multiple machine learning models to determine which performs best for gallstone disease prediction. The evaluation includes standard performance metrics and explainable AI techniques to interpret the model's decision-making process.

## Explainable AI

The project incorporates SHAP (SHapley Additive exPlanations) values to provide transparency into model predictions. This helps in understanding which features contribute most to the prediction of gallstone disease, making the model more interpretable for clinical applications.

## Course Information

This project is a capstone for CMSC 177 at the University of the Philippines Manila.