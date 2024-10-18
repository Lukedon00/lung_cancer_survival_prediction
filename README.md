# Lung Cancer Survival Prediction Project

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Installation](#installation)
- [How to Use](#how-to-use)
- [Model Details](#model-details)
- [Requirements](#requirements)
- [License](#license)
- [Conclusion](#conclusion)
  
## Overview
This project aims to predict the survival of lung cancer patients based on various features such as age, BMI, cholesterol level, cancer stage, smoking status, and treatment type. The objective is to build a machine learning model that can accurately classify patient survival outcomes.

## Dataset
The dataset used in this project is `dataset_med.csv`, which contains information about lung cancer patients, including the following features:
- `id`: Unique identifier for each patient
- `age`: Age of the patient
- `gender`: Gender of the patient (Male/Female)
- `bmi`: Body Mass Index
- `cholesterol_level`: Level of cholesterol
- `family_history`: Family history of cancer (Yes/No)
- `cancer_stage`: Stage of cancer (1, 2, 3, etc.)
- `smoking_status`: Smoking status (e.g., Non-smoker, Former smoker, Current smoker)
- `treatment_type`: Type of treatment received (e.g., Surgery, Chemotherapy)
- `survived`: Target variable indicating survival (0 = No, 1 = Yes)

### Data Source
The dataset is available in `dataset_med.csv`.

## Project Structure
```bash
Lung-Cancer-Survival-Prediction/
│
├── README.md
├── lung_cancer_survival_prediction.ipynb  # Jupyter notebook
├── decision_tree_model.pkl # Saved Decision Tree model
├── random_forest_model.pkl # Saved Random Forest model
├── scripts/
│   ├── preprocessing.py                   # Python script for data preprocessing
│   ├── decision_tree_model.py             # Python script for Decision Tree
│   └── random_forest_model.py             # Python script for Random Forest
├── dataset_med.csv                        # Dataset file
└──  requirements.txt                       # Dependencies
```


## Workflow
The project follows these main steps:
1. **Data Loading**: Load the dataset and explore its structure.
2. **Exploratory Data Analysis (EDA)**: Visualize the data to understand patterns and relationships.
3. **Data Preprocessing**: Handle missing values, encode categorical variables, and scale continuous features.
4. **Model Training**: Train Decision Tree and Random Forest models using the preprocessed data.
5. **Hyperparameter Tuning**: Optimize the Decision Tree model using GridSearchCV.
6. **Model Evaluation**: Evaluate models using accuracy, confusion matrix, and classification reports.
7. **Model Saving**: Save the trained models for future use.

## Installation
To set up the project, clone the repository and install the required packages.

Clone this repository to your local machine:
```bash
git clone https://github.com/Lukedon00/Lung_Cancer_Survival_Prediction.git
```
Navigate to the project directory:
```bash
cd lung_cancer_survival_prediction
```
Install requirement dependencies by running:
```bash
pip install -r requirements.txt
```
## How to Use
1. Preprocessing: Run `preprocessing.py` to handle data preprocessing tasks such as handling missing values, encoding categorical variables, and scaling continuous features.
   ```bash
   python scripts/preprocessing.py
   ```
2. Model Training: Train the Decision Tree and Random Forest models by running their respective scripts:
   - For Decision Tree:
     ```bash
     python scripts/decision_tree.py
     ```
   - For Random Forest:
     ```bash
     python scripts/random_forest.py
     ```
3. Jupyter Notebook: The main workflow, including data loading, EDA, preprocessing, model training, and evaluation, can be run in the provided Jupyter Notebook:
   ```bash
   jupyter notebook lung_cancer_survival_prediction.ipynb
   ```

## Model Details
### Decision Tree
- A basic Decision Tree model is trained first to establish a baseline accuracy. Hyperparameter tuning is performed using GridSearchCV to optimize the model.

### Random Forest
- A Random Forest model is trained to improve predictive performance compared to the Decision Tree. This model utilizes ensemble learning to achieve better accuracy.

## Requirements
The following packages are required for this project. Create a `requirements.txt` file with the following content:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
juypter
```

## License
This project is licensed under the MIT License.

## Conclusion 
This project demonstrates the complete workflow for predicting lung cancer survival using machine learning techniques. You can extend this project by experimenting with different models, adding more features, or improving the preprocessing steps.

For any questions or contributions, feel free to open an issue or a pull request!

### Key Sections Explained:
- **Overview**: A brief introduction to the project.
- **Dataset**: Description of the dataset and its features.
- **Project Structure**: Overview of the project folder structure to help users navigate.
- **Installation**: Instructions for setting up the project and installing dependencies.
- **How to Use**: Clear instructions for running preprocessing, training models, and executing the Jupyter Notebook.
- **Model Details**: Brief descriptions of the models used in the project.
- **Requirements**: Specifies required packages in a format suitable for a `requirements.txt` file.

Feel free to modify any sections as necessary to better fit your project!
