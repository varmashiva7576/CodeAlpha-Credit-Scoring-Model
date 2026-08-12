# CREDIT SCORING MODEL

## PROJECT TITLE

Credit Scoring Model Using Machine Learning

## OBJECTIVE

The objective of this project is to predict an individual's creditworthiness
using historical financial data. The model analyzes financial information
such as income, debt, loan amount, credit history, late payments, and credit
utilization to classify an applicant as having good or bad credit.

## TECHNOLOGIES USED

Programming Language:

* Python

Libraries:

* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## MACHINE LEARNING ALGORITHMS

The following classification algorithms are used in this project:

1. Logistic Regression
2. Decision Tree
3. Random Forest

## DATASET

The dataset is stored in:

credit_data.csv

The dataset contains financial information about applicants.

Main features include:

* Age
* Income
* Debt
* Loan Amount
* Credit History
* Late Payments
* Credit Utilization

Target variable:

* 1 = Good Credit
* 0 = Bad Credit

## FEATURE ENGINEERING

Additional features are created from the existing financial data.

1. Debt-to-Income Ratio

   Debt-to-Income Ratio = Debt / Income

2. Payment Risk

   Applicants with more than two late payments are considered to have
   higher payment risk.

## DATA PREPROCESSING

The following preprocessing steps are performed:

* Remove duplicate records
* Handle missing values
* Convert categorical variables into numerical values
* Scale numerical features where required
* Split the dataset into training and testing data

The dataset is divided into:

* 80% Training Data
* 20% Testing Data

## MODEL EVALUATION

The machine learning models are evaluated using the following metrics:

1. Accuracy
   Measures the percentage of correct predictions.

2. Precision
   Measures how many predicted positive cases are actually positive.

3. Recall
   Measures how many actual positive cases are correctly identified.

4. F1-Score
   Combines precision and recall into a single score.

5. ROC-AUC
   Measures how well the model distinguishes between good and bad credit.

## OUTPUT

The program generates:

* Model performance comparison
* Accuracy, Precision, Recall, F1-Score and ROC-AUC
* Classification report
* Confusion matrix
* ROC curve
* Random Forest feature-importance graph
* Creditworthiness prediction for a new applicant

## PROJECT STRUCTURE

Credit_Scoring_Project/
|
|-- credit_scoring.py
|-- credit_data.csv
|-- README.txt

## REQUIREMENTS

Python 3.x is required.

Install the required libraries using:

pip install pandas numpy scikit-learn matplotlib seaborn

## HOW TO RUN

Step 1:
Install Python 3.x on your computer.

Step 2:
Open the project folder in VS Code, PyCharm, Jupyter Notebook,
or another Python IDE.

Step 3:
Make sure the following files are in the same folder:

credit_scoring.py
credit_data.csv
README.txt

Step 4:
Install the required Python libraries:

pip install pandas numpy scikit-learn matplotlib seaborn

Step 5:
Run the Python program:

python credit_scoring.py

## RESULT

The program compares the performance of Logistic Regression, Decision Tree,
and Random Forest models.

The model with the highest ROC-AUC score is selected as the best model.

The selected model can then be used to predict the creditworthiness of
a new applicant.

## IMPORTANT NOTE

This project is developed for educational and academic purposes.

The predictions produced by the model should not be considered a real
financial or lending decision. Real-world credit-scoring systems require
appropriate validation, fairness testing, data protection, and regulatory
compliance.

## AUTHOR

Name: Varma Shiva 

Project    : Task 1 - Credit Scoring
