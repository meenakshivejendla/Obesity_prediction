Obesity Prediction ML Pipeline 🏋️‍♀️

A pure machine learning project for predicting obesity levels using personal and lifestyle features. Fully modular, end-to-end, and ready for VS Code. No web/backend boilerplate—just ML pipeline best practices.

📂 Project Structure
obesity_prediction/
│
├── README-obesity-ml.md       # Stepwise instructions and project guide
├── preprocess.py              # Data cleaning, encoding, scaling, SMOTE sampling
├── feature_selection.py       # Feature selection using correlation & SelectKBest
├── modeling.py                # Training & evaluation of multiple ML models
├── utils.py                   # Utility functions (e.g., plotting confusion matrix)
├── main.py                    # Complete pipeline orchestration
├── requirements.txt           # Python dependencies
├── obesity_prediction.ipynb   # Interactive Jupyter notebook
└── data/
    └── obesity_data.csv       # Kaggle dataset

⚙️ Setup Instructions

Clone the repository:

git clone <repo-url>
cd obesity_prediction


Install dependencies:

pip install -r requirements.txt


Place the dataset:

Download the Kaggle obesity dataset and place it at:

data/obesity_data.csv


Run the pipeline:

For end-to-end execution:

python main.py


For interactive exploration:

Open obesity_prediction.ipynb in Jupyter Notebook.

🧹 Pipeline Overview

The project follows best ML practices:

Preprocessing

Handle missing values

Encode categorical variables

Standardize features

Balance classes using SMOTE

Feature Selection

Correlation analysis

ANOVA F-test & Mutual Information (SelectKBest)

Top features selected for modeling

Modeling

Logistic Regression

Random Forest

Decision Tree

XGBoost

Cross-validation & robust evaluation metrics

Evaluation

Accuracy, Precision, Recall, F1-score

Confusion matrix visualization

🎯 Purpose

Build a reliable obesity prediction model

Compare multiple machine learning algorithms

Learn preprocessing, feature selection, and class balancing

Perform robust evaluation with cross-validation

👩‍💻 Author

Meenakshi Vejendla

B.Tech CSE (AI & ML) – VIT-AP

LinkedIn
