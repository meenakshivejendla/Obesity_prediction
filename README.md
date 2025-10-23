🏋️‍♀️ Obesity Prediction ML Pipeline

Welcome to your all-in-one machine learning project for predicting obesity levels based on personal and lifestyle features!

This project is pure ML based project

🎯 What This Project Does

Predicts obesity levels from simple features like Age, Weight, Gender, and Physical Activity.

Compares four powerful ML models: Logistic Regression, Random Forest, Decision Tree, and XGBoost.

Uses best ML practices:

Data cleaning & encoding

Scaling & standardization

Handling class imbalance with SMOTE

Feature selection with ANOVA F-test & Mutual Information

Cross-validation and robust evaluation metrics

📂 Project Structure
obesity_prediction/
│
├── README-obesity-ml.md       # Step-by-step instructions & project guide
├── preprocess.py              # Clean, encode, scale & balance data
├── feature_selection.py       # Analyze features & pick the best ones
├── modeling.py                # Train & evaluate ML models
├── utils.py                   # Helper functions (e.g., plot confusion matrix)
├── main.py                    # Orchestrates the full pipeline
├── requirements.txt           # Python dependencies
├── obesity_prediction.ipynb   # Interactive Jupyter notebook for exploration
└── data/
    └── obesity_data.csv       # Kaggle dataset

⚙️ Getting Started

Clone the repo

git clone <repo-url>
cd obesity_prediction


Install dependencies

pip install -r requirements.txt


Add your dataset

Download the Kaggle obesity dataset and place it in:

data/obesity_data.csv


Run the magic

Full pipeline execution:

python main.py


Step-by-step exploration:

Open obesity_prediction.ipynb in Jupyter Notebook.

🔍 How It Works

Preprocessing

Fix missing values

Encode categorical variables

Scale features

Balance the dataset with SMOTE

Feature Selection

Correlation matrix analysis

SelectKBest using ANOVA F-test & Mutual Information

Pick top features to feed the models

Modeling & Evaluation

Train multiple models

Evaluate using Accuracy, Precision, Recall, F1-score

Visualize confusion matrices

🌟 Why You’ll Love This Project

Fully plug-and-play for VS Code or Jupyter

Compare models side by side

Explore feature importance and see what really matters

Learn ML best practices in a real-world context

👩‍💻 Author

Meenakshi Vejendla

B.Tech CSE (AI & ML) – VIT-AP

LinkedIn

🚀 Ready to predict obesity like a pro? Drop the dataset in, run main.py, and let your ML models shine!
