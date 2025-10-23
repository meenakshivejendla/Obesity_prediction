from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
import pandas as pd
import numpy as np

MODELS = {
    'Logistic Regression': LogisticRegression(max_iter=300),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

METRICS = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# Train and Evaluate Models
def train_and_evaluate(X, y):
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        result = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='macro'),
            'Recall': recall_score(y_test, y_pred, average='macro'),
            'F1': f1_score(y_test, y_pred, average='macro'),
            'Confusion_Matrix': confusion_matrix(y_test, y_pred).tolist(),
            'CV_Accuracy': np.mean(cross_val_score(model, X, y, cv=5, scoring='accuracy'))
        }
        results[name] = result
    return results

if __name__ == '__main__':
    X = pd.read_csv('../data/X_best.csv')
    y = pd.read_csv('../data/y_res.csv').values.ravel()
    results = train_and_evaluate(X, y)
    for model_name, res in results.items():
        print(f'--- {model_name} ---')
        for k, v in res.items():
            print(f'{k}: {v}')
