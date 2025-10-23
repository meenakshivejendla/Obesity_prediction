# main.py
import pandas as pd
from preprocess import clean_data, encode_features, scale_features, balance_classes
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from modeling import train_and_evaluate  # Your existing modeling code

def select_k_best_features(X, y, k=3):
    """
    Select top k features using both ANOVA F-test and Mutual Information.
    Returns the selected features and transformed dataframe.
    """
    # ANOVA F-test
    selector_f = SelectKBest(score_func=f_classif, k='all')
    selector_f.fit(X, y)
    feature_scores_f = pd.DataFrame({'Feature': X.columns, 'Score': selector_f.scores_}).sort_values(by='Score', ascending=False)
    print("ANOVA F-test Scores:")
    print(feature_scores_f)

    # Mutual Information
    selector_mi = SelectKBest(score_func=mutual_info_classif, k='all')
    selector_mi.fit(X, y)
    feature_scores_mi = pd.DataFrame({'Feature': X.columns, 'Score': selector_mi.scores_}).sort_values(by='Score', ascending=False)
    print("\nMutual Information Scores:")
    print(feature_scores_mi)

    # Select top k features based on Mutual Information
    top_features = feature_scores_mi['Feature'].head(k).tolist()
    X_selected = X[top_features]

    print("\nSelected Features:", top_features)
    return X_selected, top_features

if __name__ == '__main__':
    # Load dataset
    df = pd.read_csv('obesity_data.csv')

    # Preprocessing
    df = clean_data(df)
    df = encode_features(df)

    # Features & Target
    X = df.drop(columns=['BMI'])  # Drop BMI if using it for target
    y = df['BMI'].apply(lambda bmi: 0 if bmi < 18.5 else 1 if bmi < 25 else 2 if bmi < 30 else 3)  # Categorize BMI

    # Scaling & balancing
    X_scaled, scaler = scale_features(X)
    X_res, y_res = balance_classes(X_scaled, y)

    # Convert back to DataFrame for feature selection
    X_res_df = pd.DataFrame(X_res, columns=X.columns)

    # Feature Selection
    X_best, selected_features = select_k_best_features(X_res_df, y_res, k=3)

    # Modeling & Evaluation
    results = train_and_evaluate(X_best, y_res)
    print("\nModel Performance:")
    for model_name, res in results.items():
        print(f'-- {model_name} --')
        for metric, value in res.items():
            print(f'{metric}: {value}')
