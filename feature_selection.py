# feature_selection.py
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

def select_k_best(X, y, k=5):
    selector_f = SelectKBest(score_func=f_classif, k='all')
    selector_f.fit(X, y)
    feature_scores_f = pd.DataFrame({'Feature': X.columns, 'Score': selector_f.scores_}).sort_values(by='Score', ascending=False)
    print("ANOVA F-test Scores:")
    print(feature_scores_f)

    selector_mi = SelectKBest(score_func=mutual_info_classif, k='all')
    selector_mi.fit(X, y)
    feature_scores_mi = pd.DataFrame({'Feature': X.columns, 'Score': selector_mi.scores_}).sort_values(by='Score', ascending=False)
    print("\nMutual Information Scores:")
    print(feature_scores_mi)

    selected_features = feature_scores_mi['Feature'].head(k).tolist()
    print("\nSelected Features:", selected_features)
    
    X_selected = X[selected_features]
    return X_selected, selected_features
