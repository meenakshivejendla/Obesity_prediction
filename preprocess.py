import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# Data Preprocessing Functions
def clean_data(df):
    # Fill missing values (example: median for numerical, mode for categorical)
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    for col in ['Age', 'Height', 'Weight', 'PhysicalActivity', 'CalorieIntake']:
        df[col] = df[col].fillna(df[col].median())
    return df

def encode_features(df):
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    return df

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def balance_classes(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

if __name__ == '__main__':
    # Example usage
    DATA_PATH = '../data/obesity_data.csv'
    df = pd.read_csv(DATA_PATH)
    df = clean_data(df)
    df = encode_features(df)
    X = df.drop('ObesityLevel', axis=1)
    y = df['ObesityLevel']
    X_scaled, scaler = scale_features(X)
    X_res, y_res = balance_classes(X_scaled, y)
    # Save processed data for next step
    pd.DataFrame(X_res, columns=X.columns).to_csv('../data/X_res.csv', index=False)
    pd.DataFrame(y_res, columns=['ObesityLevel']).to_csv('../data/y_res.csv', index=False)
