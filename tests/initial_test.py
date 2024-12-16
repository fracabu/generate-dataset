import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

def prepare_data(df):
    """
    Prepara i dati per il training convertendo feature categoriche in numeriche
    """
    # Crea una copia del dataframe
    df_prep = df.copy()
    
    # Converti timestamp in caratteristiche numeriche
    df_prep['Hour'] = pd.to_datetime(df_prep['Timestamp']).dt.hour
    df_prep['Day'] = pd.to_datetime(df_prep['Timestamp']).dt.day
    df_prep['Month'] = pd.to_datetime(df_prep['Timestamp']).dt.month
    
    # Codifica le variabili categoriche
    le = LabelEncoder()
    categorical_columns = ['Request_Type', 'User_Agent', 'Location']
    
    for col in categorical_columns:
        df_prep[col + '_encoded'] = le.fit_transform(df_prep[col])
    
    # Seleziona le feature per il modello
    features = ['Hour', 'Day', 'Month', 'Status_Code', 
                'Request_Type_encoded', 'User_Agent_encoded', 
                'Location_encoded', 'Session_ID']
    
    return df_prep[features], df_prep['Anomaly_Flag']

def train_model(X, y):
    """
    Addestra un Random Forest Classifier con bilanciamento delle classi
    """
    # Split dei dati
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scala le feature
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Applica SMOTE per bilanciare le classi
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    # Addestra il modello con parametri ottimizzati
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X_train_balanced, y_train_balanced)
    
    return clf, X_test_scaled, y_test, scaler

def evaluate_model(clf, X_test, y_test):
    """
    Valuta le performance del modello
    """
    # Fai predizioni
    y_pred = clf.predict(X_test)
    
    # Stampa il report di classificazione
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))
    
    # Crea matrice di confusione
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return y_pred

def main():
    # Carica il dataset
    df = pd.read_csv('advanced_cybersecurity_data.csv')
    
    # Prepara i dati
    X, y = prepare_data(df)
    
    # Addestra il modello
    model, X_test, y_test, scaler = train_model(X, y)
    
    # Valuta il modello
    y_pred = evaluate_model(model, X_test, y_test)
    
    # Mostra le feature più importanti
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return model, scaler

if __name__ == "__main__":
    model, scaler = main()