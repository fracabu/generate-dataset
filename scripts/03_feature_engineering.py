import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import logging
from pathlib import Path
from datetime import datetime

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CyberFeatureEngineer:
    def __init__(self, input_path: str = "data/raw/advanced_cybersecurity_data.csv", 
                 output_path: str = "output/processed_features.csv",
                 pipeline_path: str = "output/feature_pipeline.joblib"):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.pipeline_path = Path(pipeline_path)
        self.pipeline = None
        self.data = None
        
    def load_data(self):
        """Carica il dataset."""
        logger.info(f"Caricamento dati da {self.input_path}")
        self.data = pd.read_csv(self.input_path)
        self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])
        return self

    def engineer_features(self):
        """Feature engineering specifico per dati di cybersecurity."""
        logger.info("Creazione feature ingegnerizzate")
        
        # Converti colonne categoriche in string
        categorical_columns = ['Request_Type', 'Location', 'User_Agent']
        for col in categorical_columns:
            self.data[col] = self.data[col].astype(str)
        
        # Feature temporali
        self.data['Hour'] = self.data['Timestamp'].dt.hour
        self.data['DayOfWeek'] = self.data['Timestamp'].dt.dayofweek
        self.data['IsWeekend'] = self.data['DayOfWeek'].isin([5, 6]).astype(int)
        self.data['TimeOfDay'] = pd.cut(self.data['Hour'], 
                                      bins=[0, 6, 12, 18, 24], 
                                      labels=['Night', 'Morning', 'Afternoon', 'Evening'])
        self.data['TimeOfDay'] = self.data['TimeOfDay'].astype(str)
        
        # Feature basate su IP
        ip_stats = self.data.groupby('IP_Address').agg({
            'Session_ID': 'count',
            'Status_Code': lambda x: (x != 200).mean(),
            'Request_Type': 'nunique'
        }).reset_index()
        
        ip_stats.columns = ['IP_Address', 'IP_RequestCount', 'IP_ErrorRate', 'IP_UniqueRequestTypes']
        self.data = self.data.merge(ip_stats, on='IP_Address', how='left')
        
        # Feature basate su Session
        session_stats = self.data.groupby('Session_ID').agg({
            'IP_Address': 'nunique',
            'Request_Type': 'count'
        }).reset_index()
        
        session_stats.columns = ['Session_ID', 'Session_UniqueIPs', 'Session_RequestCount']
        self.data = self.data.merge(session_stats, on='Session_ID', how='left')
        
        # Feature di velocità
        self.data['TimeDelta'] = self.data.groupby('IP_Address')['Timestamp'].diff().dt.total_seconds()
        self.data['RequestSpeed'] = 1 / (self.data['TimeDelta'].fillna(0) + 1)
        
        # Feature categoriche
        self.data['IsBot'] = (self.data['User_Agent'] == 'Bot').astype(int)
        self.data['IsSuspiciousLocation'] = (self.data['Location'] == 'North Korea').astype(int)
        
        # Gestione valori mancanti per colonne numeriche
        numeric_columns = ['IP_RequestCount', 'IP_ErrorRate', 'IP_UniqueRequestTypes',
                         'Session_UniqueIPs', 'Session_RequestCount', 'RequestSpeed']
        self.data[numeric_columns] = self.data[numeric_columns].fillna(0)
        
        # Gestione valori mancanti per colonne categoriche
        categorical_columns = ['Request_Type', 'Location', 'User_Agent', 'TimeOfDay']
        for col in categorical_columns:
            self.data[col] = self.data[col].fillna('Unknown')
        
        # Pulizia
        self.data = self.data.drop(['Timestamp', 'TimeDelta'], axis=1)
        
        logger.info("Feature engineering completato")
        return self

    def create_pipeline(self):
        """Crea il pipeline di trasformazione."""
        # Definizione colonne
        numeric_features = ['Hour', 'IsWeekend', 'IP_RequestCount', 'IP_ErrorRate', 
                          'IP_UniqueRequestTypes', 'Session_UniqueIPs', 'Session_RequestCount', 
                          'RequestSpeed', 'IsBot', 'IsSuspiciousLocation']
        
        categorical_features = ['Request_Type', 'Location', 'User_Agent', 'TimeOfDay']
        
        # Pipeline per feature numeriche
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # Pipeline per feature categoriche
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
        ])

        # Combina i pipeline
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return self

    def fit_transform(self):
        """Applica le trasformazioni al dataset."""
        logger.info("Applicazione trasformazioni")
        
        # Separa target
        X = self.data.drop(['Anomaly_Flag', 'IP_Address', 'Session_ID'], axis=1)
        y = self.data['Anomaly_Flag']
        
        # Crea e applica pipeline
        self.create_pipeline()
        X_transformed = self.pipeline.fit_transform(X)
        
        # Ottieni i nomi delle feature
        numeric_features = ['Hour', 'IsWeekend', 'IP_RequestCount', 'IP_ErrorRate', 
                          'IP_UniqueRequestTypes', 'Session_UniqueIPs', 'Session_RequestCount', 
                          'RequestSpeed', 'IsBot', 'IsSuspiciousLocation']
        
        categorical_features = []
        for i, (name, _, _) in enumerate(self.pipeline.transformers_):
            if name == 'cat':
                feature_names = self.pipeline.named_transformers_['cat'].get_feature_names_out(['Request_Type', 'Location', 'User_Agent', 'TimeOfDay'])
                categorical_features.extend(feature_names)
        
        feature_names = numeric_features + list(categorical_features)
        
        # Combina feature trasformate e target
        transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
        transformed_df['Anomaly_Flag'] = y
        
        return transformed_df

    def save_results(self, transformed_df):
        """Salva il dataset trasformato e il pipeline."""
        logger.info(f"Salvataggio risultati in {self.output_path}")
        
        # Crea directory se non esiste
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Salva dataset e pipeline
        transformed_df.to_csv(self.output_path, index=False)
        joblib.dump(self.pipeline, self.pipeline_path)
        
        logger.info("Processo completato con successo")

def main():
    """Funzione principale per l'esecuzione dello script."""
    feature_engineer = CyberFeatureEngineer()
    feature_engineer.load_data()
    feature_engineer.engineer_features()
    transformed_df = feature_engineer.fit_transform()
    feature_engineer.save_results(transformed_df)

if __name__ == "__main__":
    main()