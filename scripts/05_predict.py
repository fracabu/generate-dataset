import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from datetime import datetime
import json
import yaml

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnomalyPredictor:
    def __init__(self, 
                 model_dir: str = "output/models",
                 pipeline_path: str = "output/feature_pipeline.joblib",
                 config_path: str = "config/predict_config.yaml"):
        """
        Inizializza il predittore di anomalie.
        
        Args:
            model_dir: Directory contenente i modelli salvati
            pipeline_path: Percorso del pipeline di feature engineering
            config_path: Percorso del file di configurazione
        """
        self.model_dir = Path(model_dir)
        self.pipeline_path = Path(pipeline_path)
        self.config = self._load_config(config_path)
        self.models = {}
        self.pipeline = None
        
    def _load_config(self, config_path: str) -> dict:
        """Carica la configurazione o usa default."""
        default_config = {
            'models_to_use': ['random_forest', 'isolation_forest', 'one_class_svm'],
            'ensemble_method': 'majority_voting',
            'output_dir': 'output/predictions',
            'threshold': 0.5
        }
        
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config non trovato in {config_path}, uso default")
            return default_config
    
    def load_models(self):
        """Carica i modelli salvati."""
        logger.info("Caricamento modelli...")
        
        # Carica pipeline di feature engineering
        self.pipeline = joblib.load(self.pipeline_path)
        logger.info("Pipeline di feature engineering caricato")
        
        # Carica modelli
        for model_name in self.config['models_to_use']:
            model_path = self.model_dir / f"{model_name}.joblib"
            if model_path.exists():
                self.models[model_name] = joblib.load(model_path)
                logger.info(f"Modello {model_name} caricato")
            else:
                logger.warning(f"Modello {model_name} non trovato in {model_path}")
        
        return self
    
    def engineer_features(self, data):
        """Applica feature engineering ai nuovi dati."""
        logger.info("Applicazione feature engineering...")
        df = data.copy()
        
        # Converti Timestamp in datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Feature temporali
        df['Hour'] = df['Timestamp'].dt.hour
        df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        df['TimeOfDay'] = pd.cut(df['Hour'], 
                               bins=[0, 6, 12, 18, 24], 
                               labels=['Night', 'Morning', 'Afternoon', 'Evening'])
        
        # Ordina per timestamp per calcolare correttamente i delta
        df = df.sort_values('Timestamp')
        
        # Feature basate su IP
        ip_stats = df.groupby('IP_Address').agg({
            'Session_ID': 'count',
            'Status_Code': lambda x: (x != 200).mean(),
            'Request_Type': 'nunique'
        }).reset_index()
        
        ip_stats.columns = ['IP_Address', 'IP_RequestCount', 'IP_ErrorRate', 'IP_UniqueRequestTypes']
        df = df.merge(ip_stats, on='IP_Address', how='left')
        
        # Feature basate su Session
        session_stats = df.groupby('Session_ID').agg({
            'IP_Address': 'nunique',
            'Request_Type': 'count'
        }).reset_index()
        
        session_stats.columns = ['Session_ID', 'Session_UniqueIPs', 'Session_RequestCount']
        df = df.merge(session_stats, on='Session_ID', how='left')
        
        # Feature di velocità
        df['TimeDelta'] = df.groupby('IP_Address')['Timestamp'].diff().dt.total_seconds()
        df['RequestSpeed'] = 1 / (df['TimeDelta'].fillna(0) + 1)
        
        # Feature categoriche
        df['IsBot'] = (df['User_Agent'] == 'Bot').astype(int)
        df['IsSuspiciousLocation'] = df['Location'].isin(['North Korea']).astype(int)
        
        # Gestione valori mancanti
        numeric_cols = ['IP_RequestCount', 'IP_ErrorRate', 'IP_UniqueRequestTypes',
                       'Session_UniqueIPs', 'Session_RequestCount', 'RequestSpeed']
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Converti colonne categoriche in string
        categorical_cols = ['Request_Type', 'Location', 'User_Agent', 'TimeOfDay']
        for col in categorical_cols:
            df[col] = df[col].astype(str)
        
        # Seleziona e ordina le colonne come nel training
        feature_cols = ['Hour', 'IsWeekend', 'IP_RequestCount', 'IP_ErrorRate', 
                       'IP_UniqueRequestTypes', 'Session_UniqueIPs', 'Session_RequestCount', 
                       'RequestSpeed', 'IsBot', 'IsSuspiciousLocation', 
                       'Request_Type', 'Location', 'User_Agent', 'TimeOfDay']
        
        return df[feature_cols]
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepara i dati per la predizione."""
        logger.info("Preparazione dati...")
        
        # Applica feature engineering
        engineered_data = self.engineer_features(data)
        
        # Applica trasformazioni
        processed_data = self.pipeline.transform(engineered_data)
        
        return processed_data
    
    def predict(self, data: pd.DataFrame) -> dict:
        """Effettua predizioni usando tutti i modelli."""
        logger.info("Effettuo predizioni...")
        
        results = {
            'predictions': {},
            'ensemble': None,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_records': int(len(data)),
                'anomalies_detected': 0
            }
        }
        
        # Predizioni singoli modelli
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                # Per classificatori che supportano probabilità
                probs = model.predict_proba(data)
                predictions = (probs[:, 1] > self.config['threshold']).astype(int)
                results['predictions'][name] = {
                    'class': [int(x) for x in predictions],
                    'probability': [float(x) for x in probs[:, 1]]
                }
            else:
                # Per modelli come IsolationForest e OneClassSVM
                predictions = pd.Series(model.predict(data)).map({1: 0, -1: 1}).astype(int)
                results['predictions'][name] = {
                    'class': [int(x) for x in predictions],
                    'probability': None
                }
        
        # Ensemble prediction
        if self.config['ensemble_method'] == 'majority_voting':
            ensemble_pred = self._majority_voting(results['predictions'])
            results['ensemble'] = [int(x) for x in ensemble_pred]
            results['metadata']['anomalies_detected'] = int(sum(ensemble_pred))
        
        return results
    
    def _majority_voting(self, predictions: dict) -> np.ndarray:
        """Applica majority voting sulle predizioni."""
        votes = []
        for model_pred in predictions.values():
            votes.append(model_pred['class'])
        return np.mean(votes, axis=0) > 0.5
    
    def save_predictions(self, results: dict, output_file: str = None):
        """Salva i risultati delle predizioni."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"predictions_{timestamp}.json"
        
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_file
        
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Predizioni salvate in {output_path}")
            logger.info(f"Totale anomalie rilevate: {results['metadata']['anomalies_detected']}")
            
        except TypeError as e:
            logger.error(f"Errore nella serializzazione JSON: {str(e)}")
            logger.info("Tentativo di convertire tutti i valori numerici in tipi Python standard...")
            
            # Funzione ricorsiva per convertire tipi numpy
            def convert_numpy_types(obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                                  np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray, list)):
                    return [convert_numpy_types(x) for x in obj]
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                return obj
            
            # Converti tutti i valori
            results = convert_numpy_types(results)
            
            # Riprova a salvare
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info("Salvataggio completato dopo la conversione dei tipi")

def main():
    """Funzione principale per l'esecuzione."""
    predictor = AnomalyPredictor()
    predictor.load_models()
    
    try:
        # Carica nuovi dati
        new_data = pd.read_csv("data/raw/new_cybersecurity_data.csv")
        logger.info(f"Caricati {len(new_data)} nuovi record da analizzare")
        
        # Prepara dati
        processed_data = predictor.prepare_data(new_data)
        
        # Effettua predizioni
        results = predictor.predict(processed_data)
        
        # Salva risultati
        predictor.save_predictions(results)
        
    except Exception as e:
        logger.error(f"Errore durante l'elaborazione: {str(e)}")
        raise

if __name__ == "__main__":
    main()