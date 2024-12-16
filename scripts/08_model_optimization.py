import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score
from sklearn.base import clone
from itertools import product
import yaml
from datetime import datetime
import time

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    def __init__(self, 
                 data_path: str = "output/processed_features.csv",
                 output_dir: str = "output/optimization",
                 max_time: int = 300):  # 5 minuti max per modello
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_time = max_time
        
        self.data = None
        self.X = None
        self.y = None
        self.models = {}
        self.results = {}

    def load_data(self):
        """Carica e prepara i dati."""
        logger.info("Caricamento dati...")
        self.data = pd.read_csv(self.data_path)
        
        self.X = self.data.drop('Anomaly_Flag', axis=1)
        self.y = self.data['Anomaly_Flag']
        
        logger.info(f"Dataset caricato: {len(self.data)} campioni")

    def optimize_random_forest(self):
        """Ottimizza Random Forest con griglia ridotta."""
        logger.info("Ottimizzazione Random Forest...")
        start_time = time.time()
        
        # Griglia ridotta
        param_grid = {
            'n_estimators': [100],
            'max_depth': [10, None],
            'min_samples_split': [2, 5],
        }
        
        best_score = float('-inf')
        best_params = None
        best_model = None
        
        for params in [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]:
            if time.time() - start_time > self.max_time:
                logger.warning("Tempo massimo raggiunto per Random Forest")
                break
                
            logger.info(f"Testing parameters: {params}")
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            model.fit(self.X, self.y)
            score = f1_score(self.y, model.predict(self.X))
            
            if score > best_score:
                best_score = score
                best_params = params
                best_model = clone(model)
        
        if best_model is not None:
            self.models['random_forest'] = best_model
            self.results['random_forest'] = {
                'best_params': best_params,
                'best_score': float(best_score)
            }
            logger.info(f"Migliori parametri RF: {best_params}")
            logger.info(f"Miglior score RF: {best_score:.4f}")

    def optimize_isolation_forest(self):
        """Ottimizza Isolation Forest con griglia ridotta."""
        logger.info("Ottimizzazione Isolation Forest...")
        start_time = time.time()
        
        param_grid = {
            'n_estimators': [100],
            'contamination': [0.1],
            'max_samples': ['auto']
        }
        
        best_score = float('-inf')
        best_params = None
        best_model = None
        
        for params in [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]:
            if time.time() - start_time > self.max_time:
                logger.warning("Tempo massimo raggiunto per Isolation Forest")
                break
                
            logger.info(f"Testing parameters: {params}")
            model = IsolationForest(**params, random_state=42, n_jobs=-1)
            model.fit(self.X)
            
            y_pred = model.predict(self.X)
            y_pred = np.where(y_pred == 1, 0, 1)
            score = f1_score(self.y, y_pred)
            
            if score > best_score:
                best_score = score
                best_params = params
                best_model = clone(model)
        
        if best_model is not None:
            self.models['isolation_forest'] = best_model
            self.results['isolation_forest'] = {
                'best_params': best_params,
                'best_score': float(best_score)
            }
            logger.info(f"Migliori parametri IF: {best_params}")
            logger.info(f"Miglior score IF: {best_score:.4f}")

    def optimize_one_class_svm(self):
        """Ottimizza One-Class SVM con griglia ridotta."""
        logger.info("Ottimizzazione One-Class SVM...")
        start_time = time.time()
        
        X_normal = self.X[self.y == 0]
        
        param_grid = {
            'kernel': ['rbf'],
            'nu': [0.1],
            'gamma': ['scale']
        }
        
        best_score = float('-inf')
        best_params = None
        best_model = None
        
        for params in [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]:
            if time.time() - start_time > self.max_time:
                logger.warning("Tempo massimo raggiunto per One-Class SVM")
                break
                
            logger.info(f"Testing parameters: {params}")
            model = OneClassSVM(**params)
            model.fit(X_normal)
            
            y_pred = model.predict(self.X)
            y_pred = np.where(y_pred == 1, 0, 1)
            score = f1_score(self.y, y_pred)
            
            if score > best_score:
                best_score = score
                best_params = params
                best_model = clone(model)
        
        if best_model is not None:
            self.models['one_class_svm'] = best_model
            self.results['one_class_svm'] = {
                'best_params': best_params,
                'best_score': float(best_score)
            }
            logger.info(f"Migliori parametri SVM: {best_params}")
            logger.info(f"Miglior score SVM: {best_score:.4f}")

    def save_results(self):
        """Salva i risultati e i modelli ottimizzati."""
        logger.info("Salvataggio risultati...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Salva modelli
        for name, model in self.models.items():
            model_path = self.output_dir / f"best_{name}_{timestamp}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Modello {name} salvato in {model_path}")
        
        # Salva risultati
        results_path = self.output_dir / f"optimization_results_{timestamp}.json"
        import json
        with open(results_path, 'w') as f:
            results_json = {
                model_name: {
                    'best_params': {
                        k: v.item() if hasattr(v, 'item') else v
                        for k, v in res['best_params'].items()
                    },
                    'best_score': float(res['best_score'])
                }
                for model_name, res in self.results.items()
            }
            json.dump(results_json, f, indent=2)
        
        logger.info(f"Risultati salvati in {results_path}")

def main():
    """Funzione principale per l'esecuzione."""
    try:
        optimizer = ModelOptimizer(max_time=300)  # 5 minuti max per modello
        optimizer.load_data()
        
        optimizer.optimize_random_forest()
        optimizer.optimize_isolation_forest()
        optimizer.optimize_one_class_svm()
        
        optimizer.save_results()
        
        logger.info("Ottimizzazione completata con successo")
        
    except Exception as e:
        logger.error(f"Errore durante l'ottimizzazione: {str(e)}")
        raise

if __name__ == "__main__":
    main()