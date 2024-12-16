import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from datetime import datetime
import optuna

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnsembleOptimizer:
    def __init__(self, 
                models_dir: str = "output/optimization",
                data_path: str = "output/processed_features.csv",
                output_dir: str = "output/ensemble"):
        self.models_dir = Path(models_dir)
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.data = None
        self.X = None
        self.y = None
        self.best_config = None
        self.optimization_history = []

    
    def load_data_and_models(self):
        """Carica dati e modelli ottimizzati."""
        logger.info("Caricamento dati e modelli...")
        logger.info(f"Directory base: {self.models_dir}")
        
        # Carica dati
        self.data = pd.read_csv(self.data_path)
        self.X = self.data.drop('Anomaly_Flag', axis=1)
        self.y = self.data['Anomaly_Flag']
        logger.info(f"Dataset caricato: {len(self.data)} campioni")
        
        # Cerca i modelli più recenti per tipo
        model_files = {
            'random_forest': list(self.models_dir.glob("best_random_forest_*.joblib")),
            'isolation_forest': list(self.models_dir.glob("best_isolation_forest_*.joblib")),
            'one_class_svm': list(self.models_dir.glob("best_one_class_svm_*.joblib"))
        }
        
        # Per ogni tipo di modello, prendi il più recente
        for model_type, files in model_files.items():
            if files:
                # Ordina per data di modifica e prendi l'ultimo
                latest_model = max(files, key=lambda x: x.stat().st_mtime)
                logger.info(f"Trovato modello {model_type}: {latest_model.name}")
                
                try:
                    self.models[model_type] = joblib.load(latest_model)
                    logger.info(f"Caricato modello {model_type}")
                except Exception as e:
                    logger.error(f"Errore nel caricamento del modello {model_type}: {str(e)}")
            else:
                logger.warning(f"Nessun modello trovato per {model_type}")

        if not self.models:
            raise ValueError("Nessun modello valido trovato nella directory")
            
        # Preparazione e fit di tutti i modelli
        X_normal = self.X[self.y == 0]
        for name, model in self.models.items():
            if isinstance(model, (IsolationForest, OneClassSVM)):
                logger.info(f"Fitting {name} sui dati normali")
                model.fit(X_normal)
            else:
                # Per i modelli supervisionati come RandomForest
                logger.info(f"Fitting {name} su tutto il dataset")
                model.fit(self.X, self.y)

        # Preparazione modelli non supervisionati
        X_normal = self.X[self.y == 0]
        for name, model in self.models.items():
            if isinstance(model, (IsolationForest, OneClassSVM)):
                logger.info(f"Fitting {name} sui dati normali")
                model.fit(X_normal)

    def get_model_predictions(self, X):
        """Ottiene predizioni da tutti i modelli."""
        predictions = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X)[:, 1]
                    predictions[name] = pred_proba
                else:
                    pred = model.predict(X)
                    pred = pd.Series(pred).map({1: 0, -1: 1}).values
                    predictions[name] = pred
                logger.debug(f"Ottenute predizioni per {name}")
            except Exception as e:
                logger.error(f"Errore nel predire con {name}: {str(e)}")
                raise
                
        return predictions

    def objective(self, trial):
        """Funzione obiettivo per Optuna."""
        # Pesi per ogni modello
        weights = {
            name: trial.suggest_float(f"weight_{name}", 0, 1)
            for name in self.models.keys()
        }
        
        # Normalizza i pesi
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        # Soglia decisionale
        threshold = trial.suggest_float("threshold", 0.3, 0.7)
        
        # Calcola predizioni pesate
        try:
            predictions = self.get_model_predictions(self.X)
            weighted_pred = np.zeros(len(self.y))
            
            for name, pred in predictions.items():
                weighted_pred += pred * weights[name]
            
            # Applica soglia
            final_pred = (weighted_pred > threshold).astype(int)
            
            # Calcola metriche
            metrics = {
                'f1': f1_score(self.y, final_pred),
                'precision': precision_score(self.y, final_pred),
                'recall': recall_score(self.y, final_pred)
            }
            
            # Salva configurazione e risultati
            self.optimization_history.append({
                'weights': weights,
                'threshold': threshold,
                'metrics': metrics
            })
            
            return metrics['f1']
            
        except Exception as e:
            logger.error(f"Errore nella funzione obiettivo: {str(e)}")
            raise

    def optimize(self, n_trials=50):
        """Esegue l'ottimizzazione dell'ensemble."""
        logger.info("Inizio ottimizzazione ensemble...")
        
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        
        # Salva miglior configurazione
        self.best_config = {
            'weights': {
                name: study.best_params[f"weight_{name}"]
                for name in self.models.keys()
            },
            'threshold': study.best_params['threshold']
        }
        
        # Normalizza i pesi finali
        total = sum(self.best_config['weights'].values())
        self.best_config['weights'] = {
            k: v/total for k, v in self.best_config['weights'].items()
        }
        
        logger.info(f"Ottimizzazione completata - Miglior F1-score: {study.best_value:.4f}")
        return study

    def evaluate_ensemble(self):
        """Valuta le performance dell'ensemble ottimizzato."""
        logger.info("Valutazione ensemble...")
        
        try:
            predictions = self.get_model_predictions(self.X)
            weighted_pred = np.zeros(len(self.y))
            
            for name, pred in predictions.items():
                weighted_pred += pred * self.best_config['weights'][name]
            
            final_pred = (weighted_pred > self.best_config['threshold']).astype(int)
            
            metrics = {
                'f1': f1_score(self.y, final_pred),
                'precision': precision_score(self.y, final_pred),
                'recall': recall_score(self.y, final_pred),
                'confusion_matrix': confusion_matrix(self.y, final_pred).tolist()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Errore nella valutazione: {str(e)}")
            raise

    def save_results(self):
        """Salva i risultati dell'ottimizzazione."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Valuta performance finali
            final_metrics = self.evaluate_ensemble()
            
            # Prepara risultati completi
            results = {
                'best_configuration': self.best_config,
                'final_metrics': final_metrics,
                'optimization_history': [
                    {
                        'weights': h['weights'],
                        'threshold': h['threshold'],
                        'metrics': {k: float(v) for k, v in h['metrics'].items()}
                    }
                    for h in self.optimization_history
                ]
            }
            
            # Salva risultati
            results_path = self.output_dir / f"ensemble_results_{timestamp}.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Genera report markdown
            report_path = self.output_dir / f"ensemble_report_{timestamp}.md"
            self._generate_report(report_path, final_metrics)
            
            logger.info(f"Risultati salvati in {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Errore nel salvataggio risultati: {str(e)}")
            raise

    def _generate_report(self, report_path, metrics):
        """Genera report markdown dei risultati."""
        try:
            with open(report_path, 'w') as f:
                f.write("# Ensemble Optimization Report\n\n")
                
                f.write("## Best Configuration\n\n")
                f.write("### Model Weights\n")
                for model, weight in self.best_config['weights'].items():
                    f.write(f"- {model}: {weight:.4f}\n")
                
                f.write(f"\n### Decision Threshold: {self.best_config['threshold']:.4f}\n\n")
                
                f.write("## Final Performance Metrics\n\n")
                f.write(f"- F1-score: {metrics['f1']:.4f}\n")
                f.write(f"- Precision: {metrics['precision']:.4f}\n")
                f.write(f"- Recall: {metrics['recall']:.4f}\n\n")
                
                f.write("### Confusion Matrix\n")
                cm = np.array(metrics['confusion_matrix'])
                f.write("```\n")
                f.write("          Predicted\n")
                f.write("         Norm  Anom\n")
                f.write(f"Actual Norm  {cm[0,0]:4d}  {cm[0,1]:4d}\n")
                f.write(f"      Anom  {cm[1,0]:4d}  {cm[1,1]:4d}\n")
                f.write("```\n\n")
                
        except Exception as e:
            logger.error(f"Errore nella generazione del report: {str(e)}")
            raise

def main():
    """Funzione principale per l'esecuzione."""
    try:
        optimizer = EnsembleOptimizer()
        optimizer.load_data_and_models()
        study = optimizer.optimize(n_trials=50)
        optimizer.save_results()
        logger.info("Processo completato con successo")
        
    except Exception as e:
        logger.error(f"Errore durante l'ottimizzazione: {str(e)}")
        raise

if __name__ == "__main__":
    main()