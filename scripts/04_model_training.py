import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, 
                           precision_recall_curve, roc_curve, auc)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnomalyModelTrainer:
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Inizializza il trainer con configurazione opzionale."""
        self.config = self._load_config(config_path)
        self.models = {}
        self.results = {}
        self.plots_dir = Path("output/plots")
        self.models_dir = Path("output/models")
        self.metrics_dir = Path("output/metrics")
        
        # Crea directories
        for dir_path in [self.plots_dir, self.models_dir, self.metrics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        """Carica configurazione o usa default."""
        default_config = {
            'data': {
                'input_path': 'output/processed_features.csv',
                'test_size': 0.2,
                'random_state': 42
            },
            'models': {
                'random_forest': {
                    'class': 'RandomForestClassifier',
                    'params': {
                        'n_estimators': 100,
                        'max_depth': 15,
                        'min_samples_split': 5
                    }
                },
                'isolation_forest': {
                    'class': 'IsolationForest',
                    'params': {
                        'contamination': 0.1,
                        'random_state': 42
                    }
                },
                'one_class_svm': {
                    'class': 'OneClassSVM',
                    'params': {
                        'kernel': 'rbf',
                        'nu': 0.1
                    }
                }
            }
        }
        
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config non trovato in {config_path}, uso default")
            return default_config

    def load_data(self):
        """Carica e prepara i dati."""
        logger.info("Caricamento dati...")
        df = pd.read_csv(self.config['data']['input_path'])
        
        # Separa features e target
        X = df.drop('Anomaly_Flag', axis=1)
        y = df['Anomaly_Flag']
        
        # Split train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state']
        )
        
        logger.info(f"Dataset diviso in train ({len(self.X_train)} samples) e test ({len(self.X_test)} samples)")
        logger.info(f"Distribuzione anomalie nel train: {self.y_train.mean()*100:.2f}%")

    def train_models(self):
        """Addestra tutti i modelli configurati."""
        logger.info("Inizio training modelli...")
        
        model_classes = {
            'RandomForestClassifier': RandomForestClassifier,
            'IsolationForest': IsolationForest,
            'OneClassSVM': OneClassSVM
        }
        
        for model_name, config in self.config['models'].items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Inizializza modello
                model_class = model_classes[config['class']]
                model = model_class(**config['params'])
                
                # Training
                if isinstance(model, (IsolationForest, OneClassSVM)):
                    # Per modelli di anomaly detection
                    model.fit(self.X_train)
                    train_pred = pd.Series(model.predict(self.X_train)).map({1: 0, -1: 1})
                    test_pred = pd.Series(model.predict(self.X_test)).map({1: 0, -1: 1})
                else:
                    # Per classificatori tradizionali
                    model.fit(self.X_train, self.y_train)
                    train_pred = model.predict(self.X_train)
                    test_pred = model.predict(self.X_test)
                
                # Salva modello e predizioni
                self.models[model_name] = {
                    'model': model,
                    'predictions': {
                        'train': train_pred,
                        'test': test_pred
                    }
                }
                
                # Calcola e salva metriche
                self._calculate_metrics(model_name)
                
                # Genera plot
                self._generate_plots(model_name)
                
            except Exception as e:
                logger.error(f"Errore nel training di {model_name}: {str(e)}")
                continue

    def _calculate_metrics(self, model_name):
        """Calcola metriche per un modello."""
        predictions = self.models[model_name]['predictions']
        
        self.results[model_name] = {
            'train': classification_report(self.y_train, predictions['train'], output_dict=True),
            'test': classification_report(self.y_test, predictions['test'], output_dict=True),
            'confusion_matrix': confusion_matrix(self.y_test, predictions['test']).tolist()
        }

    def _generate_plots(self, model_name):
        """Genera plot per un modello."""
        predictions = self.models[model_name]['predictions']
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(self.y_test, predictions['test'], normalize='true'),
                   annot=True, fmt='.2%', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.savefig(self.plots_dir / f'{model_name}_confusion_matrix.png')
        plt.close()
        
        # ROC Curve
        if hasattr(self.models[model_name]['model'], 'predict_proba'):
            probs = self.models[model_name]['model'].predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, probs)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            plt.savefig(self.plots_dir / f'{model_name}_roc.png')
            plt.close()

    def save_results(self):
        """Salva tutti i risultati."""
        logger.info("Salvataggio risultati...")
        
        # Salva modelli
        for name, model_dict in self.models.items():
            joblib.dump(model_dict['model'], 
                       self.models_dir / f'{name}.joblib')
        
        # Salva metriche
        with open(self.metrics_dir / 'metrics.yaml', 'w') as f:
            yaml.dump(self.results, f)
        
        logger.info("Salvataggio completato")
        
        # Mostra risultati principali
        for name, result in self.results.items():
            logger.info(f"\nRisultati per {name}:")
            logger.info(f"F1-score (test): {result['test']['weighted avg']['f1-score']:.3f}")
            logger.info(f"Precision (test): {result['test']['weighted avg']['precision']:.3f}")
            logger.info(f"Recall (test): {result['test']['weighted avg']['recall']:.3f}")

def main():
    """Funzione principale per l'esecuzione."""
    trainer = AnomalyModelTrainer()
    trainer.load_data()
    trainer.train_models()
    trainer.save_results()

if __name__ == "__main__":
    main()