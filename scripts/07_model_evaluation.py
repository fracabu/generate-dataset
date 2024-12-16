import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import logging
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_curve, roc_curve, auc,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, 
                 predictions_path: str = "output/predictions",
                 models_path: str = "output/models",
                 test_data_path: str = "data/raw/new_cybersecurity_data.csv",
                 output_path: str = "output/evaluation"):
        """
        Inizializza il valutatore di modelli.
        
        Args:
            predictions_path: Directory contenente le predizioni
            models_path: Directory contenente i modelli
            test_data_path: Path del dataset di test
            output_path: Directory per i risultati della valutazione
        """
        self.predictions_dir = Path(predictions_path)
        self.models_dir = Path(models_path)
        self.test_data_path = Path(test_data_path)
        self.output_dir = Path(output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sottodir per i plot
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.data = None
        self.predictions = None
        self.evaluation_results = {}

    def load_data(self):
        """Carica dati e predizioni."""
        logger.info("Caricamento dati e predizioni...")
        
        # Carica dati di test
        self.data = pd.read_csv(self.test_data_path)
        
        # Trova l'ultimo file di predizioni
        prediction_files = list(self.predictions_dir.glob("predictions_*.json"))
        if not prediction_files:
            raise FileNotFoundError("Nessun file di predizioni trovato")
        
        latest_prediction = max(prediction_files, key=lambda x: x.stat().st_mtime)
        
        # Carica predizioni
        with open(latest_prediction) as f:
            self.predictions = json.load(f)

    def evaluate_models(self):
        """Valuta le performance dei modelli."""
        logger.info("Valutazione modelli...")
        
        true_labels = self.data['Anomaly_Flag']
        
        for model_name, pred_data in self.predictions['predictions'].items():
            logger.info(f"Valutazione {model_name}...")
            
            pred_labels = pred_data['class']
            pred_probs = pred_data['probability'] if pred_data['probability'] else None
            
            model_eval = {
                'confusion_matrix': confusion_matrix(true_labels, pred_labels).tolist(),
                'classification_report': classification_report(true_labels, pred_labels, output_dict=True)
            }
            
            if pred_probs:
                precision, recall, pr_thresholds = precision_recall_curve(true_labels, pred_probs)
                fpr, tpr, roc_thresholds = roc_curve(true_labels, pred_probs)
                
                model_eval.update({
                    'pr_curve': {
                        'precision': precision.tolist(),
                        'recall': recall.tolist(),
                        'thresholds': pr_thresholds.tolist()
                    },
                    'roc_curve': {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                        'thresholds': roc_thresholds.tolist()
                    },
                    'auc_roc': auc(fpr, tpr),
                    'average_precision': average_precision_score(true_labels, pred_probs)
                })
            
            self.evaluation_results[model_name] = model_eval
            
            # Genera plot
            self._generate_model_plots(model_name, model_eval)
        
        # Valuta ensemble
        logger.info("Valutazione ensemble...")
        ensemble_eval = {
            'confusion_matrix': confusion_matrix(true_labels, self.predictions['ensemble']).tolist(),
            'classification_report': classification_report(true_labels, self.predictions['ensemble'], output_dict=True)
        }
        self.evaluation_results['ensemble'] = ensemble_eval
        self._generate_model_plots('ensemble', ensemble_eval)

    def _generate_model_plots(self, model_name: str, evaluation: dict):
        """Genera plot per un modello."""
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(evaluation['confusion_matrix'], 
                   annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.savefig(self.plots_dir / f'{model_name}_confusion_matrix.png')
        plt.close()
        
        # ROC e PR curves se disponibili
        if 'roc_curve' in evaluation:
            plt.figure(figsize=(8, 6))
            plt.plot(evaluation['roc_curve']['fpr'], 
                    evaluation['roc_curve']['tpr'], 
                    label=f'ROC curve (AUC = {evaluation["auc_roc"]:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend()
            plt.savefig(self.plots_dir / f'{model_name}_roc_curve.png')
            plt.close()
            
            plt.figure(figsize=(8, 6))
            plt.plot(evaluation['pr_curve']['recall'],
                    evaluation['pr_curve']['precision'],
                    label=f'PR curve (AP = {evaluation["average_precision"]:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {model_name}')
            plt.legend()
            plt.savefig(self.plots_dir / f'{model_name}_pr_curve.png')
            plt.close()

    def analyze_error_patterns(self):
        """Analizza i pattern degli errori."""
        logger.info("Analisi pattern degli errori...")
        
        error_patterns = {}
        true_labels = self.data['Anomaly_Flag']
        
        for model_name, pred_data in self.predictions['predictions'].items():
            pred_labels = pred_data['class']
            
            # Trova gli indici degli errori
            false_positives = (pred_labels == 1) & (true_labels == 0)
            false_negatives = (pred_labels == 0) & (true_labels == 1)
            
            # Analizza caratteristiche degli errori
            error_patterns[model_name] = {
                'false_positives': {
                    'count': int(false_positives.sum()),
                    'top_locations': self.data[false_positives]['Location'].value_counts().head(5).to_dict(),
                    'top_request_types': self.data[false_positives]['Request_Type'].value_counts().head(5).to_dict(),
                    'status_codes': self.data[false_positives]['Status_Code'].value_counts().to_dict()
                },
                'false_negatives': {
                    'count': int(false_negatives.sum()),
                    'top_locations': self.data[false_negatives]['Location'].value_counts().head(5).to_dict(),
                    'top_request_types': self.data[false_negatives]['Request_Type'].value_counts().head(5).to_dict(),
                    'status_codes': self.data[false_negatives]['Status_Code'].value_counts().to_dict()
                }
            }
        
        self.evaluation_results['error_patterns'] = error_patterns

    def save_results(self):
        """Salva i risultati della valutazione."""
        logger.info("Salvataggio risultati...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"evaluation_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        logger.info(f"Risultati salvati in {results_file}")
        
        # Genera report in markdown
        self._generate_markdown_report(timestamp)

    def _generate_markdown_report(self, timestamp: str):
        """Genera un report in formato markdown."""
        report_file = self.output_dir / f"evaluation_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Evaluation Report\n\n")
            
            for model_name, eval_data in self.evaluation_results.items():
                if model_name != 'error_patterns':
                    f.write(f"## {model_name}\n\n")
                    
                    # Metriche principali
                    if 'classification_report' in eval_data:
                        metrics = eval_data['classification_report']['weighted avg']
                        f.write("### Main Metrics\n")
                        f.write(f"- Precision: {metrics['precision']:.3f}\n")
                        f.write(f"- Recall: {metrics['recall']:.3f}\n")
                        f.write(f"- F1-score: {metrics['f1-score']:.3f}\n")
                    
                    if 'auc_roc' in eval_data:
                        f.write(f"- AUC-ROC: {eval_data['auc_roc']:.3f}\n")
                        f.write(f"- Average Precision: {eval_data['average_precision']:.3f}\n")
                    
                    f.write("\n")
            
            # Pattern degli errori
            f.write("## Error Patterns Analysis\n\n")
            for model_name, patterns in self.evaluation_results['error_patterns'].items():
                f.write(f"### {model_name}\n")
                f.write(f"- False Positives: {patterns['false_positives']['count']}\n")
                f.write(f"- False Negatives: {patterns['false_negatives']['count']}\n\n")

def main():
    """Funzione principale per l'esecuzione."""
    try:
        evaluator = ModelEvaluator()
        evaluator.load_data()
        evaluator.evaluate_models()
        evaluator.analyze_error_patterns()
        evaluator.save_results()
        logger.info("Valutazione completata con successo")
        
    except Exception as e:
        logger.error(f"Errore durante la valutazione: {str(e)}")
        raise

if __name__ == "__main__":
    main()