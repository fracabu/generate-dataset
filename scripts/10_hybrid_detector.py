import pandas as pd 
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import yaml
from sklearn.base import BaseEstimator, ClassifierMixin

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RuleBasedDetector:
    """Detector basato su regole."""
    def __init__(self, rules_config: Dict):
        self.rules = rules_config
        
    def check_ip_rules(self, ip: str) -> bool:
        """Verifica regole basate su IP."""
        if ip in self.rules['ip_blacklist']:
            return True
        return False
        
    def check_request_rate(self, ip: str, requests: List[Dict]) -> bool:
        """Verifica rate delle richieste."""
        ip_requests = [r for r in requests if r['ip_address'] == ip]
        rate = len(ip_requests) / self.rules['time_window']
        return rate > self.rules['max_request_rate']
        
    def check_location_rules(self, location: str) -> bool:
        """Verifica regole basate su location."""
        return location in self.rules['location_blacklist']
        
    def check_pattern_rules(self, request: Dict) -> bool:
        """Verifica pattern sospetti nelle richieste."""
        if request['status_code'] in self.rules['suspicious_status_codes']:
            return True
        if any(pattern in request['user_agent'] for pattern in self.rules['suspicious_user_agents']):
            return True
        return False

class HybridDetector(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 config_path: str = "config/hybrid_config.yaml",
                 models_dir: str = "output/models",
                 ensemble_config_path: str = "output/ensemble/ensemble_results.json"):
        """
        Inizializza il detector ibrido.
        
        Args:
            config_path: Path della configurazione
            models_dir: Directory dei modelli
            ensemble_config_path: Path della configurazione ensemble
        """
        self.config_path = Path(config_path)
        self.models_dir = Path(models_dir)
        self.ensemble_config_path = Path(ensemble_config_path)
        
        self.config = self._load_config()
        self.models = {}
        self.ensemble_config = None
        self.rule_detector = RuleBasedDetector(self.config['rules'])
        self.request_buffer = []
        
    def _load_config(self) -> dict:
        """Carica configurazione."""
        default_config = {
            'rules': {
                'ip_blacklist': [],
                'location_blacklist': ['North Korea', 'Anonymous Proxy'],
                'suspicious_status_codes': [400, 401, 403, 404, 500],
                'suspicious_user_agents': ['bot', 'crawler', 'spider'],
                'time_window': 60,  # secondi
                'max_request_rate': 100  # richieste/minuto
            },
            'hybrid': {
                'rule_weight': 0.3,
                'ml_weight': 0.7,
                'threshold': 0.8
            }
        }
        
        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
                return {**default_config, **config}
        except FileNotFoundError:
            logger.warning(f"Config non trovato in {self.config_path}, uso default")
            return default_config
        

    def get_ml_prediction(self, features: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Ottiene predizione dai modelli ML."""
        model_predictions = {}
        
        # Predizioni singoli modelli
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(features.reshape(1, -1))[0][1]
            else:
                pred = model.predict(features.reshape(1, -1))[0]
                prob = 1.0 if pred == -1 else 0.0
            model_predictions[name] = prob
        
        # Calcola score ensemble
        ensemble_score = sum(prob * self.ensemble_config['weights'][name]
                           for name, prob in model_predictions.items())
        
        return ensemble_score, model_predictions

    def get_rule_prediction(self, request: Dict) -> Tuple[float, Dict[str, bool]]:
        """Ottiene predizione dalle regole."""
        rule_results = {
            'ip_check': self.rule_detector.check_ip_rules(request['ip_address']),
            'rate_check': self.rule_detector.check_request_rate(
                request['ip_address'], self.request_buffer
            ),
            'location_check': self.rule_detector.check_location_rules(request['location']),
            'pattern_check': self.rule_detector.check_pattern_rules(request)
        }
        
        # Calcola score regole
        rule_score = sum(rule_results.values()) / len(rule_results)
        
        return rule_score, rule_results

    def predict(self, X: np.ndarray, request_data: Dict) -> np.ndarray:
        """
        Predice anomalie usando approccio ibrido.
        
        Args:
            X: Features per modelli ML
            request_data: Dati della richiesta per regole
            
        Returns:
            Array di predizioni (0: normale, 1: anomalia)
        """
        # Aggiorna buffer richieste
        self.request_buffer.append(request_data)
        if len(self.request_buffer) > 1000:  # Mantieni ultimi 1000 record
            self.request_buffer = self.request_buffer[-1000:]
        
        # Ottieni predizioni ML e regole
        ml_score, ml_predictions = self.get_ml_prediction(X)
        rule_score, rule_results = self.get_rule_prediction(request_data)
        
        # Combina scores con pesi configurati
        hybrid_score = (
            self.config['hybrid']['ml_weight'] * ml_score +
            self.config['hybrid']['rule_weight'] * rule_score
        )
        
        # Applica soglia
        is_anomaly = hybrid_score > self.config['hybrid']['threshold']
        
        # Prepara risultato dettagliato
        result = {
            'timestamp': datetime.now().isoformat(),
            'hybrid_score': hybrid_score,
            'ml_score': ml_score,
            'rule_score': rule_score,
            'ml_predictions': ml_predictions,
            'rule_results': rule_results,
            'is_anomaly': is_anomaly
        }
        
        # Salva risultato
        self._save_result(result)
        
        return np.array([1 if is_anomaly else 0])

    def predict_proba(self, X: np.ndarray, request_data: Dict) -> np.ndarray:
        """Restituisce probabilità di anomalia."""
        # Ottieni predizioni ML e regole
        ml_score, _ = self.get_ml_prediction(X)
        rule_score, _ = self.get_rule_prediction(request_data)
        
        # Combina scores
        hybrid_score = (
            self.config['hybrid']['ml_weight'] * ml_score +
            self.config['hybrid']['rule_weight'] * rule_score
        )
        
        # Formato richiesto da sklearn
        return np.array([[1 - hybrid_score, hybrid_score]])

    def _save_result(self, result: Dict):
        """Salva risultato predizione."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("output/hybrid_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / f"prediction_{timestamp}.json", 'w') as f:
            json.dump(result, f, indent=2)

def main():
    """Funzione principale per l'esecuzione."""
    try:
        detector = HybridDetector()
        detector.load_models()
        
        # Test su alcuni dati di esempio
        X = np.random.rand(10, 10)  # Esempio features
        request_data = {
            'ip_address': '192.168.1.1',
            'location': 'USA',
            'user_agent': 'Mozilla/5.0',
            'status_code': 200
        }
        
        predictions = detector.predict(X, request_data)
        probabilities = detector.predict_proba(X, request_data)
        
        logger.info(f"Predictions: {predictions}")
        logger.info(f"Probabilities: {probabilities}")
        
    except Exception as e:
        logger.error(f"Errore durante l'esecuzione: {str(e)}")
        raise

if __name__ == "__main__":
    main()