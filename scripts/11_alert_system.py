import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yaml
import time
from typing import Dict, List, Any, Optional

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AlertSystem:
    def __init__(self, 
                 config_path: str = "config/alert_config.yaml",
                 models_dir: str = "output/models",
                 ensemble_config_path: str = "output/ensemble/ensemble_results.json"):
        """
        Inizializza il sistema di alerting.
        
        Args:
            config_path: Path della configurazione alerting
            models_dir: Directory dei modelli
            ensemble_config_path: Path della configurazione ensemble
        """
        self.config_path = Path(config_path)
        self.models_dir = Path(models_dir)
        self.ensemble_config_path = Path(ensemble_config_path)
        
        self.config = self._load_config()
        self.models = {}
        self.ensemble_config = None
        self.alert_history = []
        self.alert_buffer = []
        
    def _load_config(self) -> dict:
        """Carica configurazione da YAML."""
        default_config = {
            'alert_thresholds': {
                'critical': 0.9,
                'high': 0.8,
                'medium': 0.7,
                'low': 0.6
            },
            'time_windows': {
                'buffer_size': 100,
                'aggregation_window': 300,  # 5 minuti
                'rate_threshold': 10  # alerts/minute
            },
            'notification': {
                'email': {
                    'enabled': False,
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'sender_email': '',
                    'sender_password': '',
                    'recipients': []
                },
                'slack': {
                    'enabled': False,
                    'webhook_url': ''
                }
            },
            'alert_rules': {
                'consecutive_anomalies': 3,
                'anomaly_rate_threshold': 0.2,
                'ip_blacklist': [],
                'location_blacklist': ['North Korea', 'Anonymous Proxy']
            }
        }
        
        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
                return {**default_config, **config}
        except FileNotFoundError:
            logger.warning(f"Config non trovato in {self.config_path}, uso default")
            return default_config

    def load_models_and_config(self):
        """Carica modelli e configurazione ensemble."""
        logger.info("Caricamento modelli e configurazione...")
        
        # Carica configurazione ensemble
        try:
            with open(self.ensemble_config_path) as f:
                self.ensemble_config = json.load(f)
            logger.info("Configurazione ensemble caricata")
        except FileNotFoundError:
            logger.error(f"Configurazione ensemble non trovata in {self.ensemble_config_path}")
            raise
        
        # Carica modelli
        model_files = list(self.models_dir.glob("*.joblib"))
        for model_file in model_files:
            model_name = model_file.stem.split('_')[0]
            self.models[model_name] = joblib.load(model_file)
            logger.info(f"Caricato modello: {model_name}")

    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa una singola richiesta e genera alert se necessario.
        
        Args:
            request_data: Dati della richiesta HTTP
            
        Returns:
            Dict con risultati dell'analisi e eventuali alert
        """
        # Prepara features
        features = self._extract_features(request_data)
        
        # Calcola probabilità di anomalia per ogni modello
        model_predictions = {}
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
        
        # Genera risultato
        result = {
            'timestamp': datetime.now().isoformat(),
            'request_id': request_data.get('request_id', ''),
            'ip_address': request_data.get('ip_address', ''),
            'model_predictions': model_predictions,
            'ensemble_score': ensemble_score,
            'is_anomaly': ensemble_score > self.ensemble_config['threshold'],
            'alert_level': self._get_alert_level(ensemble_score)
        }
        
        # Aggiorna buffer e genera alert se necessario
        self._update_alert_buffer(result)
        alerts = self._generate_alerts(result)
        
        if alerts:
            result['alerts'] = alerts
            self._send_notifications(alerts)
        
        return result

    def _extract_features(self, request_data: Dict[str, Any]) -> np.ndarray:
        """Estrae feature dalla richiesta."""
        # Implementa estrazione feature come in feature engineering
        # Questo è un placeholder
        features = np.zeros(10)  # Adatta alla dimensionalità corretta
        return features

    def _get_alert_level(self, score: float) -> str:
        """Determina il livello di alert basato sullo score."""
        thresholds = self.config['alert_thresholds']
        
        if score >= thresholds['critical']:
            return 'CRITICAL'
        elif score >= thresholds['high']:
            return 'HIGH'
        elif score >= thresholds['medium']:
            return 'MEDIUM'
        elif score >= thresholds['low']:
            return 'LOW'
        return 'INFO'

    def _update_alert_buffer(self, result: Dict[str, Any]):
        """Aggiorna il buffer delle alert con nuovo risultato."""
        self.alert_buffer.append(result)
        
        # Mantieni solo gli ultimi N risultati
        buffer_size = self.config['time_windows']['buffer_size']
        if len(self.alert_buffer) > buffer_size:
            self.alert_buffer = self.alert_buffer[-buffer_size:]

    def _generate_alerts(self, current_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genera alert basati sulle regole configurate."""
        alerts = []
        
        # Controlla regole
        rules = self.config['alert_rules']
        
        # Regola 1: Anomalie consecutive
        recent_results = self.alert_buffer[-rules['consecutive_anomalies']:]
        if (len(recent_results) >= rules['consecutive_anomalies'] and 
            all(r['is_anomaly'] for r in recent_results)):
            alerts.append({
                'type': 'CONSECUTIVE_ANOMALIES',
                'level': 'HIGH',
                'message': f"Rilevate {rules['consecutive_anomalies']} anomalie consecutive",
                'details': {
                    'anomalies': recent_results
                }
            })
        
        # Regola 2: Rate anomalie
        window_size = self.config['time_windows']['aggregation_window']
        window_start = datetime.now() - timedelta(seconds=window_size)
        window_results = [
            r for r in self.alert_buffer 
            if datetime.fromisoformat(r['timestamp']) > window_start
        ]
        
        anomaly_rate = sum(1 for r in window_results if r['is_anomaly']) / len(window_results) if window_results else 0
        
        if anomaly_rate > rules['anomaly_rate_threshold']:
            alerts.append({
                'type': 'HIGH_ANOMALY_RATE',
                'level': 'CRITICAL',
                'message': f"Rate anomalie ({anomaly_rate:.2%}) sopra la soglia",
                'details': {
                    'rate': anomaly_rate,
                    'threshold': rules['anomaly_rate_threshold'],
                    'window_size': window_size
                }
            })
        
        # Regola 3: IP e Location blacklist
        ip = current_result['ip_address']
        if ip in rules['ip_blacklist']:
            alerts.append({
                'type': 'BLACKLISTED_IP',
                'level': 'CRITICAL',
                'message': f"Rilevato IP in blacklist: {ip}",
                'details': {'ip': ip}
            })
        
        return alerts

    def _send_notifications(self, alerts: List[Dict[str, Any]]):
        """Invia notifiche per gli alert generati."""
        if not alerts:
            return
            
        # Email notifications
        if self.config['notification']['email']['enabled']:
            self._send_email_notification(alerts)
            
        # Slack notifications
        if self.config['notification']['slack']['enabled']:
            self._send_slack_notification(alerts)

    def _send_email_notification(self, alerts: List[Dict[str, Any]]):
        """Invia notifiche email."""
        email_config = self.config['notification']['email']
        
        # Crea messaggio
        msg = MIMEMultipart()
        msg['Subject'] = f"Security Alert - {alerts[0]['level']}"
        msg['From'] = email_config['sender_email']
        msg['To'] = ', '.join(email_config['recipients'])
        
        # Corpo del messaggio
        body = "Security Alerts:\n\n"
        for alert in alerts:
            body += f"Type: {alert['type']}\n"
            body += f"Level: {alert['level']}\n"
            body += f"Message: {alert['message']}\n"
            body += f"Details: {json.dumps(alert['details'], indent=2)}\n\n"
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Invia email
        try:
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                server.starttls()
                server.login(email_config['sender_email'], email_config['sender_password'])
                server.send_message(msg)
                logger.info("Email notification sent successfully")
        except Exception as e:
            logger.error(f"Error sending email notification: {str(e)}")

    def _send_slack_notification(self, alerts: List[Dict[str, Any]]):
        """Invia notifiche Slack."""
        # Implementa invio notifiche Slack
        pass

    def run(self, interval: int = 60):
        """
        Esegue il sistema di alerting in continuo.
        
        Args:
            interval: Intervallo di check in secondi
        """
        logger.info(f"Starting alert system with {interval}s interval")
        
        while True:
            try:
                # Simula ricezione richieste
                # In produzione, questo sarebbe integrato con il sistema reale
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Alert system stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in alert system: {str(e)}")
                continue

def main():
    """Funzione principale per l'esecuzione."""
    try:
        alert_system = AlertSystem()
        alert_system.load_models_and_config()
        alert_system.run()
        
    except Exception as e:
        logger.error(f"Error initializing alert system: {str(e)}")
        raise

if __name__ == "__main__":
    main()