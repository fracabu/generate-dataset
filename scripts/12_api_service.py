from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import uvicorn
from pydantic import BaseModel, Field
import yaml

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Modelli Pydantic per la validazione
class RequestData(BaseModel):
    ip_address: str = Field(..., example="192.168.1.1")
    timestamp: Optional[str] = Field(None, example="2024-12-16T10:00:00")
    request_type: str = Field(..., example="GET")
    status_code: int = Field(..., example=200)
    user_agent: str = Field(..., example="Mozilla/5.0")
    session_id: str = Field(..., example="abc123")
    location: str = Field(..., example="USA")

class PredictionResponse(BaseModel):
    request_id: str
    timestamp: str
    prediction: Dict[str, Any]
    anomaly_score: float
    is_anomaly: bool
    details: Dict[str, Any]

class APIService:
    def __init__(self, 
                 config_path: str = "config/api_config.yaml",
                 models_dir: str = "output/models",
                 feature_pipeline_path: str = "output/feature_pipeline.joblib",
                 hybrid_detector_path: str = "output/models/hybrid_detector.joblib"):
        """
        Inizializza il servizio API.
        
        Args:
            config_path: Path della configurazione
            models_dir: Directory dei modelli
            feature_pipeline_path: Path del pipeline di feature engineering
            hybrid_detector_path: Path del detector ibrido
        """
        self.config = self._load_config(config_path)
        self.models_dir = Path(models_dir)
        self.feature_pipeline = None
        self.hybrid_detector = None
        self.request_buffer = []
        
        # Carica modelli e pipeline
        self._load_models(feature_pipeline_path, hybrid_detector_path)
        
        # Inizializza FastAPI
        self.app = FastAPI(
            title="Anomaly Detection API",
            description="API per il rilevamento di anomalie in tempo reale",
            version="1.0.0"
        )
        
        # Aggiungi CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config['cors']['allowed_origins'],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Registra routes
        self._setup_routes()

    def _load_config(self, config_path: str) -> dict:
        """Carica configurazione."""
        default_config = {
            'server': {
                'host': '0.0.0.0',
                'port': 8000,
                'workers': 4
            },
            'cors': {
                'allowed_origins': ['*']
            },
            'rate_limiting': {
                'enabled': True,
                'max_requests': 100,
                'time_window': 60
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/api.log'
            }
        }
        
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                return {**default_config, **config}
        except FileNotFoundError:
            logger.warning(f"Config non trovato in {config_path}, uso default")
            return default_config

    def _load_models(self, feature_pipeline_path: str, hybrid_detector_path: str):
        """Carica modelli e pipeline."""
        try:
            self.feature_pipeline = joblib.load(feature_pipeline_path)
            logger.info("Feature pipeline caricato")
            
            self.hybrid_detector = joblib.load(hybrid_detector_path)
            logger.info("Hybrid detector caricato")
            
        except Exception as e:
            logger.error(f"Errore nel caricamento modelli: {str(e)}")
            raise

    def _setup_routes(self):
        """Configura le routes dell'API."""
        
        @self.app.get("/")
        async def root():
            """Health check endpoint."""
            return {"status": "ok", "timestamp": datetime.now().isoformat()}

        @self.app.get("/status")
        async def status():
            """Stato dettagliato del servizio."""
            return {
                "status": "operational",
                "models_loaded": bool(self.hybrid_detector),
                "pipeline_loaded": bool(self.feature_pipeline),
                "requests_processed": len(self.request_buffer),
                "timestamp": datetime.now().isoformat()
            }

        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: RequestData, background_tasks: BackgroundTasks):
            """Endpoint principale per predizioni."""
            try:
                # Prepara i dati
                features = self._prepare_features(request)
                
                # Ottieni predizione
                prediction = self._get_prediction(features, request.dict())
                
                # Aggiorna buffer in background
                background_tasks.add_task(self._update_request_buffer, request.dict())
                
                return prediction
                
            except Exception as e:
                logger.error(f"Errore durante la predizione: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/metrics")
        async def metrics():
            """Metriche del servizio."""
            return self._get_metrics()

    def _prepare_features(self, request: RequestData) -> np.ndarray:
        """Prepara features per la predizione."""
        # Converti request in DataFrame
        df = pd.DataFrame([request.dict()])
        
        # Applica feature engineering
        features = self.feature_pipeline.transform(df)
        
        return features

    def _get_prediction(self, features: np.ndarray, request_data: Dict) -> Dict:
        """Ottiene predizione dal modello."""
        # Ottieni predizioni dal detector ibrido
        prediction = self.hybrid_detector.predict(features, request_data)
        probabilities = self.hybrid_detector.predict_proba(features, request_data)
        
        result = {
            'request_id': f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'prediction': {
                'class': int(prediction[0]),
                'probability': float(probabilities[0][1])
            },
            'anomaly_score': float(probabilities[0][1]),
            'is_anomaly': bool(prediction[0]),
            'details': {
                'features_shape': features.shape,
                'model_version': getattr(self.hybrid_detector, 'version', 'unknown')
            }
        }
        
        return result

    def _update_request_buffer(self, request_data: Dict):
        """Aggiorna il buffer delle richieste."""
        self.request_buffer.append({
            **request_data,
            'timestamp': datetime.now().isoformat()
        })
        
        # Mantieni solo ultimi N record
        if len(self.request_buffer) > 1000:
            self.request_buffer = self.request_buffer[-1000:]

    def _get_metrics(self) -> Dict:
        """Calcola metriche del servizio."""
        # Calcola statistiche di base
        total_requests = len(self.request_buffer)
        if total_requests == 0:
            return {"error": "No data available"}
        
        recent_requests = [r for r in self.request_buffer[-100:]]
        
        return {
            'total_requests': total_requests,
            'requests_per_minute': len(recent_requests),
            'unique_ips': len(set(r['ip_address'] for r in recent_requests)),
            'status_codes': {
                str(k): v for k, v in 
                pd.Series([r['status_code'] for r in recent_requests]).value_counts().items()
            },
            'timestamp': datetime.now().isoformat()
        }

    def run(self):
        """Avvia il server API."""
        config = self.config['server']
        uvicorn.run(
            self.app,
            host=config['host'],
            port=config['port'],
            workers=config['workers']
        )

def main():
    """Funzione principale per l'esecuzione."""
    try:
        service = APIService()
        service.run()
        
    except Exception as e:
        logger.error(f"Errore durante l'avvio del servizio: {str(e)}")
        raise

if __name__ == "__main__":
    main()