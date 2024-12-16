import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestDataGenerator:
    def __init__(self, num_rows=5000):
        """
        Inizializza il generatore di dati di test.
        
        Args:
            num_rows: Numero di righe da generare
        """
        self.num_rows = num_rows
        self.output_dir = Path("data/raw")
        
        # Configurazioni di base
        self.user_agents = ['Chrome', 'Firefox', 'Safari', 'Edge', 'Opera', 'Bot', 'Unknown_Bot']
        self.locations = ['USA', 'Germany', 'France', 'India', 'China', 'Brazil', 'Canada', 'Russia']
        self.request_types = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']
        
    def generate_ip(self):
        """Genera un indirizzo IP casuale."""
        return f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"
    
    def generate_base_data(self):
        """Genera i dati base del dataset."""
        logger.info("Generazione dati base...")
        
        # Genera timestamp più recenti
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        timestamps = [start_time + timedelta(minutes=random.randint(1, 1440*7)) 
                     for _ in range(self.num_rows)]
        timestamps.sort()
        
        # Genera altri dati casuali
        data = {
            'Timestamp': timestamps,
            'IP_Address': [self.generate_ip() for _ in range(self.num_rows)],
            'Request_Type': [random.choice(self.request_types) for _ in range(self.num_rows)],
            'Status_Code': [random.choice([200, 404, 500, 403, 301, 400, 502]) for _ in range(self.num_rows)],
            'User_Agent': [random.choice(self.user_agents) for _ in range(self.num_rows)],
            'Session_ID': [random.randint(5000, 9999) for _ in range(self.num_rows)],
            'Location': [random.choice(self.locations) for _ in range(self.num_rows)]
        }
        
        return pd.DataFrame(data)
    
    def inject_anomalies(self, df):
        """Inietta vari tipi di anomalie nel dataset."""
        logger.info("Iniezione anomalie...")
        
        # Inizializza flag anomalie
        df['Anomaly_Flag'] = 0
        
        # 1. Attacco DDoS (molte richieste da pochi IP)
        ddos_ips = [self.generate_ip() for _ in range(3)]
        num_ddos = int(self.num_rows * 0.03)  # 3% del dataset
        ddos_indices = random.sample(range(self.num_rows), num_ddos)
        
        for idx in ddos_indices:
            df.loc[idx, 'IP_Address'] = random.choice(ddos_ips)
            df.loc[idx, 'Request_Type'] = 'GET'
            df.loc[idx, 'Status_Code'] = random.choice([200, 404, 500])
            df.loc[idx, 'Anomaly_Flag'] = 1
        
        # 2. Scanning attacks (richieste sequenziali da un IP)
        scanner_ip = self.generate_ip()
        num_scans = int(self.num_rows * 0.02)  # 2% del dataset
        scan_indices = random.sample(range(self.num_rows), num_scans)
        
        for idx in scan_indices:
            df.loc[idx, 'IP_Address'] = scanner_ip
            df.loc[idx, 'Request_Type'] = random.choice(['GET', 'HEAD'])
            df.loc[idx, 'Status_Code'] = 404
            df.loc[idx, 'Anomaly_Flag'] = 1
        
        # 3. Suspicious locations
        suspicious_locations = ['North Korea', 'Anonymous Proxy']
        num_suspicious = int(self.num_rows * 0.01)  # 1% del dataset
        suspicious_indices = random.sample(range(self.num_rows), num_suspicious)
        
        for idx in suspicious_indices:
            df.loc[idx, 'Location'] = random.choice(suspicious_locations)
            df.loc[idx, 'Anomaly_Flag'] = 1
        
        # 4. Malicious bot activity
        num_bots = int(self.num_rows * 0.02)  # 2% del dataset
        bot_indices = random.sample(range(self.num_rows), num_bots)
        
        for idx in bot_indices:
            df.loc[idx, 'User_Agent'] = 'Unknown_Bot'
            df.loc[idx, 'Request_Type'] = random.choice(['POST', 'PUT'])
            df.loc[idx, 'Status_Code'] = random.choice([400, 403, 500])
            df.loc[idx, 'Anomaly_Flag'] = 1
        
        return df
    
    def generate_dataset(self):
        """Genera il dataset completo."""
        # Crea directory di output se non esiste
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Genera dataset base
        df = self.generate_base_data()
        
        # Inietta anomalie
        df = self.inject_anomalies(df)
        
        # Salva il dataset
        output_path = self.output_dir / "new_cybersecurity_data.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset salvato in {output_path}")
        
        # Mostra statistiche
        self.print_statistics(df)
        
        return df
    
    def print_statistics(self, df):
        """Stampa statistiche sul dataset generato."""
        logger.info("\nStatistiche del dataset di test:")
        logger.info(f"Numero totale di record: {len(df)}")
        logger.info(f"Numero di anomalie: {df['Anomaly_Flag'].sum()}")
        logger.info(f"Percentuale anomalie: {df['Anomaly_Flag'].mean()*100:.2f}%")
        logger.info(f"Distribuzione Location:\n{df['Location'].value_counts().head()}")
        logger.info(f"Distribuzione Request_Type:\n{df['Request_Type'].value_counts()}")
        logger.info(f"Distribuzione Status_Code:\n{df['Status_Code'].value_counts()}")

def main():
    """Funzione principale per l'esecuzione."""
    generator = TestDataGenerator()
    generator.generate_dataset()

if __name__ == "__main__":
    main()