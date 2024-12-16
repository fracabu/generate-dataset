import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Numero di righe del dataset
num_rows = 10000

# Genera timestamp simulati
start_time = datetime(2023, 1, 1, 0, 0, 0)
timestamps = [start_time + timedelta(minutes=i) for i in range(num_rows)]

# Genera indirizzi IP casuali
def generate_ip():
    return f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"

ip_addresses = [generate_ip() for _ in range(num_rows)]

# Tipi di richieste HTTP
request_types = ['GET', 'POST', 'PUT', 'DELETE']

# Genera richieste casuali
requests = [random.choice(request_types) for _ in range(num_rows)]

# Genera codici di stato HTTP
status_codes = [random.choice([200, 404, 500, 403, 301]) for _ in range(num_rows)]

# Genera flag di anomalia
anomaly_flags = [1 if random.random() < 0.05 else 0 for _ in range(num_rows)]

# Colonne aggiuntive
user_agents = ['Chrome', 'Firefox', 'Safari', 'Edge', 'Opera', 'Bot']
session_ids = [random.randint(1000, 5000) for _ in range(num_rows)]
locations = ['USA', 'Germany', 'France', 'India', 'China', 'Brazil', 'Canada']

# Crea il dataset
data = {
    'Timestamp': timestamps,
    'IP_Address': ip_addresses,
    'Request_Type': requests,
    'Status_Code': status_codes,
    'Anomaly_Flag': anomaly_flags,
    'User_Agent': [random.choice(user_agents) for _ in range(num_rows)],
    'Session_ID': session_ids,
    'Location': [random.choice(locations) for _ in range(num_rows)]
}

cyber_data = pd.DataFrame(data)

# Introduci attacchi DoS
suspicious_ip = generate_ip()
for i in range(50):
    cyber_data.loc[random.randint(0, num_rows-1), 'IP_Address'] = suspicious_ip

# Introduci anomalie geo
anomalous_location = 'North Korea'
for i in range(10):
    idx = random.randint(0, num_rows-1)
    cyber_data.loc[idx, 'Location'] = anomalous_location
    cyber_data.loc[idx, 'Anomaly_Flag'] = 1

# Esporta il dataset
cyber_data.to_csv('advanced_cybersecurity_data.csv', index=False)

# Mostra le prime righe
print(cyber_data.head())
