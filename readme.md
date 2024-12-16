# 🛡️ Cyber Anomaly Shield

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Un sistema avanzato di rilevamento anomalie per la sicurezza informatica, che utilizza machine learning per identificare pattern sospetti nel traffico HTTP.


## 📊 Dataset originale

Questo progetto utilizza un dataset originale creato artificialmente per l'analisi della sicurezza informatica e il rilevamento delle anomalie.

### Link al Dataset
Per il progetto è stato utilizzato il dataset che ho sviluppato personalmente è che è pubblico su Kaggle [HTTP Log Anomaly Detection](https://www.kaggle.com/datasets/fcwebdev/synthetic-cybersecurity-logs-for-anomaly-detection) disponibile su Kaggle.
- **Versione**: 1.0
- **Dimensione**: 10,000 record
- **Formato**: CSV

### Caratteristiche del Dataset:

| Feature     | Descrizione                                  | Tipo      |
|-------------|----------------------------------------------|-----------|
| Timestamp   | Data e ora di ogni log                       | Datetime  |
| IP_Address  | Indirizzi IP simulati                        | String    |
| Request_Type| GET, POST, PUT, DELETE                       | String    |
| Status_Code | Codici di risposta HTTP                      | Integer   |
| Anomaly_Flag| Flag per anomalie (1=anomalia, 0=normale)    | Binary    |
| User_Agent  | Identificazione browser/dispositivo          | String    |
| Session_ID  | ID sessione simulati                         | Integer   |
| Location    | Localizzazione geografica                    | String    |

## 🚀 Features dell'Applicazione

### 1. Dashboard Interattivo
- **Monitoraggio Real-time**
  - Metriche chiave di sicurezza
  - Visualizzazioni interattive
  - Trend temporali
- **Analisi Statistica**
  - Distribuzione delle richieste
  - Pattern di traffico
  - Hot-spot di anomalie

### 2. Training del Modello
- **Opzioni Multiple di Modelli**
  - Random Forest
  - Isolation Forest
  - Modello Ibrido
- **Configurazione Parametri**
  - Dimensione test set
  - Iperparametri
  - Bilanciamento classi
- **Metriche di Performance**
  - Accuracy, Precision, Recall
  - Matrice di confusione
  - Curve ROC/PR

### 3. Rilevamento Live
- **Analisi in Tempo Reale**
  - Valutazione immediata
  - Score di rischio
  - Alert configurabili
- **Visualizzazione Risultati**
  - Grafici interattivi
  - Indicatori di rischio
  - Dettagli anomalie

## 🛠️ Installazione

### Prerequisiti
- Python 3.9+
- pip
- Git

### Setup

1. Clone del repository:
```bash
git clone [url-repository]
cd cybersecurity-anomaly-detector
```

2. Creazione dell'ambiente virtuale:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Installazione dipendenze:
```bash
pip install -r requirements.txt
```

## 💻 Utilizzo

### Avvio Applicazione
```bash
streamlit run streamlit_app/app.py
```

### Esempi di Utilizzo

1. **Dashboard**
   - Visualizza metriche in tempo reale
   - Analizza trend temporali
   - Esporta report

2. **Training**
   - Seleziona modello
   - Configura parametri
   - Valuta performance

3. **Detection**
   - Inserisci parametri richiesta
   - Analizza risultati
   - Configura alert

## 📁 Struttura del Progetto

```
cybersecurity-anomaly-detector/
├── data/
│   ├── raw/                  # Dataset originali
│   └── processed/            # Dataset processati
├── output/
│   ├── models/              # Modelli salvati
│   ├── plots/              # Grafici generati
│   └── metrics/            # Metriche di performance
├── scripts/
│   ├── 01_generate_dataset.py
│   ├── 02_data_exploration.py
│   └── ...
├── src/
│   ├── features/           # Feature engineering
│   ├── models/            # Implementazioni modelli
│   └── utils/             # Utility functions
├── streamlit_app/
│   ├── app.py            # Main app
│   └── components/       # Componenti UI
├── static/
│   ├── css/
│   └── images/
├── tests/                # Unit tests
├── requirements.txt
└── README.md
```

## 🤝 Contributing

Le contribuzioni sono benvenute! Per contribuire:

1. Fork del repository
2. Crea un branch feature (`git checkout -b feature/AmazingFeature`)
3. Commit dei cambiamenti (`git commit -m 'Add AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Apri una Pull Request

### Linee Guida
- Segui PEP 8
- Aggiungi test per nuove feature
- Aggiorna la documentazione
- Mantieni il codice pulito e commentato

## 📄 License

Distribuito sotto licenza MIT. Vedi `LICENSE.md` per maggiori informazioni.

## ✨ Acknowledgments

- Dataset creato e messo a disposizione su Kaggle
- Ispirato da reali log di sicurezza HTTP
- Grazie alla community di sicurezza informatica

## 📧 Contatti

[Il tuo Nome] - [@tuotwitter](https://twitter.com/tuotwitter) - email@example.com

Project Link: [https://github.com/username/repo](https://github.com/username/repo)

## 📚 Documentazione Aggiuntiva

- [Guida Utente](docs/user_guide.md)
- [API Reference](docs/api_reference.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)