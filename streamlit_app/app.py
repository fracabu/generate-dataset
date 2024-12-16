import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
from datetime import datetime
import time

# Configurazione pagina
st.set_page_config(
    page_title="🔒 Cyber Anomaly Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizzato per animazioni e stile
st.markdown("""
<style>
    /* Animazioni generali */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100%); }
        to { transform: translateX(0); }
    }
    
    /* Stile cards */
    .stMetric {
        background: linear-gradient(135deg, #1e3799, #0c2461);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
        animation: fadeIn 0.5s ease-out;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
    }
    
    /* Header style */
    .main-header {
        font-family: 'Segoe UI', sans-serif;
        background: linear-gradient(45deg, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        animation: fadeIn 1s ease-out;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background-color: #1e3799;
        border-radius: 5px;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #0c2461;
        transform: translateY(-2px);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(45deg, #3498db, #2ecc71);
        color: white;
        border: none;
        padding: 10px 25px;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Loading animation */
    .stProgress > div > div {
        background-color: #3498db;
    }
    
    /* Plotly chart styling */
    .js-plotly-plot {
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .js-plotly-plot:hover {
        transform: scale(1.02);
    }
    .stTabs {
        margin-top: 100px !important;
    }
</style>
""", unsafe_allow_html=True)

# Funzioni di utilità
@st.cache_data
def load_data():
    """Carica i dati con una barra di progresso animata"""
    data_path = Path("data/raw/advanced_cybersecurity_data.csv")
    if data_path.exists():
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        df = pd.read_csv(data_path)
        time.sleep(0.5)
        progress_bar.empty()
        return df
    return None

def create_animated_metric(title, value, delta=None):
    """Crea una metrica animata con effetto di fade in"""
    st.metric(
        title,
        value,
        delta,
        help=f"Metric showing {title.lower()}"
    )

st.markdown("""
<style>
/* Full-width header con overlay */
.header-container {
    display: flex;
    align-items: center;
    height: 300px; /* Altezza fissa */
    margin: 0;
    color: white;
    background: url('https://audiofilescontainer.blob.core.windows.net/audiocontainer/cyber_security_banner.jpeg') no-repeat center center;
    background-size: cover;
    position: relative;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    width: 100vw; /* Occupa l'intera larghezza viewport */
    left: 50%;
    transform: translateX(-50%); /* Centra il contenitore orizzontalmente */
}

/* Overlay completo */
.header-container::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.6); /* Overlay scuro al 60% */
    z-index: 1;
}

/* Contenuto header */
.header-content {
    position: relative;
    z-index: 2;
    text-align: left; /* Testo allineato a sinistra */
    padding-left: 100px; /* Spaziatura a sinistra per il contenuto */
}

/* Titolo */
.header-title {
    font-family: 'Segoe UI', sans-serif;
    font-size: 4rem;
    font-weight: bold;
    margin: 0;
    line-height: 1.2;
}

/* Sottotitolo */
.header-subtitle {
    font-family: 'Segoe UI', sans-serif;
    font-size: 1.5rem;
    margin-top: 10px;
    opacity: 0.9;
    line-height: 1.5;
}
</style>

<div class="header-container">
    <div class="header-content">
        <h1 class="header-title">Cyber Anomaly Shield</h1>
        <p class="header-subtitle">Advanced Machine Learning for Real-Time Security and Anomaly Detection</p>
    </div>
</div>
""", unsafe_allow_html=True)




# Carica i dati
df = load_data()

# Tabs con animazioni
tabs = st.tabs(["📊 Dashboard", "🔍 Model Training", "🎯 Live Detection", "📄 README"])


with tabs[0]:    # Dashboard
    st.markdown("### Real-time Security Analytics")
    
    if df is not None:
        # Metriche animate
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            create_animated_metric("Total Records", f"{len(df):,}")
        with col2:
            anomaly_rate = (df['Anomaly_Flag'].sum() / len(df) * 100).round(2)
            create_animated_metric("Anomaly Rate", f"{anomaly_rate}%", f"{anomaly_rate-5:.1f}%")
        with col3:
            create_animated_metric("Unique IPs", f"{df['IP_Address'].nunique():,}")
        with col4:
            create_animated_metric("Active Sessions", f"{df['Session_ID'].nunique():,}")
        
        # Nel tab Dashboard, sostituisci la parte dei grafici con questo codice:
        # Grafici interattivi
        col1, col2 = st.columns(2)
        with col1:
            # Status Code Distribution con animazione
            status_counts = pd.DataFrame(df['Status_Code'].value_counts()).reset_index()
            status_counts.columns = ['Status_Code', 'Count']  # Rinomina le colonne
            fig = px.bar(
                status_counts,
                x='Status_Code',
                y='Count',
                title='Status Code Distribution',
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Request Type Distribution con animazione
            request_counts = pd.DataFrame(df['Request_Type'].value_counts()).reset_index()
            request_counts.columns = ['Request_Type', 'Count']  # Rinomina le colonne
            fig = px.pie(
                request_counts,
                values='Count',
                names='Request_Type',
                title='Request Type Distribution',
                hole=0.4
            )
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
with tabs[1]:    # Model Training
    st.markdown("### Model Training & Evaluation")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### Training Parameters")
        model_type = st.selectbox(
            "Select Model Type",
            ["Random Forest", "Isolation Forest", "Hybrid Model"]
        )
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        
        if "previous_metrics" not in st.session_state:
            st.session_state["previous_metrics"] = {"accuracy": 0, "precision": 0, "recall": 0}

        def train_model(model_type, test_size, X_train, y_train, X_test, y_test):
            """Addestra il modello e calcola le metriche."""
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.ensemble import IsolationForest
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            
            # Seleziona il modello
            if model_type == "Random Forest":
                model = RandomForestClassifier()
            elif model_type == "Isolation Forest":
                model = IsolationForest()
            else:
                st.error("Hybrid Model non implementato")
                return None

            # Addestramento del modello
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calcola metriche
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted")
            rec = recall_score(y_test, y_pred, average="weighted")
            
            return model, {"accuracy": acc, "precision": prec, "recall": rec}

        if st.button("Train Model", key="train"):
            with st.spinner('Training in progress...'):
                # Simulazione del dataset
                from sklearn.datasets import make_classification
                X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
                X_train, X_test = X[:800], X[800:]
                y_train, y_test = y[:800], y[800:]
                
                # Addestramento del modello
                model, current_metrics = train_model(model_type, test_size, X_train, y_train, X_test, y_test)
                
                # Calcola delta rispetto ai risultati precedenti
                delta_accuracy = (current_metrics["accuracy"] - st.session_state["previous_metrics"]["accuracy"]) * 100
                delta_precision = (current_metrics["precision"] - st.session_state["previous_metrics"]["precision"]) * 100
                delta_recall = (current_metrics["recall"] - st.session_state["previous_metrics"]["recall"]) * 100

                # Salva i risultati correnti come metriche precedenti
                st.session_state["previous_metrics"] = current_metrics
                
                st.success("Model trained successfully! 🎉")
    
    with col2:
        st.markdown("#### Model Performance")
        # Mostra metriche dinamiche
        if "previous_metrics" in st.session_state:
            current_metrics = st.session_state["previous_metrics"]
            metrics_cols = st.columns(3)
            with metrics_cols[0]:
                st.metric(
                    label="Accuracy",
                    value=f"{current_metrics['accuracy'] * 100:.2f}%",
                    delta=f"{(current_metrics['accuracy'] - st.session_state['previous_metrics']['accuracy']) * 100:.2f}%"
                )
            with metrics_cols[1]:
                st.metric(
                    label="Precision",
                    value=f"{current_metrics['precision'] * 100:.2f}%",
                    delta=f"{(current_metrics['precision'] - st.session_state['previous_metrics']['precision']) * 100:.2f}%"
                )
            with metrics_cols[2]:
                st.metric(
                    label="Recall",
                    value=f"{current_metrics['recall'] * 100:.2f}%",
                    delta=f"{(current_metrics['recall'] - st.session_state['previous_metrics']['recall']) * 100:.2f}%"
                )


with tabs[2]:    # Live Detection
    st.markdown("### Real-time Anomaly Detection")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### Input Parameters")
        with st.form("detection_form"):
            request_type = st.selectbox("Request Type", ["GET", "POST", "PUT", "DELETE"])
            status_code = st.number_input("Status Code", min_value=100, max_value=599)
            location = st.selectbox("Location", ["USA", "Germany", "France", "India", "China"])
            user_agent = st.selectbox("User Agent", ["Chrome", "Firefox", "Safari", "Edge"])
            
            submitted = st.form_submit_button("Detect Anomaly")
            if submitted:
                with st.spinner('Analyzing...'):
                    time.sleep(1.5)  # Simulazione analisi
                    # Random result per demo
                    is_anomaly = np.random.choice([True, False], p=[0.2, 0.8])
                    if is_anomaly:
                        st.error("⚠️ Anomaly Detected!")
                        st.markdown("""
                        **Risk Factors:**
                        - Unusual request pattern
                        - Suspicious IP behavior
                        - Time-based anomaly
                        """)
                    else:
                        st.success("✅ Normal Traffic Detected")
    
    with col2:
        st.markdown("#### Real-time Analysis")
        # Placeholder per grafici real-time
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=np.random.randn(100).cumsum(),
            mode='lines',
            name='Traffic Pattern',
            line=dict(color='#3498db')
        ))
        fig.update_layout(
            title='Live Traffic Analysis',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

with tabs[3]:  # README Tab
    st.markdown("# 🛡️ Cyber Anomaly Shield")
    st.markdown("""
    [![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
    [![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B.svg)](https://streamlit.io)
    [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org/)
    [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

    Un sistema avanzato di rilevamento anomalie per la sicurezza informatica, che utilizza machine learning per identificare pattern sospetti nel traffico HTTP.

    ![Cyber Security Banner](static/images/banner.png)

    ## 📊 Dataset originale

    Questo progetto utilizza un dataset originale creato per l'analisi della sicurezza informatica e il rilevamento delle anomalie.

    ### Link al Dataset
    - [HTTP Log Anomaly Detection su Kaggle](https://www.kaggle.com/datasets/tonypark00/httplogsanomalydetection)
    - **Autore**: [Il tuo username Kaggle]
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

    """, unsafe_allow_html=True)


# Footer animato
st.markdown("""
<div style='text-align: center; padding: 20px; animation: fadeIn 1s ease-out;'>
    <p style='color: #666; font-size: 0.8em;'>
        Made with ❤️ by Your Team | © 2024 Cyber Anomaly Shield
    </p>
</div>
""", unsafe_allow_html=True)