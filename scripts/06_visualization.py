import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging

# Configurazione pagina
st.set_page_config(
    page_title="Cybersecurity Analytics",
    page_icon="🔒",
    layout="wide",
)

# Stile CSS personalizzato
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .stat-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #1E1E1E;
        margin: 10px;
    }
    .stMetric div {
        background-color: #2E2E2E;
        padding: 15px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

class AnomalyDashboard:
    def __init__(self):
        self.predictions_dir = Path("output/predictions")
        self.raw_data_path = Path("data/raw/new_cybersecurity_data.csv")
        self.data = None
        self.predictions = None
        self.raw_predictions = {}
        self.color_scheme = {
            'background': '#1E1E1E',
            'text': '#FFFFFF',
            'accent': '#FF4B4B',
            'success': '#00CC96',
            'warning': '#FFA500',
            'grid': '#333333'
        }
        
    def load_data(self):
        """Carica i dati e le predizioni."""
        self.data = pd.read_csv(self.raw_data_path)
        self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])
        
        prediction_files = list(self.predictions_dir.glob("predictions_*.json"))
        latest_prediction = max(prediction_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_prediction) as f:
            self.predictions = json.load(f)
        
        # Salva probabilità e predizioni
        for model_name, preds in self.predictions['predictions'].items():
            if preds['probability'] is not None:
                self.raw_predictions[model_name] = preds['probability']
                self.data[f'prob_{model_name}'] = preds['probability']
            self.data[f'pred_{model_name}'] = preds['class']
        
        self.data['pred_ensemble'] = self.predictions['ensemble']
    
    def update_predictions(self, threshold):
        """Aggiorna le predizioni basate sulla nuova soglia."""
        # Aggiorna predizioni per singoli modelli
        for model_name in self.predictions['predictions'].keys():
            if f'prob_{model_name}' in self.data.columns:
                self.data[f'pred_{model_name}'] = (self.data[f'prob_{model_name}'] > threshold).astype(int)
        
        # Ricalcola ensemble
        pred_columns = [col for col in self.data.columns if col.startswith('pred_') and col != 'pred_ensemble']
        self.data['pred_ensemble'] = (self.data[pred_columns].mean(axis=1) > 0.5).astype(int)
    
    def create_plotly_theme(self):
        """Crea tema personalizzato per i grafici."""
        return {
            'plot_bgcolor': self.color_scheme['background'],
            'paper_bgcolor': self.color_scheme['background'],
            'font': {'color': self.color_scheme['text']},
            'xaxis': {
                'gridcolor': self.color_scheme['grid'],
                'title_font': {'size': 18}
            },
            'yaxis': {
                'gridcolor': self.color_scheme['grid'],
                'title_font': {'size': 18}
            }
        }
    
    def run_dashboard(self):
        """Esegue il dashboard."""
        st.markdown('<p class="big-font">🔒 Dashboard Analisi Anomalie Cybersecurity</p>', unsafe_allow_html=True)
        
        # Slider per soglia
        threshold = st.slider(
            "🎚️ Soglia di Rilevamento Anomalie",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Regola la sensibilità del rilevamento. Valori più bassi rilevano più anomalie."
        )
        
        # Aggiorna predizioni
        self.update_predictions(threshold)
        
        # Statistiche principali
        total_anomalies = int(self.data['pred_ensemble'].sum())
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📝 Totale Records", f"{len(self.data):,}")
        with col2:
            st.metric("⚠️ Anomalie Rilevate", str(total_anomalies))
        with col3:
            st.metric("📈 Percentuale Anomalie", f"{(total_anomalies/len(self.data))*100:.1f}%")
        
        # Tabs principali
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", "⏱️ Analisi Temporale", "🌍 Analisi Spaziale", "🔍 Dettagli Modelli"
        ])
        
        with tab1:
            self._show_overview()
        with tab2:
            self._show_temporal_analysis()
        with tab3:
            self._show_spatial_analysis()
        with tab4:
            self._show_model_details()
    
    def _show_overview(self):
        """Mostra overview delle anomalie."""
        st.markdown('<p class="medium-font">📈 Overview delle Anomalie</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_predictions = {
                model: self.data[f'pred_{model}'].sum() 
                for model in self.predictions['predictions'].keys()
            }
            model_predictions['ensemble'] = self.data['pred_ensemble'].sum()
            
            model_comparison = pd.DataFrame({
                'Modello': list(model_predictions.keys()),
                'Anomalie': list(model_predictions.values())
            })
            
            fig = go.Figure(data=[
                go.Bar(x=model_comparison['Modello'], 
                      y=model_comparison['Anomalie'],
                      marker_color=self.color_scheme['accent'])
            ])
            fig.update_layout(**self.create_plotly_theme())
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            anomaly_status = self.data[self.data['pred_ensemble'] == 1]['Status_Code'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=anomaly_status.index,
                values=anomaly_status.values,
                hole=.3
            )])
            fig.update_layout(**self.create_plotly_theme())
            st.plotly_chart(fig, use_container_width=True)

    def _show_temporal_analysis(self):
        """Mostra analisi temporale."""
        st.markdown('<p class="medium-font">⏰ Analisi Temporale</p>', unsafe_allow_html=True)
        
        hourly_anomalies = self.data.set_index('Timestamp')\
            .resample('H')\
            .agg({'pred_ensemble': 'sum'})
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly_anomalies.index,
            y=hourly_anomalies['pred_ensemble'],
            mode='lines+markers',
            line=dict(color=self.color_scheme['accent'], width=2),
            marker=dict(size=6)
        ))
        fig.update_layout(**self.create_plotly_theme())
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistiche temporali
        col1, col2 = st.columns(2)
        with col1:
            peak_hour = hourly_anomalies['pred_ensemble'].idxmax()
            st.metric("⚡ Ora di Picco", peak_hour.strftime('%H:00'))
        with col2:
            st.metric("🔝 Max Anomalie/Ora", int(hourly_anomalies['pred_ensemble'].max()))

    def _show_spatial_analysis(self):
        """Mostra analisi spaziale."""
        st.markdown('<p class="medium-font">🌍 Analisi Spaziale</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            location_anomalies = self.data[self.data['pred_ensemble'] == 1]['Location'].value_counts()
            fig = go.Figure(data=[
                go.Bar(
                    x=location_anomalies.index,
                    y=location_anomalies.values,
                    marker_color=self.color_scheme['warning']
                )
            ])
            fig.update_layout(**self.create_plotly_theme())
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            suspicious_ips = self.data[self.data['pred_ensemble'] == 1]['IP_Address'].value_counts().head(10)
            fig = go.Figure(data=[
                go.Bar(
                    x=suspicious_ips.values,
                    y=suspicious_ips.index,
                    orientation='h',
                    marker_color=self.color_scheme['warning']
                )
            ])
            fig.update_layout(**self.create_plotly_theme())
            st.plotly_chart(fig, use_container_width=True)

    def _show_model_details(self):
        """Mostra dettagli dei modelli."""
        st.markdown('<p class="medium-font">🔍 Performance dei Modelli</p>', unsafe_allow_html=True)
        
        # Metriche in colonne
        cols = st.columns(len(self.predictions['predictions']) + 1)
        for idx, (model_name, model_pred) in enumerate(self.predictions['predictions'].items()):
            with cols[idx]:
                accuracy = (self.data[f'pred_{model_name}'] == self.data['Anomaly_Flag']).mean()
                st.metric(
                    f"Accuratezza {model_name}",
                    f"{accuracy:.2%}",
                    delta=f"{(accuracy-0.5)*100:+.1f}pp vs random"
                )
        
        # Matrice di correlazione
        st.markdown("### 🔄 Correlazione tra Modelli")
        pred_cols = [col for col in self.data.columns if col.startswith('pred_')]
        corr_matrix = self.data[pred_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=pred_cols,
            y=pred_cols,
            colorscale='RdBu',
            zmid=0
        ))
        fig.update_layout(**self.create_plotly_theme())
        st.plotly_chart(fig, use_container_width=True)

def main():
    try:
        dashboard = AnomalyDashboard()
        dashboard.load_data()
        dashboard.run_dashboard()
    except Exception as e:
        st.error(f"⚠️ Errore: {str(e)}")
        logging.error(f"Errore nell'esecuzione del dashboard: {str(e)}")

if __name__ == "__main__":
    main()