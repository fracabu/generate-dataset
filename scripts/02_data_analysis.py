import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

def load_dataset():
    """Carica il dataset da data/raw"""
    data_path = Path("data/raw/advanced_cybersecurity_data.csv")
    return pd.read_csv(data_path)

def analyze_temporal_patterns(df):
    """Analizza i pattern temporali"""
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    
    # Analisi oraria
    hourly_stats = df.groupby('Hour')['Anomaly_Flag'].agg(['count', 'mean'])
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    hourly_stats['count'].plot(kind='bar', ax=ax1, title='Requests per Hour')
    hourly_stats['mean'].plot(kind='bar', ax=ax2, title='Anomaly Rate per Hour')
    plt.tight_layout()
    plt.savefig('output/plots/temporal_patterns.png')
    plt.close()
    
    return hourly_stats

def analyze_ip_patterns(df):
    """Analizza i pattern degli IP"""
    ip_stats = df.groupby('IP_Address').agg({
        'Anomaly_Flag': ['count', 'mean'],
        'Location': 'nunique',
        'User_Agent': 'nunique'
    }).round(3)
    
    ip_stats.columns = ['Request_Count', 'Anomaly_Rate', 'Unique_Locations', 'Unique_UserAgents']
    
    # Salva statistiche
    ip_stats.to_csv('output/metrics/ip_statistics.csv')
    
    return ip_stats

def analyze_request_patterns(df):
    """Analizza i pattern delle richieste"""
    request_stats = df.groupby(['Request_Type', 'Status_Code']).agg({
        'Anomaly_Flag': ['count', 'mean']
    }).round(3)
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Request_Type', y='Anomaly_Flag')
    plt.title('Anomaly Rate by Request Type')
    plt.savefig('output/plots/request_patterns.png')
    plt.close()
    
    return request_stats

def generate_report(df, temporal_stats, ip_stats, request_stats):
    """Genera un report di analisi"""
    with open('output/metrics/analysis_report.md', 'w') as f:
        f.write("# Cybersecurity Data Analysis Report\n\n")
        
        f.write("## Dataset Overview\n")
        f.write(f"- Total Records: {len(df):,}\n")
        f.write(f"- Unique IPs: {df['IP_Address'].nunique():,}\n")
        f.write(f"- Overall Anomaly Rate: {df['Anomaly_Flag'].mean()*100:.2f}%\n\n")
        
        f.write("## Temporal Analysis\n")
        f.write("### Hourly Statistics\n")
        f.write(temporal_stats.to_markdown())
        f.write("\n\n")
        
        f.write("## IP Analysis\n")
        f.write("### Top 10 IPs by Anomaly Rate\n")
        f.write(ip_stats.sort_values('Anomaly_Rate', ascending=False).head(10).to_markdown())
        f.write("\n\n")
        
        f.write("## Request Pattern Analysis\n")
        f.write(request_stats.to_markdown())

def main():
    print("Starting data analysis...")
    
    # Crea cartelle output se non esistono
    Path("output/plots").mkdir(parents=True, exist_ok=True)
    Path("output/metrics").mkdir(parents=True, exist_ok=True)
    
    # Carica dati
    df = load_dataset()
    print("Dataset loaded successfully.")
    
    # Esegui analisi
    temporal_stats = analyze_temporal_patterns(df)
    print("Temporal analysis completed.")
    
    ip_stats = analyze_ip_patterns(df)
    print("IP analysis completed.")
    
    request_stats = analyze_request_patterns(df)
    print("Request pattern analysis completed.")
    
    # Genera report
    generate_report(df, temporal_stats, ip_stats, request_stats)
    print("Report generated successfully.")
    
    print("\nAnalysis complete. Check output folder for results.")

if __name__ == "__main__":
    main()