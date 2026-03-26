"""
===========================================================================
CART-Prognosemodell und Vergleich zu Baseline
===========================================================================
Autor:   Sami Stephan

Datum:   25.03.2026

Ziel:    Trainiert CART-Modelle pro PSP und führt sequenzielle
         Business Value Evaluation durch.
         
Input:   PSP_Jan_Feb_2019.xlsx   (Rohdaten)
         
Output:  cart_analyse_{timestamp}.txt
         cart_eda_plots.png
         cart_eda_korrelationsmatrix.png
         cart_psp_data_with_features.csv
         cart_hyperparameter_tuning_{psp_name}.png
         cart_tree_{psp_name}.png
         cart_evaluation_psp_recommender_cal.png
         cart_sensitivity_opport.png
         cart_best_params_per_psp.pkl
         cart_psp_recommender_cal.pkl
         cart_hyperparameter_results_{psp_name}.csv
         cart_hyperparameter_results.csv
         cart_results_psp_recommender_cal.csv
         
===========================================================================
"""

import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, cross_val_score
from itertools import product
from scipy import stats
import pickle
import sys
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# KONFIGURATION
# =============================================================================

# Seed einmalig setzen
RANDOM_STATE = 13
rng = np.random.default_rng(seed=RANDOM_STATE) 

# Anteil Testdatensatz am Gesamtdatensatz
TEST_SIZE = 0.2

# Kalibrierung nach CART-Analyse
CALIB_METH = 'isotonic' # Methoden: 'isotonic', 'sigmoid'

# Liste der PSP
PSP_LIST = ['Moneycard', 'Goldcard', 'UK_Card', 'Simplecard']
# mittlerer Warenkorbwert aus Daten
AVERAGE_BASKET_VALUE = 200.0  
# Gewinnmarge als Anteil am Warenkorbwert festlegen
PROFIT_MARGIN        = 0.05
# Daraus berechneter mittlerer Gewinn bei erfolgreicher Transaktion
REVENUE_PER_SUCCESS  = AVERAGE_BASKET_VALUE * PROFIT_MARGIN
# Opportunitätskosten für Fehlschlag als Anteil am Warenkorbwert
C_OPPORT             = 0.04
# aus Baseline-Daten ermittele maximale Anzahl an Versuchen
MAX_ATTEMPTS         = 10     
# PSP-Gebühren
FEES = {
    'Moneycard':  {'success':  5.0, 'failure': 2.0},
    'Goldcard':   {'success': 10.0, 'failure': 5.0},
    'UK_Card':    {'success':  3.0, 'failure': 1.0},
    'Simplecard': {'success':  1.0, 'failure': 0.5}
}

# Parameter-Grid für Tuning
PARAM_GRID = {
    'max_depth': [3, 4, 5, 6],
    'min_samples_leaf': [20, 40, 80, 160],
    'min_samples_split': [50, 100, 200, 400],
    'class_weight': ['balanced']
}

# Cross-Validation
CV_FOLDS = 5

# =============================================================================
# SETUP
# =============================================================================

BASE_DIR = r"C:\Users\samis\OneDrive - IU International University of Applied Sciences\Dokumente\Master IU Data Science\Model Engineering\Code"

# NEUE ORDNERSTRUKTUR
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_FIG_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
REPORTS_LOG_DIR = os.path.join(BASE_DIR, 'reports', 'logs')
REPORTS_RES_DIR = os.path.join(BASE_DIR, 'reports', 'results')

# Verzeichnisse anlegen, falls noch nicht vorhanden
dirs = [BASE_DIR, DATA_RAW_DIR, DATA_PROC_DIR, MODELS_DIR, REPORTS_FIG_DIR, REPORTS_LOG_DIR, REPORTS_RES_DIR]
for d in dirs:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# LOGGING
# =============================================================================

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = os.path.join(REPORTS_LOG_DIR, f'cart_analyse_{timestamp}.txt')

class Tee:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

tee = Tee(log_filename)
sys.stdout = tee

# Konfiguration für bessere Lesbarkeit
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 140)
plt.rcParams['figure.figsize'] = (20, 24)
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

# =============================================================================
# 1. DATEN LADEN
# =============================================================================

print("\n" + "=" * 100)
print("1. DATEN LADEN")
print("=" * 100)

df = pd.read_excel(os.path.join(DATA_RAW_DIR,'PSP_Jan_Feb_2019.xlsx'))
df = df.sort_values('tmsp').reset_index(drop=True)

print(f"\nDatensatz geladen: {len(df)} Zeilen, {len(df.columns)} Spalten")

# =============================================================================
# 2. EXPLORATIVE DATENANALYSE (EDA)
# =============================================================================

print("\n" + "=" * 100)
print("2. EXPLORATIVE DATENANALYSE (EDA)")
print("=" * 100)

# =============================================================================
# 2.1 DATENQUALITÄT
# =============================================================================

print("\n" + "=" * 100)
print("2.1 DATENQUALITÄT")
print("=" * 100)

print(f"\nZeitraum:")
print(f"  {df['tmsp'].min()} bis {df['tmsp'].max()}")
print(f"\nDauer:")
print(f"  {(df['tmsp'].max() - df['tmsp'].min()).days} Tage")

# Prüfe auf fehlende Werte
missing_summary = pd.DataFrame({
    'Anzahl_Missing': df.isnull().sum(),
    'Prozent_Missing': (df.isnull().sum() / len(df) * 100).round(2),
    'Datentyp': df.dtypes
})

print("\nFehlende Werte pro Spalte:")
print(missing_summary)

# =============================================================================
# 2.2 DUPLIKATE
# =============================================================================

print("\n" + "=" * 100)
print("2.2 DUPLIKATE")
print("=" * 100)

# Exakte Duplikate
n_duplicates = df.duplicated().sum()
print(f"\nExakte Duplikate (alle Spalten identisch): {n_duplicates}")

# Purchase Groups ermitteln (Kaufversuche innerhalb 10 min Zeitfenster)
df['transaction_key'] = (
    df['amount'].round(2).astype(str) + '_' +
    df['country'] + '_' + df['card']
)
df = df.sort_values(['transaction_key','tmsp']).reset_index(drop=True)

df['new_group'] = (
    df.groupby('transaction_key')['tmsp']
      .diff()
      .gt(pd.Timedelta(minutes=10))
      .fillna(True)
)

df['group_id'] = df.groupby('transaction_key')['new_group'].cumsum()

df['purchase_group'] = (
    df['transaction_key'] + '_' +
    df.groupby(['transaction_key','group_id'])['tmsp']
      .transform('min')
      .dt.strftime('%Y%m%d_%H%M%S')
)

group_sizes = df.groupby('purchase_group').size()
n_multi_attempts = (group_sizes > 1).sum()
n_single_attempts = (group_sizes == 1).sum()

print(f"\nKaufversuche (id. Betrag, Land, Karte, innerhalb 10 min): {len(group_sizes):,}")
print(f"  - Einzelversuche:   {n_single_attempts:,} ({n_single_attempts/len(group_sizes)*100:.1f}%)")
print(f"  - Mehrfachversuche: {n_multi_attempts:,} ({n_multi_attempts/len(group_sizes)*100:.1f}%)")

# =============================================================================
# 2.3 DATENTYPEN UND WERTEBEREICHE
# =============================================================================

print("\n" + "=" * 100)
print("2.3 DATENTYPEN UND WERTEBEREICHE")
print("=" * 100)

print("\nDatentypen inkl. Kaufversuch (purchase_group) und Hilfsvariablen:")
print(df.dtypes)

print("\n--- KATEGORIELLE VARIABLEN ---")
categorical_cols = ['country', 'PSP', 'card']
for col in categorical_cols:
    unique_vals = df[col].unique()
    print(f"\n{col}: {len(unique_vals)} Werte")
    print(f"  Werte: {sorted(unique_vals)}")
    print(f"  Verteilung: {df[col].value_counts().values}")

print("\n--- BINÄRE VARIABLEN ---")
binary_cols = ['success', '3D_secured']
for col in binary_cols:
    print(f"\n{df[col].value_counts().to_string()}")

print("\n--- NUMERISCHE VARIABLE ---")

print( "\namount (Betrag):")
print(f"  Min: {df['amount'].min()}")
print(f"  Max: {df['amount'].max()}")
print(f"  Median: {df['amount'].median()}")
print(f"  Mean: {df['amount'].mean():.2f}")
print(f"  Std: {df['amount'].std():.2f}")

print("\nQuantile (Betrag):")
quantiles = df['amount'].quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
for q, val in quantiles.items():
    print(f"  {q*100:>5.0f}%: {val:>7.0f} €")

# =============================================================================
# 2.4 ERFOLGSRATE AUF TRANSAKTIONSEBENE NACH KATEGORIEN
# =============================================================================

print("\n" + "=" * 100)
print("2.4 ERFOLGSRATEN AUF TRANSAKTIONSEBENE NACH KATEGORIEN")
print("=" * 100)

print("\n--- Nach PSP ---")
psp_stats = df.groupby('PSP').agg({
    'success': ['count', 'sum', 'mean']
}).round(4)
psp_stats.columns = ['Anzahl', 'Erfolge', 'Erfolgsrate']
psp_stats = psp_stats.sort_values('Erfolgsrate', ascending=False)
print(psp_stats)

print("\n--- Nach Land ---")
country_stats = df.groupby('country').agg({
    'success': ['count', 'sum', 'mean']
}).round(4)
country_stats.columns = ['Anzahl', 'Erfolge', 'Erfolgsrate']
print(country_stats)

print("\n--- Nach Kartentyp ---")
card_stats = df.groupby('card').agg({
    'success': ['count', 'sum', 'mean']
}).round(4)
card_stats.columns = ['Anzahl', 'Erfolge', 'Erfolgsrate']
print(card_stats)

print("\n--- Nach 3D-Secure Status ---")
secure_stats = df.groupby('3D_secured').agg({
    'success': ['count', 'sum', 'mean']
}).round(4)
secure_stats.columns = ['Anzahl', 'Erfolge', 'Erfolgsrate']
secure_stats.index = ['Nicht gesichert', 'Gesichert']
print(secure_stats)

# =============================================================================
# 2.5 ZEITLICHE MUSTER AUF TRANSAKTIONSEBENE
# =============================================================================

print("\n" + "=" * 100)
print("2.5 ZEITLICHE MUSTER AUF TRANSAKTIONSEBENE")
print("=" * 100)

# Extrahiere Zeitkomponenten
df['hour'] = df['tmsp'].dt.hour
df['day_of_week'] = df['tmsp'].dt.dayofweek  # 0=Montag, 6=Sonntag
df['day_of_month'] = df['tmsp'].dt.day
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

print("\n--- Nach Stunde ---")
hour_stats = df.groupby('hour').agg({
    'success': ['count', 'mean']
}).round(4)
hour_stats.columns = ['Anzahl', 'Erfolgsrate']
print(hour_stats.sort_values('Erfolgsrate', ascending=False).head(24))

print("\n--- Nach Wochentag ---")
weekday_names = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag']
weekday_stats = df.groupby('day_of_week').agg({
    'success': ['count', 'mean']
}).round(4)
weekday_stats.columns = ['Anzahl', 'Erfolgsrate']
weekday_stats.index = weekday_names
print(weekday_stats)

print("\n--- Wochenende vs. Wochentag ---")
weekend_stats = df.groupby('is_weekend').agg({
    'success': ['count', 'mean']
}).round(4)
weekend_stats.columns = ['Anzahl', 'Erfolgsrate']
weekend_stats.index = ['Wochentag', 'Wochenende']
print(weekend_stats)

# =============================================================================
# 2.6 INTERAKTIONSEFFEKTE AUF TRANSAKTIONSEBENE
# =============================================================================

print("\n" + "=" * 100)
print("2.6 INTERAKTIONSEFFEKTE AUF TRANSAKTIONSEBENE")
print("=" * 100)

print("\n--- PSP × 3D_secured ---")
interaction_psp_3d = (
    df
    .groupby(['PSP', '3D_secured'])['success']
    .agg(Anzahl='count', Erfolgsrate='mean')
    .round(4)
)
print(interaction_psp_3d.to_string())

print("\n--- PSP × Kartentyp ---")
interaction_psp_card = (
    df
    .groupby(['PSP', 'card'])['success']
    .agg(Anzahl='count', Erfolgsrate='mean')
    .round(4)
)
print(interaction_psp_card)

print("\n--- PSP × Land ---")
interaction_psp_country = (
    df
    .groupby(['PSP', 'country'])['success'] 
    .agg(Anzahl='count', Erfolgsrate='mean')
    .round(4)
)
print(interaction_psp_country)

# =============================================================================
# 2.7 MERKMALSVARIABLEN
# =============================================================================

print("\n" + "=" * 100)
print("2.7 MERKMALSVARIABLEN")
print("=" * 100)

feature_names = [
    'amount', '3D_secured', 'hour', 'day_of_week', 'day_of_month', 'is_weekend', 
    'is_night', 'is_morning', 'is_afternoon', 'is_evening',
    'in_germany', 'in_austria', 'in_switzerland', 
    'is_visa', 'is_master', 'is_diners','n_previous_attempts', 'n_previous_failures',
    'tried_Moneycard', 'tried_Goldcard', 'tried_UK_Card', 'tried_Simplecard']

print(f"\nAnzahl Features: {len(feature_names)}")

print(f"\n--- Originale Features (aus Rohdaten) ---")
print(f"  {'amount':30s} Transaktionsbetrag (€)")
print(f"  {'3D_secured':30s} 3D-Secure aktiviert (0/1)")

print(f"\n--- Zeitfeatures (abgeleitet aus tmsp) ---")
print(f"  {'hour':30s} Stunde (0-23)")
print(f"  {'day_of_week':30s} Wochentag (0=Mo, 6=So)")
print(f"  {'day_of_month':30s} Tag im Monat (1-31)")

print(f"\n--- One-Hot-Kodierung Tag/Zeit (abgeleitet aus tmsp) ---")
print(f"  {'is_weekend':30s} Wochenende (Sa/So = 1)")
print(f"  {'is_night':30s} Nacht (0-5 Uhr = 1)")
print(f"  {'is_morning':30s} Morgen (6-11 Uhr = 1)")
print(f"  {'is_afternoon':30s} Nachmittag (12-17 Uhr = 1)")
print(f"  {'is_evening':30s} Abend (18-23 Uhr = 1)")

print(f"\n--- One-Hot-Kodierung Länder (abgeleitet aus country) ---")
print(f"  {'in_germany':30s} Land = Germany (1/0)")
print(f"  {'in_austria':30s} Land = Austria (1/0)")
print(f"  {'in_switzerland':30s} Land = Switzerland (1/0)")

print(f"\n--- One-Hot-Kodierung Karten (abgeleitet aus card) ---")
print(f"  {'is_visa':30s} Karte = Visa (1/0)")
print(f"  {'is_master':30s} Karte = Master (1/0)")
print(f"  {'is_diners':30s} Karte = Diners (1/0)")

print(f"\n--- Sequenz-Features (abgeleitet aus purchase_group) ---")
print(f"  {'n_previous_attempts':30s} Anzahl bisheriger Versuche in dieser Gruppe")
print(f"  {'n_previous_failures':30s} Anzahl bisheriger Fehlschläge in dieser Gruppe")
print(f"  {'tried_<PSP>':30s} Anzahl bisheriger Versuche bei jeweiligem PSP")

# Purchase Groups bereits in 2.2 berechnet
# Time Features bereits in 2.5 berechnet

df['time_period'] = pd.cut(
    df['hour'], bins=[0, 6, 12, 18, 24],
    labels=['night', 'morning', 'afternoon', 'evening'],
    include_lowest=True)

df['is_night'] = (df['time_period'] == 'night').astype(int)
df['is_morning'] = (df['time_period'] == 'morning').astype(int)
df['is_afternoon'] = (df['time_period'] == 'afternoon').astype(int)
df['is_evening'] = (df['time_period'] == 'evening').astype(int)

# Country Features
df['in_germany'] = (df['country'] == 'Germany').astype(int)
df['in_austria'] = (df['country'] == 'Austria').astype(int)
df['in_switzerland'] = (df['country'] == 'Switzerland').astype(int)

# Card Features
df['is_visa'] = (df['card'] == 'Visa').astype(int)
df['is_master'] = (df['card'] == 'Master').astype(int)
df['is_diners'] = (df['card'] == 'Diners').astype(int)

# Sequenz-Features
df['n_previous_attempts'] = df.groupby('purchase_group').cumcount()
df['n_previous_failures'] = df.groupby('purchase_group')['success'].apply(
    lambda x: (1 - x).cumsum().shift(1, fill_value=0)
).values

for psp in PSP_LIST:
    df[f'tried_{psp}'] = df.groupby('purchase_group')['PSP'].apply(
        lambda x: (x.shift(1) == psp).cumsum()
    ).values

# Speichere verarbeiteten Datensatz
df.to_csv(os.path.join(DATA_PROC_DIR,'cart_psp_data_with_features.csv'), index=False, sep=';', decimal=',')
print("\nDatensatz mit Features gespeichert: psp_data_with_features.csv")

# =============================================================================
# 2.8 EDA GRAFIKEN AUF TRANSAKTIONSEBENE
# =============================================================================

print("\n" + "=" * 100)
print("2.8 EDA GRAFIKEN AUF TRANSAKTIONSEBENE")
print("=" * 100)

# Erstelle große Figure mit Subplots
fig = plt.figure(figsize=(20, 28))
gs = fig.add_gridspec(6, 3, hspace=0.5, wspace=0.5)

#  Korrelationsmatrix
plt.figure(figsize=(10, 6))

numeric_cols = ['success', 'amount', '3D_secured', 'hour', 'is_afternoon', 
                'day_of_week', 'is_weekend', 'day_of_month', 
                'is_visa', 'is_master', 'is_diners', 
                'in_germany', 'in_austria', 'in_switzerland']
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})

plt.tight_layout()
plot_path_corr = os.path.join(REPORTS_FIG_DIR, 'cart_eda_korrelationsmatrix.png')
plt.savefig(plot_path_corr, dpi=300, bbox_inches='tight')
print(f"\nKorrelationsmatrix: {plot_path_corr}")
plt.close()

# =============================================================================
# 2.8.1 ERFOLGSRATEN NACH KATEGORIEN
# =============================================================================

# 1. Transaktionsvolumen pro PSP
ax1 = fig.add_subplot(gs[0, 0])
psp_counts = df['PSP'].value_counts()
psp_counts.plot(kind='bar', ax=ax1, color='lightcoral')
ax1.set_title('Anzahl Transaktionen pro PSP', fontweight='bold', fontsize=12)
ax1.set_ylabel('Anzahl')
ax1.set_xlabel('PSP')
for i, v in enumerate(psp_counts):
    ax1.text(i, v + 500, f'{v:,}', ha='center', fontsize=9)
ax1.tick_params(axis='x', rotation=45)
ax1.set_ylim(0, 30000)

# 2. Erfolgsrate pro PSP
ax2 = fig.add_subplot(gs[0, 1])
psp_success = df.groupby('PSP')['success'].mean().sort_values(ascending=False)
psp_success.plot(kind='bar', ax=ax2, color='steelblue')
ax2.set_title('Erfolgsrate pro PSP', fontweight='bold', fontsize=12)
ax2.set_ylabel('Erfolgsrate')
ax2.set_xlabel('PSP')
ax2.set_ylim(0, 0.5)
for i, v in enumerate(psp_success):
    ax2.text(i, v + 0.01, f'{v:.1%}', ha='center', fontsize=9)
ax2.tick_params(axis='x', rotation=45)

# 3. Erfolgsrate nach 3D_secured
ax3 = fig.add_subplot(gs[0, 2])
secure_success = df.groupby('3D_secured')['success'].mean()
colors = ['#ff6b6b', '#51cf66']
secure_success.plot(kind='bar', ax=ax3, color=colors)
ax3.set_title('Erfolgsrate nach 3D-Secure Status', fontweight='bold', fontsize=12)
ax3.set_ylabel('Erfolgsrate')
ax3.set_xlabel('3D-Secure')
ax3.set_xticklabels(['Nein', 'Ja'], rotation=0)
ax3.set_ylim(0, 0.3)
for i, v in enumerate(secure_success):
    ax3.text(i, v + 0.01, f'{v:.1%}', ha='center', fontsize=9)

# =============================================================================
# 2.8.2 BETRAGSANALYSE
# =============================================================================

# 1. Betragsverteilung
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(df['amount'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
ax4.set_title('Verteilung der Beträge', fontweight='bold', fontsize=12)
ax4.set_xlabel('Betrag (€)')
ax4.set_ylabel('Häufigkeit')
ax4.axvline(df['amount'].median(), color='red', linestyle='--', linewidth=2, label=f'Median: {df["amount"].median():.0f}€')
ax4.axvline(df['amount'].mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {df["amount"].mean():.0f}€')
ax4.legend()

# 2. Erfolgsrate vs. Betrag (Scatter + Smoothed)
ax5 = fig.add_subplot(gs[1, 1])
amount_bins = pd.cut(df['amount'], bins=20)
amount_success = df.groupby(amount_bins)['success'].mean()
amount_midpoints = [interval.mid for interval in amount_success.index]
ax5.plot(amount_midpoints, amount_success.values, marker='o', linewidth=2, markersize=6, color='darkred')
ax5.set_title('Erfolgsrate vs. Betragshöhe', fontweight='bold', fontsize=12)
ax5.set_xlabel('Betrag (€)')
ax5.set_ylabel('Erfolgsrate')
ax5.grid(True, alpha=0.3)

# 3. Boxplot Beträge nach Erfolg
ax6 = fig.add_subplot(gs[1, 2])
df.boxplot(column='amount', by='success', ax=ax6)
ax6.set_title('Betragsverteilung nach Erfolg', fontweight='bold', fontsize=12)
ax6.set_xlabel('Erfolg')
ax6.set_ylabel('Betrag (€)')
ax6.set_xticklabels(['Fehlschlag', 'Erfolg'])
plt.sca(ax6)
plt.xticks([1, 2], ['Fehlschlag', 'Erfolg'])

# =============================================================================
# 2.8.3 ZEITLICHE MUSTER
# =============================================================================

# 1. Erfolgsrate nach Stunde
ax7 = fig.add_subplot(gs[2, :2])
hour_success = df.groupby('hour')['success'].mean()
ax7.plot(hour_success.index, hour_success.values, marker='o', linewidth=2, markersize=8, color='purple')
ax7.set_title('Erfolgsrate nach Tageszeit', fontweight='bold', fontsize=12)
ax7.set_xlabel('Stunde')
ax7.set_ylabel('Erfolgsrate')
ax7.set_xticks(range(0, 24, 2))
ax7.grid(True, alpha=0.3)
ax7.legend()

# 2. Erfolgsrate nach Wochentag
ax8 = fig.add_subplot(gs[2, 2])
weekday_names = ['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So']
weekday_success = df.groupby('day_of_week')['success'].mean()
colors_weekday = ['steelblue'] * 5 + ['orange', 'orange']
ax8.bar(range(7), weekday_success.values, color=colors_weekday)
ax8.set_title('Erfolgsrate nach Wochentag', fontweight='bold', fontsize=12)
ax8.set_xlabel('Wochentag')
ax8.set_ylabel('Erfolgsrate')
ax8.set_xticks(range(7))
ax8.set_xticklabels(weekday_names)
ax8.set_ylim(0, 0.25)
for i, v in enumerate(weekday_success.values):
    ax8.text(i, v + 0.005, f'{v:.1%}', ha='center', fontsize=8)

# =============================================================================
# 2.8.4 MEHRFACHVERSUCHE
# =============================================================================

# 1. Verteilung der Versuchsanzahl
ax9 = fig.add_subplot(gs[3, 0])
df['attempt_number'] = df.groupby('purchase_group').cumcount() + 1
df_attempts = df.drop_duplicates(subset=['purchase_group', 'attempt_number'])
group_sizes = df_attempts.groupby('purchase_group').size()
attempt_dist = group_sizes.value_counts().sort_index()
ax9.bar(attempt_dist.index[:12], attempt_dist.values[:12], color='teal')
ax9.set_title('Verteilung Anzahl Zahlungsversuche', fontweight='bold', fontsize=12)
ax9.set_xlabel('Anzahl Versuche')
ax9.set_ylabel('Anzahl Kaufversuche')
for i, v in enumerate(attempt_dist.values[:12]):
    ax9.text(attempt_dist.index[i], v + 100, f'{v:,}', ha='center', fontsize=8)

# 2. Erfolgsrate nach Versuchsnummer
ax10 = fig.add_subplot(gs[3, 1])
# Erstelle attempt_number
df_sorted = df.sort_values(['purchase_group', 'tmsp'])
df_sorted['attempt_number'] = df_sorted.groupby('purchase_group').cumcount() + 1
attempt_success = df_sorted.groupby('attempt_number')['success'].mean()
ax10.plot(attempt_success.index[:12], attempt_success.values[:12], marker='o', linewidth=2, markersize=8, color='darkgreen')
ax10.set_title('Erfolgsrate nach Versuchsnummer', fontweight='bold', fontsize=12)
ax10.set_xlabel('Versuchsnummer')
ax10.set_ylabel('Erfolgsrate')
ax10.set_xticks(range(1, 12))
ax10.grid(True, alpha=0.3)

# 3. PSP-Wechsel Häufigkeit
ax11 = fig.add_subplot(gs[3, 2])
group_sizes = df.groupby('purchase_group').size()
multi_attempt_groups_ids = group_sizes[group_sizes > 1].index
multi_attempt_groups = df[df['purchase_group'].isin(multi_attempt_groups_ids)]

# Anzahl eindeutiger PSPs pro Gruppe
psp_nunique = multi_attempt_groups.groupby('purchase_group')['PSP'].nunique()
n_with_switch = (psp_nunique > 1).sum()
n_without_switch = (psp_nunique == 1).sum()

switch_data = pd.Series({0: n_without_switch, 1: n_with_switch})
labels = ['Kein Wechsel', 'PSP gewechselt']
colors_switch = ['lightcoral', 'lightgreen']
ax11.pie(switch_data.values, labels=labels, autopct='%1.1f%%', colors=colors_switch, startangle=90)
ax11.set_title('PSP-Wechsel bei Mehrfachversuchen', fontweight='bold', fontsize=12)

# =============================================================================
# 2.8.5 INTERAKTIONSEFFEKTE
# =============================================================================

# 1. PSP × 3D_secured
ax12 = fig.add_subplot(gs[4, :2])
interaction_data = df.groupby(['PSP', '3D_secured'])['success'].mean().unstack()
interaction_data.plot(kind='bar', ax=ax12)
ax12.set_title('Interaktionseffekt: PSP × 3D-Secure', fontweight='bold', fontsize=12)
ax12.set_ylabel('Erfolgsrate')
ax12.set_xlabel('PSP')
ax12.legend(['Nicht gesichert', 'Gesichert'], title='3D-Secure')
ax12.tick_params(axis='x', rotation=45)
ax12.grid(True, alpha=0.3)

# 2. PSP × Kartentyp
ax13 = fig.add_subplot(gs[4, 2])
card_psp = df.groupby(['card', 'PSP'])['success'].mean().unstack()
card_psp.plot(kind='bar', ax=ax13)
ax13.set_title('Erfolgsrate: Kartentyp × PSP', fontweight='bold', fontsize=12)
ax13.set_ylabel('Erfolgsrate')
ax13.set_xlabel('Kartentyp')
ax13.legend(title='PSP', fontsize=8, bbox_to_anchor=(1.1, 1.05))
ax13.tick_params(axis='x', rotation=45)

# =============================================================================
# 2.8.6 PSP-SPEZIFISCHE ANALYSEN
# =============================================================================

# 1. Durchschnittlicher Betrag pro PSP
ax14 = fig.add_subplot(gs[5, 0])
psp_amount = df.groupby('PSP')['amount'].mean().sort_values(ascending=False)
psp_amount.plot(kind='bar', ax=ax14, color='lightseagreen')
ax14.set_title('Durchschnittlicher Betrag pro PSP', fontweight='bold', fontsize=12)
ax14.set_ylabel('Durchschnitt (€)')
ax14.set_xlabel('PSP')
ax14.tick_params(axis='x', rotation=45)
ax14.set_ylim(0, 250)
for i, v in enumerate(psp_amount):
    ax14.text(i, v + 2, f'{v:.0f}€', ha='center', fontsize=9)

# 2. 3D-Secure Anteil pro PSP
ax15 = fig.add_subplot(gs[5, 1])
psp_3d = df.groupby('PSP')['3D_secured'].mean().sort_values(ascending=False)
psp_3d.plot(kind='bar', ax=ax15, color='mediumpurple')
ax15.set_title('Anteil 3D-Secure pro PSP', fontweight='bold', fontsize=12)
ax15.set_ylabel('Anteil')
ax15.set_xlabel('PSP')
ax15.tick_params(axis='x', rotation=45)
ax15.set_ylim(0, 0.3)
for i, v in enumerate(psp_3d):
    ax15.text(i, v + 0.01, f'{v:.1%}', ha='center', fontsize=9)

# 3. Erfolgsrate pro Land und PSP
ax16 = fig.add_subplot(gs[5, 2])
country_psp = df.groupby(['country', 'PSP'])['success'].mean().unstack()
country_psp.plot(kind='bar', ax=ax16)
ax16.set_title('Erfolgsrate: Land × PSP', fontweight='bold', fontsize=12)
ax16.set_ylabel('Erfolgsrate')
ax16.set_xlabel('Land')
ax16.legend(title='PSP', fontsize=8, bbox_to_anchor=(1.1, 1.05))
ax16.tick_params(axis='x', rotation=45)

# Titel für gesamte Figure
fig.suptitle('PSP-Datenanalyse: Explorative Plots', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plot_path_EDA = os.path.join(REPORTS_FIG_DIR, 'cart_eda_plots.png')
plt.savefig(plot_path_EDA, dpi=300, bbox_inches='tight')
print(f"\nEDA-Plots: {plot_path_EDA}")
plt.close()

# =============================================================================
# 3. CART ANALYSE
# =============================================================================

print("\n" + "=" * 100)
print(f"3. CART ANALYSE")
print(f"=" * 100)

# =============================================================================
# 3.1 BUSINESS VALUE PARAMETER
# =============================================================================

print("\n" + "=" * 100)
print(f"3.1 BUSINESS VALUE PARAMETER")
print(f"=" * 100)
print(f"\nMittlerer Warenkorbwert: {AVERAGE_BASKET_VALUE:.2f}€")
print(f"Gewinnmarge:               {PROFIT_MARGIN:.1%}")
print(f"Gewinn pro Erfolg:        {REVENUE_PER_SUCCESS:.2f}€")
print(f"Opportunitätskosten:       {C_OPPORT:.1%}")

# =============================================================================
# 3.2 TRAIN-TEST SPLIT
# =============================================================================

print("\n" + "=" * 100)
print("3.2 TRAIN-TEST SPLIT")
print("=" * 100)

# print(f"\nRandom Seed: {RANDOM_STATE}")
print(f"\nAnteil Testdaten am Gesamtdatensatz: {TEST_SIZE:.0%}")

# Train-Test Split mit Berücksichtigung der Kaufvorhaben
purchase_success = df.groupby('purchase_group')['success'].max()
train_groups, test_groups = train_test_split(
    purchase_success.index, test_size=TEST_SIZE,
    random_state=RANDOM_STATE, stratify=purchase_success.values
)

df_train = df[df['purchase_group'].isin(train_groups)].copy()
df_test = df[df['purchase_group'].isin(test_groups)].copy()

print(f"\nTransaktionen:")
print(f"  Train: {len(df_train):,}")
print(f"  Test:  {len(df_test):,}")

print(f"\nPurchase Groups:")
print(f"  Train: {len(train_groups):,}")
print(f"  Test:  {len(test_groups):,}")

print(f"\nErfolgsrate (Transaktionsebene):")
print(f"  Train: {df_train['success'].mean():.2%}")
print(f"  Test:  {df_test['success'].mean():.2%}")

print(f"\nErfolgsrate (Gruppenebene):")
print(f"  Train: {purchase_success.loc[train_groups].mean():.2%}")
print(f"  Test:  {purchase_success.loc[test_groups].mean():.2%}")

# =============================================================================
# 3.3 HYPERPARAMETER-TUNING PRO PSP
# =============================================================================
# Parameterwahl nach One-Standard-Error Rule (Breiman et al., 1984):
# Wähle das einfachste Modell (wenigste Blätter), dessen CV-AUC innerhalb
# eines Standardfehlers des besten Modells liegt.
# =============================================================================

print("\n" + "=" * 100)
print("3.3 HYPERPARAMETER-TUNING PRO PSP")
print("=" * 100)

# Berechne Anzahl Kombinationen
n_combinations = 1
for values in PARAM_GRID.values():
    n_combinations *= len(values)

print(f"\nParameter-Grid:")
for param, values in PARAM_GRID.items():
    print(f"  {param:20s}: {values}")
print(f"\nKombinationen pro PSP: {n_combinations}")
print(f"Stratifizierte Kreuzvalidierung: {CV_FOLDS}-fach")
print(f"Geschätzte Durchläufe gesamt: {n_combinations * CV_FOLDS * len(PSP_LIST):,}")

def evaluate_params_single_psp(params, X, y, groups):
    """
    Evaluiert eine Parameter-Kombination für einen einzelnen PSP.
    Overfit-Diagnose: Train-AUC vs. CV-AUC (kein separater innerer Split nötig).
    StratifiedGroupKFold stellt sicher, dass Purchase Groups nicht über Folds
    aufgeteilt werden (keine Data Leakage durch korrelierte Transaktionen).
    """
    
    if len(X) < 50 or y.nunique() < 2:
        return {
            'cv_auc_mean': np.nan, 'cv_auc_se': np.nan,
            'train_auc': np.nan, 'overfit_gap': np.nan,
            'n_leaves': 0, 'depth': 0
        }
    
    try:
        cart = DecisionTreeClassifier(
            max_depth=params['max_depth'],
            min_samples_leaf=params['min_samples_leaf'],
            min_samples_split=params['min_samples_split'],
            class_weight=params['class_weight'],
            random_state=RANDOM_STATE
        )
        
        # Cross-Validation AUC (gruppiert nach Kaufversuch)
        cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(cart, X, y, cv=cv, groups=groups, scoring='roc_auc')
        cv_auc_mean = cv_scores.mean()
        cv_auc_se = cv_scores.std() / np.sqrt(CV_FOLDS)
        
        # Fit auf gesamten Trainingsdaten für Train-AUC und Baumkomplexität
        cart.fit(X, y)
        
        train_proba = cart.predict_proba(X)
        if train_proba.shape[1] < 2:
            return {
                'cv_auc_mean': cv_auc_mean, 'cv_auc_se': cv_auc_se,
                'train_auc': np.nan, 'overfit_gap': np.nan,
                'n_leaves': cart.get_n_leaves(), 'depth': cart.get_depth()
            }
        
        train_auc = roc_auc_score(y, train_proba[:, 1])
        
        return {
            'cv_auc_mean': cv_auc_mean,
            'cv_auc_se': cv_auc_se,
            'train_auc': train_auc,
            'overfit_gap': train_auc - cv_auc_mean,
            'n_leaves': cart.get_n_leaves(),
            'depth': cart.get_depth()
        }
    
    except Exception as e:
        return {
            'cv_auc_mean': np.nan, 'cv_auc_se': np.nan,
            'train_auc': np.nan, 'overfit_gap': np.nan,
            'n_leaves': 0, 'depth': 0
        }

# Generiere alle Parameter-Kombinationen
param_names = list(PARAM_GRID.keys())
param_values = list(PARAM_GRID.values())
all_combinations = list(product(*param_values))

# Speichere beste Parameter pro PSP
best_params_per_psp = {}
all_psp_results = {}

for psp_name in PSP_LIST:
    
    print("\n" + "=" * 100)
    print(f"GRID SEARCH FÜR: {psp_name}")
    print("=" * 100)
    
    # Filtere Daten für diesen PSP
    df_psp = df_train[df_train['PSP'] == psp_name].copy()
    print(f"  {len(df_psp):,} Transaktionen für {psp_name}")
    
    X = df_psp[feature_names]
    y = df_psp['success']
    groups = df_psp['purchase_group']
    
    # Grid Search für diesen PSP
    psp_results = []
    t0_psp = time.time()
    
    for combo_idx, combo in enumerate(all_combinations):
        params = dict(zip(param_names, combo))
        
        # Evaluiere
        result = evaluate_params_single_psp(params, X, y, groups)
        
        psp_results.append({
            **params,
            **result
        })
        
    # Erstelle DataFrame und sortiere
    df_results = pd.DataFrame(psp_results)
    df_results = df_results.sort_values('cv_auc_mean', ascending=False).reset_index(drop=True)
    
    # Speichere für Zusammenfassung
    all_psp_results[psp_name] = df_results
    
    # Beste Parameter für diesen PSP (One-Standard-Error Rule)
    # Breiman et al. (1984): Wähle das einfachste Modell, dessen CV-AUC
    # innerhalb einer Standardabweichung des besten Modells liegt.
    valid_results = df_results.dropna(subset=['cv_auc_mean'])
    
    # 1. Bestes Modell und Schwellenwert
    best_row = valid_results.iloc[0]
    best_cv_auc = best_row['cv_auc_mean']
    best_cv_se = best_row['cv_auc_se']
    threshold_1se = best_cv_auc - best_cv_se
   
    # 2. Alle Kandidaten innerhalb der 1SE-Bandbreite
    candidates_1se = valid_results[valid_results['cv_auc_mean'] >= threshold_1se].copy()
    
    # 3. Einfachstes Modell wählen:
    candidates_1se = candidates_1se.sort_values(
        ['n_leaves', 'depth', 'max_depth', 'min_samples_leaf', 'min_samples_split'],
        ascending=[True, True, True, False, False]
    )
    selected = candidates_1se.iloc[0]
    
    print(f"\n  1SE-Regel: Bester CV-AUC={best_cv_auc:.4f} ± {best_cv_se:.4f}")
    print(f"             Schwellenwert={threshold_1se:.4f}")
    print(f"             Kandidaten innerhalb 1SE: {len(candidates_1se)} von {len(valid_results)}")
    print(f"             Gewählt: {int(selected['n_leaves'])} Blätter "
          f"(depth={int(selected['max_depth'])}, leaf={int(selected['min_samples_leaf'])})")
    
    best_params_per_psp[psp_name] = {
        'max_depth': int(selected['max_depth']),
        'min_samples_leaf': int(selected['min_samples_leaf']),
        'min_samples_split': int(selected['min_samples_split']),
        'class_weight': selected['class_weight'],
        'cv_auc': selected['cv_auc_mean'],
        'cv_auc_se': selected['cv_auc_se'],
        'train_auc': selected['train_auc'],
        'overfit_gap': selected['overfit_gap'],
        'n_leaves': int(selected['n_leaves']),
        'best_cv_auc': best_cv_auc,
        'threshold_1se': threshold_1se
    }
    
    # Ausgabe Top 10 (nach CV-AUC) + gewähltes 1SE-Modell
    print(f"\n  --- TOP 10 nach CV-AUC für {psp_name} ---")
    print(f"  {'Rank':>4s} {'Depth':>6s} {'MinLeaf':>8s} {'MinSplit':>9s} {'CV-AUC':>8s} {'SE':>6s} {'Train-AUC':>10s} {'Overfit':>8s} {'Blätter':>8s}")
    print("  " + "-" * 82)
    
    for i, (_, row) in enumerate(valid_results.head(10).iterrows()):
        train_str = f"{row['train_auc']:.4f}" if pd.notna(row['train_auc']) else "N/A"
        of_str = f"{row['overfit_gap']:+.4f}" if pd.notna(row['overfit_gap']) else "N/A"
        print(f"  {i+1:>4d} {int(row['max_depth']):>6d} {int(row['min_samples_leaf']):>8d} "
              f"{int(row['min_samples_split']):>9d} {row['cv_auc_mean']:>8.4f} {row['cv_auc_se']:>5.4f} "
              f"{train_str:>10s} {of_str:>8s} {int(row['n_leaves']):>8d}")
    
    # Speichere CSV für diesen PSP
    csv_path = os.path.join(REPORTS_RES_DIR, f'cart_hyperparameter_results_{psp_name}.csv')
    df_results.to_csv(csv_path, index=False, sep=';', decimal=',')
    print(f"\n  CSV: {csv_path}")
    
    # Visualisierung für diesen PSP
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    default_split = PARAM_GRID['min_samples_split'][0]
    default_leaf = PARAM_GRID['min_samples_leaf'][0]
    
    # 1. Heatmap: max_depth vs min_samples_leaf
    ax = axes[0, 0]
    try:
        subset = df_results[(df_results['class_weight'] == 'balanced') & 
                            (df_results['min_samples_split'] == default_split)].dropna(subset=['cv_auc_mean'])
        if len(subset) > 0:
            pivot = subset.pivot_table(values='cv_auc_mean', index='min_samples_leaf', columns='max_depth')
            pivot = pivot.dropna(how='all', axis=0).dropna(how='all', axis=1)
            if pivot.size > 0 and not pivot.isnull().all().all():
                sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGn', ax=ax, cbar_kws={'label': 'CV-AUC'})
                ax.set_title(f'CV-AUC: Depth vs MinLeaf\n(balanced, split={default_split})', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'Keine Daten', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'Keine Daten', ha='center', va='center', transform=ax.transAxes)
    except:
        ax.text(0.5, 0.5, 'Fehler', ha='center', va='center', transform=ax.transAxes)
    
    # 2. Heatmap: max_depth vs min_samples_split
    ax = axes[0, 1]
    try:
        subset = df_results[(df_results['class_weight'] == 'balanced') & 
                            (df_results['min_samples_leaf'] == default_leaf)].dropna(subset=['cv_auc_mean'])
        if len(subset) > 0:
            pivot = subset.pivot_table(values='cv_auc_mean', index='min_samples_split', columns='max_depth')
            pivot = pivot.dropna(how='all', axis=0).dropna(how='all', axis=1)
            if pivot.size > 0 and not pivot.isnull().all().all():
                sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGn', ax=ax, cbar_kws={'label': 'CV-AUC'})
                ax.set_title(f'CV-AUC: Depth vs MinSplit\n(balanced, leaf={default_leaf})', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'Keine Daten', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'Keine Daten', ha='center', va='center', transform=ax.transAxes)
    except:
        ax.text(0.5, 0.5, 'Fehler', ha='center', va='center', transform=ax.transAxes)
    
    # 3. CV-AUC Stabilität (SE pro max_depth)
    ax = axes[0, 2]
    try:
        stability = df_results.groupby('max_depth')['cv_auc_se'].mean()
        ax.bar(stability.index.astype(str), stability.values, color='steelblue', alpha=0.7, edgecolor='black')
        for i, (d, v) in enumerate(stability.items()):
            ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax.set_ylabel('Ø CV-AUC SE', fontweight='bold')
        ax.set_xlabel('max_depth', fontweight='bold')
        ax.set_title('CV-Stabilität nach Tiefe', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    except:
        ax.text(0.5, 0.5, 'Fehler', ha='center', va='center', transform=ax.transAxes)
    
    # 4. CV-AUC vs Overfit-Gap
    ax = axes[1, 0]
    try:
        valid_data = df_results.dropna(subset=['cv_auc_mean', 'overfit_gap'])
        if len(valid_data) > 0:
            scatter = ax.scatter(valid_data['cv_auc_mean'], valid_data['overfit_gap'], 
                                 c=valid_data['max_depth'], cmap='viridis', alpha=0.7, s=50)
            ax.axhline(0, color='red', linestyle='--', alpha=0.5)
            plt.colorbar(scatter, ax=ax, label='max_depth')
    except:
        pass
    ax.set_xlabel('CV-AUC', fontweight='bold')
    ax.set_ylabel('Overfit-Gap (Train - CV)', fontweight='bold')
    ax.set_title('AUC vs Overfitting', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 5. CV-AUC vs Anzahl Blätter
    ax = axes[1, 1]
    try:
        valid_data = df_results.dropna(subset=['n_leaves', 'cv_auc_mean'])
        if len(valid_data) > 0:
            scatter = ax.scatter(valid_data['n_leaves'], valid_data['cv_auc_mean'],
                                 c=valid_data['max_depth'], cmap='viridis', alpha=0.7, s=50)
            plt.colorbar(scatter, ax=ax, label='max_depth')
    except:
        pass
    ax.set_xlabel('Anzahl Blätter', fontweight='bold')
    ax.set_ylabel('CV-AUC', fontweight='bold')
    ax.set_title('Modellkomplexität vs AUC', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 6. Top 10 Kombinationen
    ax = axes[1, 2]
    try:
        top10 = df_results.dropna(subset=['cv_auc_mean']).head(10)
        if len(top10) > 0:
            y_pos = np.arange(len(top10))
            ax.barh(y_pos, top10['cv_auc_mean'], color='forestgreen', alpha=0.7, edgecolor='black')
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"d={int(r['max_depth'])},l={int(r['min_samples_leaf'])}" 
                                for _, r in top10.iterrows()], fontsize=9)
            ax.invert_yaxis()
            ax.barh(0, top10.iloc[0]['cv_auc_mean'], color='gold', alpha=0.9, edgecolor='black')
    except:
        pass
    ax.set_xlabel('CV-AUC', fontweight='bold')
    ax.set_title('Top 10 Kombinationen', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle(f'CART Hyperparameter-Tuning: {psp_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(REPORTS_FIG_DIR, f'cart_hyperparameter_tuning_{psp_name}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Plot: {plot_path}")
    plt.close()

# =============================================================================
# 3.4 BESTE PARAMETER PRO PSP (1SE-REGEL)
# =============================================================================

print("\n" + "=" * 100)
print("3.4 BESTE PARAMETER PRO PSP (1SE-REGEL)")
print("=" * 100)

print(f"\n{'PSP':15s} {'Depth':>6s} {'MinLeaf':>8s} {'MinSplit':>9s} {'CV-AUC':>8s} {'SE':>6s} {'Blätter':>8s} {'Overfit':>8s} {'Best-AUC':>9s} {'Delta':>6s}")
print("-" * 95)

for psp_name in PSP_LIST:
    if psp_name in best_params_per_psp:
        bp = best_params_per_psp[psp_name]
        of_str = f"{bp['overfit_gap']:+.4f}" if pd.notna(bp['overfit_gap']) else "N/A"
        delta = bp['best_cv_auc'] - bp['cv_auc']
        print(f"{psp_name:15s} {bp['max_depth']:>6d} {bp['min_samples_leaf']:>8d} "
              f"{bp['min_samples_split']:>9d} {bp['cv_auc']:>8.4f} {bp['cv_auc_se']:>6.3f} "
              f"{bp['n_leaves']:>8d} {of_str:>8s} {bp['best_cv_auc']:>9.4f} {delta:>6.4f}")
    else:
        print(f"{psp_name:15s}   --- Keine gültigen Ergebnisse ---")

print(f"\n  Delta = Differenz zwischen bestem CV-AUC und gewähltem Modell (≤ 1 SE)")
print(f"  Overfit = Train-AUC − CV-AUC")

# Empfohlene Konfiguration als Python-Dict ausgeben
print("\nEmpfohlene Konfiguration:")
print("PSP_CART_PARAMS = {")
for psp_name in PSP_LIST:
    if psp_name in best_params_per_psp:
        bp = best_params_per_psp[psp_name]
        cw_str = f"'{bp['class_weight']}'" if bp['class_weight'] == 'balanced' else 'None'
        print(f"    '{psp_name}': {{")
        print(f"        'max_depth': {bp['max_depth']},")
        print(f"        'min_samples_leaf': {bp['min_samples_leaf']},")
        print(f"        'min_samples_split': {bp['min_samples_split']},")
        print(f"        'class_weight': {cw_str}")
        print(f"    }},")
print("}")

# Kombiniere alle Ergebnisse und speichere
all_results_combined = []
for psp_name, df_res in all_psp_results.items():
    df_res = df_res.copy()
    df_res['psp'] = psp_name
    all_results_combined.append(df_res)

if all_results_combined:
    df_train_results = pd.concat(all_results_combined, ignore_index=True)
    combined_csv_path = os.path.join(REPORTS_RES_DIR, f'cart_hyperparameter_results.csv')
    df_train_results.to_csv(combined_csv_path, index=False, sep=';', decimal=',')
    print(f"\nKombinierte CSV: {combined_csv_path}")

best_params_path = os.path.join(MODELS_DIR, f'cart_best_params_per_psp.pkl')
with open(best_params_path, 'wb') as f:
    pickle.dump(best_params_per_psp, f)
print(f"Beste Parameter (pkl): {best_params_path}")

# =============================================================================
# 3.5 FINALE MODELLE TRAINIEREN UND KALIBRIEREN
# =============================================================================

print("\n" + "=" * 100)
print("3.5 FINALE MODELLE TRAINIEREN UND KALIBRIEREN")
print("=" * 100)

cart_models = {}

for psp_name in PSP_LIST:
    bp = best_params_per_psp[psp_name]
    df_psp_train = df_train[df_train['PSP'] == psp_name]
    
    X_psp = df_psp_train[feature_names]
    y_psp = df_psp_train['success']
    
    # Finales CART-Modell mit besten Parametern
    cart = DecisionTreeClassifier(
        max_depth=bp['max_depth'],
        min_samples_leaf=bp['min_samples_leaf'],
        min_samples_split=bp['min_samples_split'],
        class_weight=bp['class_weight'],
        random_state=RANDOM_STATE
    )
    
    # Kalibrierung
    groups_psp = df_psp_train['purchase_group']
    
    cv_grouped = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_splits = list(cv_grouped.split(X_psp, y_psp, groups=groups_psp))
    cal_model = CalibratedClassifierCV(cart, method=CALIB_METH, cv=cv_splits)
    cal_model.fit(X_psp, y_psp)

    cart_models[psp_name] = cal_model
    
    # Evaluation auf Testdaten
    df_psp_test = df_test[df_test['PSP'] == psp_name]
    if len(df_psp_test) > 0 and df_psp_test['success'].nunique() == 2:
        X_test_psp = df_psp_test[feature_names]
        y_test_psp = df_psp_test['success']
        test_proba = cal_model.predict_proba(X_test_psp)[:, 1]
        test_auc = roc_auc_score(y_test_psp, test_proba)
        test_brier = brier_score_loss(y_test_psp, test_proba)
        print(f"\n  {psp_name}: n_train={len(X_psp):,} | Test-AUC={test_auc:.4f} | Brier={test_brier:.4f}")
    else:
        print(f"\n  {psp_name}: n_train={len(X_psp):,} | Test-Evaluation nicht möglich")

print(f"\nKalibrierungsmethode: {CALIB_METH}")
print(f"Anzahl trainierte Modelle: {len(cart_models)}")

# Speichere kalibrierte Modelle
model_path = os.path.join(MODELS_DIR, 'cart_psp_recommender_cal.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(cart_models, f)
print(f"Modelle gespeichert: {model_path}")

# =============================================================================
# 3.6 BAUMVISUALISIERUNGEN
# =============================================================================

print("\n" + "=" * 100)
print("3.6 BAUMVISUALISIERUNGEN")
print("=" * 100)

for psp_name in PSP_LIST:
    if psp_name not in best_params_per_psp:
        continue
    
    bp = best_params_per_psp[psp_name]
    df_train_psp = df_train[df_train['PSP'] == psp_name]
    
    # Unkalibrierter Baum für Visualisierung (mit besten Parametern aus Tuning)
    cart_viz = DecisionTreeClassifier(
        max_depth=bp['max_depth'],
        min_samples_leaf=bp['min_samples_leaf'],
        min_samples_split=bp['min_samples_split'],
        class_weight=bp['class_weight'],
        random_state=RANDOM_STATE
    )
    cart_viz.fit(df_train_psp[feature_names], df_train_psp['success'])

    fig, ax = plt.subplots(1, 1, figsize=(24, 12))
    plot_tree(cart_viz, feature_names=feature_names,
              class_names=['Failure', 'Success'],
              filled=True, rounded=True, fontsize=8, 
              ax=ax, proportion=True, max_depth=5)
    ax.set_title(f'CART: {psp_name} (depth={bp["max_depth"]}, '
                 f'leaf={bp["min_samples_leaf"]}, split={bp["min_samples_split"]})',
                 fontsize=16, fontweight='bold')
    
    tree_path = os.path.join(REPORTS_FIG_DIR, f'cart_tree_{psp_name}.png')
    plt.savefig(tree_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  {psp_name}: {tree_path}")
    
    # Text-Regeln
    tree_text = export_text(cart_viz, feature_names=feature_names, max_depth=3)
    print(f"\n  Regeln {psp_name} (Top 3 Ebenen):")
    for line in tree_text.split('\n'):
        print(f"    {line}")
    print()

# =============================================================================
# 3.7 SEQUENZIELLE SIMULATION (REPLAY-METHODE)
# =============================================================================

print("\n" + "=" * 100)
print("3.7 SEQUENZIELLE SIMULATION (REPLAY-METHODE)")
print("=" * 100)

# Berechnung Success Rate für Baseline
t0 = time.time()
base_results = []
base_psp_1, base_psp_2, base_psp_3plus = [], [], []

# Performance-Optimierung: Pre-group für O(1)-Zugriff statt O(n)-Filterung pro Iteration
test_groups_dict = {gid: grp.sort_values('tmsp') for gid, grp in df_test.groupby('purchase_group')}

for idx, group_id in enumerate(test_groups):
    group = test_groups_dict.get(group_id)  # Dictionary-Lookup
    
    if group is None or len(group) == 0:
        continue
    
    if len(group) >= 1: base_psp_1.append(group.iloc[0]['PSP'])
    if len(group) >= 2: base_psp_2.append(group.iloc[1]['PSP'])
    if len(group) >= 3:
        for _, row in group.iloc[2:].iterrows():
            base_psp_3plus.append(row['PSP'])
    
    total_cost = 0
    had_success = False
    n_attempts = 0
    
    for _, row in group.iterrows():
        psp = row['PSP']
        n_attempts += 1
        if row['success']:
            total_cost += FEES[psp]['success']
            had_success = True
            break
        else:
            total_cost += FEES[psp]['failure']
    
    base_results.append({
        'purchase_group': group_id,
        'n_attempts': n_attempts,
        'total_cost': total_cost,
        'success': had_success,
        'profit': (REVENUE_PER_SUCCESS if had_success else 0) - total_cost
    })

df_baseline = pd.DataFrame(base_results)

# =============================================================================
# CART SEQUENZIELLE SIMULATION (REPLAY-METHODE)
# =============================================================================
# Die Replay-Methode kombiniert echte Daten mit Simulation:
# - Wenn CART denselben PSP empfiehlt wie in den echten Daten: nutze echtes Outcome
# - Wenn CART einen anderen PSP empfiehlt: simuliere Outcome basierend auf Modell
# =============================================================================

def recommend_psp(features_dict, models, amount=None):
    amt = amount if amount is not None else AVERAGE_BASKET_VALUE
    feature_array = np.array([features_dict[f] for f in feature_names]).reshape(1, -1)
    
    details = []
    for psp_name, model in models.items():
        prob = model.predict_proba(feature_array)[0, 1]
        exp_cost = prob * FEES[psp_name]['success'] + (1-prob) * FEES[psp_name]['failure']
        exp_revenue = prob * amt * PROFIT_MARGIN
        bv = exp_revenue - exp_cost - (1-prob) * C_OPPORT * amt
        details.append({'psp': psp_name, 'prob': prob, 'exp_cost': exp_cost,
                        'exp_revenue': exp_revenue, 'bv': bv})
    
    details.sort(key=lambda x: x['bv'], reverse=True)
    best = details[0]
    
    if amount is not None:
        return best['psp'], best['prob'], details
    return best['psp'], best['prob']

t0 = time.time()

cart_sim_results = []
cart_psp_1, cart_psp_2, cart_psp_3plus = [], [], []
n_groups = len(test_groups)

# Tracking für Replay-Statistiken
n_real_outcomes = 0
n_simulated_outcomes = 0

for idx, group_id in enumerate(test_groups):

    group = test_groups_dict.get(group_id)

    if group is None or len(group) == 0:
        continue

    first_tx = group.iloc[0]

    base_features = {
        'amount': first_tx['amount'],
        '3D_secured': first_tx['3D_secured'],
        'hour': first_tx['hour'],
        'day_of_week': first_tx['day_of_week'],
        'day_of_month': first_tx['day_of_month'],
        'is_weekend': first_tx['is_weekend'],
        'is_night': first_tx['is_night'],
        'is_morning': first_tx['is_morning'],
        'is_afternoon': first_tx['is_afternoon'],
        'is_evening': first_tx['is_evening'],
        'in_germany': first_tx['in_germany'],
        'in_austria': first_tx['in_austria'],
        'in_switzerland': first_tx['in_switzerland'],
        'is_visa': first_tx['is_visa'],
        'is_master': first_tx['is_master'],
        'is_diners': first_tx['is_diners']
    }

    attempt = 0
    tried_psps = {psp: 0 for psp in PSP_LIST}
    previous_failures = 0
    total_cost = 0
    success = False
    first_psp = None
    second_psp = None
    total_expected_bv = 0

    while attempt < min(len(group), MAX_ATTEMPTS):

        features = {
            **base_features,
            'n_previous_attempts': attempt,
            'n_previous_failures': previous_failures,
            'tried_Moneycard': tried_psps['Moneycard'],
            'tried_Goldcard': tried_psps['Goldcard'],
            'tried_UK_Card': tried_psps['UK_Card'],
            'tried_Simplecard': tried_psps['Simplecard']
        }

        rec_psp, prob = recommend_psp(features, cart_models)

        if attempt == 0:
            first_psp = rec_psp
            cart_psp_1.append(rec_psp)
        elif attempt == 1:
            second_psp = rec_psp
            cart_psp_2.append(rec_psp)
        else:
            cart_psp_3plus.append(rec_psp)
    
        actual_tx = group.iloc[attempt]

        # Expected Business Value (deterministisch)
        exp_cost = prob * FEES[rec_psp]['success'] + (1 - prob) * FEES[rec_psp]['failure']
        expected_bv = prob * REVENUE_PER_SUCCESS - exp_cost - (1 - prob) * C_OPPORT * AVERAGE_BASKET_VALUE
        total_expected_bv += expected_bv

        # REPLAY-METHODE: Nutze echtes Outcome wenn gleicher PSP, sonst simuliere
        if actual_tx['PSP'] == rec_psp:
            # Gleiches PSP: nutze echtes Outcome aus Daten
            outcome = bool(actual_tx['success'])
            n_real_outcomes += 1
        else:
            # Anderes PSP: simuliere basierend auf Modell-Wahrscheinlichkeit
            outcome = rng.random() < prob
            n_simulated_outcomes += 1
        
        cost = FEES[rec_psp]['success'] if outcome else FEES[rec_psp]['failure']
        total_cost += cost
        
        if outcome:
            success = True
            attempt += 1 
            break
        
        tried_psps[rec_psp] += 1
        previous_failures += 1
        attempt += 1

    # Berechne Profit
    profit = (REVENUE_PER_SUCCESS if success else 0) - total_cost

    cart_sim_results.append({
        'purchase_group': group_id,
        'n_attempts': attempt,
        'success': success,
        'total_cost': total_cost,
        'profit': profit,
        'expected_bv': total_expected_bv,
        'first_psp': first_psp,
        'second_psp': second_psp
    })

df_cart = pd.DataFrame(cart_sim_results)

# Ergebnisse in Tabelle ausgeben
base_success = df_baseline['success'].mean()
base_attempts = df_baseline['n_attempts'].mean()
base_cost = df_baseline['total_cost'].mean()
base_profit = df_baseline['profit'].mean()
base_total = df_baseline['profit'].sum()

cart_success = df_cart['success'].mean()
cart_attempts = df_cart['n_attempts'].mean()
cart_cost = df_cart['total_cost'].mean()
cart_profit = df_cart['profit'].mean()
cart_total = df_cart['profit'].sum()

print(f"\nOpportunitätskosten: {C_OPPORT:.1%}")
print(f"\n{'Metrik':25s} {'Baseline':>12s} {'CART':>12s} {'Differenz':>12s}")
print("-" * 65)
print(f"{'Success Rate':25s} {base_success:12.1%} {cart_success:12.1%} {(cart_success-base_success):+12.1%}")
print(f"{'Ø Attempts':25s} {base_attempts:12.2f} {cart_attempts:12.2f} {(cart_attempts-base_attempts):+12.2f}")
print(f"{'Ø Cost':25s} {base_cost:11.2f}€ {cart_cost:11.2f}€ {(cart_cost-base_cost):+11.2f}€ ")
print(f"{'Ø Profit':25s} {base_profit:11.2f}€ {cart_profit:11.2f}€ {(cart_profit-base_profit):+11.2f}€")
print(f"\nHochrechnung (monatlicher Vorteil gegenüber Baseline): {(cart_total - base_total) / 2:,.0f}€")

# Profit CART vs Baseline ---
merged_ch = df_cart.merge(df_baseline[['purchase_group', 'profit']], on='purchase_group', suffixes=('_cart', '_base'))
diff_ch = merged_ch['profit_cart'] - merged_ch['profit_base']

t_ch, p_ch = stats.ttest_rel(merged_ch['profit_cart'], merged_ch['profit_base'])

print(f"\nStatistik (Paired t-Test):")
print(f"  p-Wert für Profit: {p_ch:.4f}")

# Replay-Statistiken
total_outcomes = n_real_outcomes + n_simulated_outcomes
print(f"\nReplay-Methode:")
print(f"  Echte Outcomes:      {n_real_outcomes:,} ({n_real_outcomes/total_outcomes*100:.1f}%)")
print(f"  Simulierte Outcomes: {n_simulated_outcomes:,} ({n_simulated_outcomes/total_outcomes*100:.1f}%)")
print(f"  Total Outcomes:      {total_outcomes:,}")

# Speichere CART-Ergebnisse
cart_results_path = os.path.join(REPORTS_RES_DIR, 'cart_results_psp_recommender_cal.csv')
df_cart.to_csv(cart_results_path, index=False, sep=';', decimal=',')
print(f"\nCART-Ergebnisse in CSV-Datei abgelegt:")
print(f"  {cart_results_path}")

# Grafische Darstellung
fig2, axes = plt.subplots(3, 3, figsize=(16, 14))

# ===== ZEILE 1: Verteilungen =====

# Baseline Attempts Verteilung
ax = axes[0, 0]
ax.hist(df_baseline['n_attempts'], bins=range(1, df_baseline['n_attempts'].max()+2), 
       alpha=0.7, color='gray', edgecolor='black')
ax.axvline(base_attempts, color='red', linestyle='--', linewidth=2, 
          label=f'Ø {base_attempts:.2f}')
ax.set_xlabel('Anzahl Versuche', fontweight='bold')
ax.set_ylabel('Anzahl Kaufversuche', fontweight='bold')
ax.set_title('Verteilung: Versuche pro Kaufversuch (Baseline)', fontweight='bold', fontsize=11)
ax.set_xlim(0, 12)
ax.set_ylim(0, 4000)
ax.legend()
ax.grid(True, alpha=0.3)

# CART Attempts Verteilung
ax = axes[0, 1]
ax.hist(df_cart['n_attempts'], bins=range(1, df_cart['n_attempts'].max()+2), 
       alpha=0.7, color='orange', edgecolor='black')
ax.axvline(cart_attempts, color='red', linestyle='--', linewidth=2, 
          label=f'Ø {cart_attempts:.2f}')
ax.set_xlabel('Anzahl Versuche', fontweight='bold')
ax.set_ylabel('Anzahl Kaufversuche', fontweight='bold')
ax.set_title('Verteilung: Versuche pro Kaufversuch (CART)', fontweight='bold', fontsize=11)
ax.set_xlim(0, 12)
ax.set_ylim(0, 4000)
ax.legend()
ax.grid(True, alpha=0.3)

# Profit-Differenz CART - Baseline
ax = axes[0, 2]
ax.hist(diff_ch, bins=16, alpha=0.7, color='forestgreen', edgecolor='black')
ax.axvline(diff_ch.mean(), color='red', linestyle='--', linewidth=2,
          label=f'Ø {diff_ch.mean():.2f} €')
ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.set_xlabel('Profit-Diff. CART − Baseline (€)', fontweight='bold')
ax.set_ylabel('Anzahl', fontweight='bold')
ax.set_title(f'Profit-Differenz (p={p_ch:.4f})', fontweight='bold', fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ===== ZEILE 2: Baseline vs CART Vergleich =====

x_comp = [0, 1]
colors_comp = ['gray', 'orange']

# Vergleich Success Rate
ax = axes[1, 0]
rates_comp = [base_success * 100, cart_success * 100]
bars = ax.bar(x_comp, rates_comp, color=colors_comp, alpha=0.7, edgecolor='black', width=0.6)
for bar, rate in zip(bars, rates_comp):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylabel('Erfolgsrate (%)', fontweight='bold')
ax.set_title('Erfolgsrate: Baseline vs CART', fontweight='bold', fontsize=11)
ax.set_xticks(x_comp)
ax.set_xticklabels(['Baseline', 'CART'])
ax.set_ylim([0, max(rates_comp) * 1.2])
ax.grid(True, alpha=0.3, axis='y')

# Vergleich Kosten
ax = axes[1, 1]
costs_comp = [base_cost, cart_cost]
bars = ax.bar(x_comp, costs_comp, color=colors_comp, alpha=0.7, edgecolor='black', width=0.6)
for bar, cost in zip(bars, costs_comp):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{cost:.2f}€', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylabel('Ø Kosten (€)', fontweight='bold')
ax.set_title('Durchschnittliche Kosten: Baseline vs CART', fontweight='bold', fontsize=11)
ax.set_ylim(0, 4)
ax.set_xticks(x_comp)
ax.set_xticklabels(['Baseline', 'CART'])
ax.grid(True, alpha=0.3, axis='y')

# Vergleich Profit
ax = axes[1, 2]
profits_comp = [base_profit, cart_profit]
bars = ax.bar(x_comp, profits_comp, color=colors_comp, alpha=0.7, edgecolor='black', width=0.6)
for bar, profit in zip(bars, profits_comp):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{profit:.2f}€', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylabel('Ø Profit (€)', fontweight='bold')
ax.set_title('Profit: Baseline vs CART', fontweight='bold', fontsize=11)
ax.set_xticks(x_comp)
ax.set_xticklabels(['Baseline', 'CART'])
ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.grid(True, alpha=0.3, axis='y')

# ===== ZEILE 3: PSP-Wahl nach Versuchsnummer =====

x_psp_3 = np.arange(len(PSP_LIST))
width_3 = 0.35

def plot_psp_distribution(ax, base_list, model_list, title):
    """Helper-Funktion für PSP-Verteilungsplot"""
    if len(base_list) == 0 and len(model_list) == 0:
        ax.text(0.5, 0.5, 'Keine Daten', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontweight='bold', fontsize=11)
        return
    
    base_dist = pd.Series(base_list).value_counts(normalize=True).reindex(PSP_LIST, fill_value=0)
    model_dist = pd.Series(model_list).value_counts(normalize=True).reindex(PSP_LIST, fill_value=0)
    
    ax.bar(x_psp_3 - width_3/2, [base_dist[p]*100 for p in PSP_LIST], 
           width_3, label='Baseline', color='gray', alpha=0.7, edgecolor='black')
    ax.bar(x_psp_3 + width_3/2, [model_dist[p]*100 for p in PSP_LIST], 
           width_3, label='CART', color='orange', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Anteil (%)', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_xticks(x_psp_3)
    ax.set_xticklabels([p.replace('_', '\n') for p in PSP_LIST], fontsize=9)
    ax.set_ylim(0, 70)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

# PSP-Wahl 1. Versuch
plot_psp_distribution(axes[2, 0], base_psp_1, cart_psp_1, 'PSP-Wahl: 1. Versuch')

# PSP-Wahl 2. Versuch
plot_psp_distribution(axes[2, 1], base_psp_2, cart_psp_2, 'PSP-Wahl: 2. Versuch')

# PSP-Wahl 3+ Versuche
plot_psp_distribution(axes[2, 2], base_psp_3plus, cart_psp_3plus, 'PSP-Wahl: 3+ Versuche')

plt.suptitle(f'Sequenzielle Evaluation - CART Modell (Opportunitätskosten: {C_OPPORT:.1%})', 
             fontsize=14, fontweight='bold')

plt.tight_layout()

plot_path = os.path.join(REPORTS_FIG_DIR, 'cart_evaluation_psp_recommender_cal.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\nGrafische Darstellung der Ergebnisse:")
print(f"  {plot_path}")
plt.close()

# =============================================================================
# 3.8 SENSITIVITÄTSANALYSE: OPPORTUNITÄTSKOSTEN
# =============================================================================

print("\n" + "=" * 100)
print("3.8 SENSITIVITÄTSANALYSE: OPPORTUNITÄTSKOSTEN")
print("=" * 100)

def run_simulation(c_opport_val, cart_models, test_groups, test_groups_dict, seed=RANDOM_STATE):
    """Führt die Replay-Simulation mit gegebenem C_OPPORT durch."""
    
    sim_rng = np.random.default_rng(seed=seed)  # Reproduzierbar pro Durchlauf
    
    def recommend_psp_sens(features_dict, models):
        feature_array = np.array([features_dict[f] for f in feature_names]).reshape(1, -1)
        best_psp, best_value, best_prob = None, -float('inf'), 0
        for psp_name, model in models.items():
            prob = model.predict_proba(feature_array)[0, 1]
            exp_cost = prob * FEES[psp_name]['success'] + (1-prob) * FEES[psp_name]['failure']
            bv = prob * REVENUE_PER_SUCCESS - exp_cost - (1-prob) * c_opport_val * AVERAGE_BASKET_VALUE
            if bv > best_value:
                best_value, best_psp, best_prob = bv, psp_name, prob
        return best_psp, best_prob
    
    sim_results = []
    
    for group_id in test_groups:
        group = test_groups_dict.get(group_id)
        if group is None or len(group) == 0:
            continue
        
        first_tx = group.iloc[0]
        base_feat = {
            'amount': first_tx['amount'], '3D_secured': first_tx['3D_secured'],
            'hour': first_tx['hour'], 'day_of_week': first_tx['day_of_week'],
            'day_of_month': first_tx['day_of_month'], 'is_weekend': first_tx['is_weekend'],
            'is_night': first_tx['is_night'], 'is_morning': first_tx['is_morning'],
            'is_afternoon': first_tx['is_afternoon'], 'is_evening': first_tx['is_evening'],
            'in_germany': first_tx['in_germany'], 'in_austria': first_tx['in_austria'],
            'in_switzerland': first_tx['in_switzerland'],
            'is_visa': first_tx['is_visa'], 'is_master': first_tx['is_master'],
            'is_diners': first_tx['is_diners']
        }
        
        attempt = 0
        tried_psps = {psp: 0 for psp in PSP_LIST}
        previous_failures = 0
        total_cost = 0
        success = False
        
        while attempt < min(len(group), MAX_ATTEMPTS):
            feat = {
                **base_feat,
                'n_previous_attempts': attempt,
                'n_previous_failures': previous_failures,
                'tried_Moneycard': tried_psps['Moneycard'],
                'tried_Goldcard': tried_psps['Goldcard'],
                'tried_UK_Card': tried_psps['UK_Card'],
                'tried_Simplecard': tried_psps['Simplecard']
            }
            
            rec_psp, prob = recommend_psp_sens(feat, cart_models)
            actual_tx = group.iloc[attempt]
            
            if actual_tx['PSP'] == rec_psp:
                outcome = bool(actual_tx['success'])
            else:
                outcome = sim_rng.random() < prob
            
            cost = FEES[rec_psp]['success'] if outcome else FEES[rec_psp]['failure']
            total_cost += cost
            
            if outcome:
                success = True
                attempt += 1
                break
            
            tried_psps[rec_psp] += 1
            previous_failures += 1
            attempt += 1
        
        profit = (REVENUE_PER_SUCCESS if success else 0) - total_cost
        sim_results.append({'success': success, 'profit': profit, 'n_attempts': attempt})
    
    df_sim = pd.DataFrame(sim_results)
    return df_sim['success'].mean(), df_sim['profit'].mean(), df_sim['profit'].sum()

# Sensitivitätsanalyse durchführen
C_OPPORT_VALUES = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]

print(f"\nBaseline Success Rate: {base_success:.1%}")

sens_results = []
for c_val in C_OPPORT_VALUES:
    t0_sens = time.time()
    sr, avg_profit, total_profit = run_simulation(c_val, cart_models, test_groups, test_groups_dict)
    elapsed = time.time() - t0_sens
    sens_results.append({
        'c_opport': c_val, 'success_rate': sr,
        'avg_profit': avg_profit, 'total_profit': total_profit
    })

df_sens = pd.DataFrame(sens_results)

# Tabelle
print(f"\n{'Opp.-Kosten':>12s} {'Success':>10s} {'Ø Profit':>10s} {'Δ Profit':>10s} {'Total':>12s}")
print("-" * 60)
for _, row in df_sens.iterrows():
    delta = row['avg_profit'] - base_profit
    print(f"{row['c_opport']:12.1%} {row['success_rate']:10.1%} {row['avg_profit']:9.2f}€ "
          f"{delta:+9.2f}€ {row['total_profit']:11,.0f}€")

# Visualisierung
fig_sens, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Success Rate
ax1.plot(df_sens['c_opport'] * 100, df_sens['success_rate'] * 100, 
         'o-', color='steelblue', linewidth=2, markersize=8)
ax1.axhline(base_success * 100, color='gray', linestyle='--', linewidth=1.5, label='Baseline')
ax1.set_xlabel('Opportunitätskosten (%)', fontweight='bold')
ax1.set_ylabel('Success Rate (%)', fontweight='bold')
ax1.set_title('Success Rate vs. Opportunitätskosten', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Ø Profit
ax2.plot(df_sens['c_opport'] * 100, df_sens['avg_profit'], 
         'o-', color='forestgreen', linewidth=2, markersize=8)
ax2.axhline(base_profit, color='gray', linestyle='--', linewidth=1.5, label='Baseline')
ax2.set_xlabel('Opportunitätskosten (%)', fontweight='bold')
ax2.set_ylabel('Ø Profit (€)', fontweight='bold')
ax2.set_title('Profit vs. Opportunitätskosten', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('Sensitivitätsanalyse: Einfluss der Opportunitätskosten', 
             fontsize=13, fontweight='bold')
plt.tight_layout()

sens_plot_path = os.path.join(REPORTS_FIG_DIR, f'cart_sensitivity_opport.png')
plt.savefig(sens_plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSensitivitätsplot: {sens_plot_path}")

# =============================================================================
# 4. BEISPIEL: PSP-EMPFEHLUNG BEIM ERSTEM VERSUCH
# =============================================================================
# Demonstriert die Funktionsweise des CART-Recommenders anhand einer
# konkreten Beispiel-Transaktion
# =============================================================================

print("\n" + "=" * 100)
print("4. BEISPIEL: PSP-EMPFEHLUNG BEIM ERSTEN VERSUCH")
print("=" * 100)

example_transaction = {
    'tmsp': '11.02.2019  07:17:42',
    'country': 'Germany',
    'amount': 240,
    '3D_secured': 0,
    'card': 'Master'
}

print(f"\nBeispiel-Transaktion:")
for key, value in example_transaction.items():
    print(f"  {key}: {value}")

ts = pd.Timestamp(example_transaction['tmsp'])
features = {
    'amount': example_transaction['amount'],
    '3D_secured': example_transaction['3D_secured'],
    'hour': ts.hour, 'day_of_week': ts.dayofweek, 'day_of_month': ts.day,
    'is_weekend': int(ts.dayofweek >= 5),
    'is_night': int(ts.hour < 6), 'is_morning': int(6 <= ts.hour < 12),
    'is_afternoon': int(12 <= ts.hour < 18), 'is_evening': int(ts.hour >= 18),
    'in_germany': int(example_transaction['country'] == 'Germany'),
    'in_austria': int(example_transaction['country'] == 'Austria'),
    'in_switzerland': int(example_transaction['country'] == 'Switzerland'),
    'is_visa': int(example_transaction['card'] == 'Visa'),
    'is_master': int(example_transaction['card'] == 'Master'),
    'is_diners': int(example_transaction['card'] == 'Diners'),
    'n_previous_attempts': 0, 'n_previous_failures': 0,
    'tried_Moneycard': 0, 'tried_Goldcard': 0, 'tried_UK_Card': 0, 'tried_Simplecard': 0
}

best_psp, best_prob, details = recommend_psp(features, cart_models, 
                                              amount=example_transaction['amount'])

print(f"\nEmpfehlung 1. Versuch: {best_psp}")
print(f"\nÜbersicht: alle PSP-Bewertungen")
print(f"\n{'PSP':15s} {'P(Erfolg)':>10s} {'E[Kosten]':>10s} {'E[Gewinn]':>10s} {'BV':>8s}")
print("-" * 60)
for d in details:
    print(f"{d['psp']:15s} {d['prob']:9.1%} {d['exp_cost']:9.2f}€ "
          f"{d['exp_revenue']:9.2f}€ {d['bv']:+9.2f}€")

# =============================================================================
# SCHLIESSE LOGGING
# =============================================================================

print(f"\n{'=' * 100}")
print(f"LOG ABGESCHLOSSEN")
print(f"Beendet: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 100}")

sys.stdout = tee.terminal
tee.log.close()

print(f"\nLog gespeichert: {log_filename}")
print(f"Größe: {os.path.getsize(log_filename) / 1024:.1f} KB")
