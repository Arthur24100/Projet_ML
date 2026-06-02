"""
build_model.py — Mission 1 MLOps (version entraînement dans Docker)
====================================================================
Rôle : ce script tourne DANS le conteneur Docker "model-builder".
Contrairement à la version train_and_save.py (qui tourne sur ta machine),
cette version entraîne le modèle directement dans Docker à partir du CSV.

Pourquoi deux versions ?
  - train_and_save.py : entraînement complet sur toutes les données, hors Docker,
    avec les ressources complètes de ta machine. Produit les meilleurs .pkl.
  - build_model.py (ce fichier) : entraînement allégé DANS Docker, pour les cas
    où on veut que Docker soit autonome (pas de .pkl pré-générés requis).
    Utilise un échantillon de 150 000 lignes et des modèles simplifiés pour
    tenir dans la RAM limitée de Docker Desktop (~4-8 Go par défaut sur Mac).

Flux d'exécution :
  1. Lit economy.csv depuis /data/ (volume monté via docker-compose.yml)
  2. Préprocesse, entraîne, évalue
"""

import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor, HistGradientBoostingRegressor,
    ExtraTreesRegressor, StackingRegressor, BaggingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── Dossier de sortie ──────────────────────────────────────────────────────────
# Configuré via docker-compose.yml (environment: MODEL_DIR=/shared_model).
# os.environ.get fournit une valeur par défaut si la variable n'est pas définie —
# utile pour tester le script localement sans Docker.
OUTPUT_DIR = os.environ.get("MODEL_DIR", "/shared_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT & NETTOYAGE
# ══════════════════════════════════════════════════════════════════════════════
print("Chargement des données...")

# Le CSV est lu depuis /data/economy.csv — chemin monté dans docker-compose.yml :
#   volumes:
#     - ./data:/data
# Cela mappe le dossier data/ local vers /data/ dans le conteneur.
df = pd.read_csv('/data/economy.csv')

# La colonne stop peut contenir des retours à la ligne dans certaines entrées
# (ex: "1-stop\n2h layover"). On garde uniquement la première partie avant \n
# pour uniformiser les valeurs : 'non-stop', '1-stop', '2+-stop'.
df['stop'] = df['stop'].str.strip().str.split('\n').str[0].str.strip()

# price est stocké en string avec virgules comme séparateur de milliers ("5,953").
# str.replace supprime les virgules, astype(float) convertit en numérique.
df['price'] = df['price'].str.replace(',', '').astype(float)

# Suppression des lignes strictement identiques sur toutes les colonnes.
# Les doublons biaiseraient l'entraînement : le modèle verrait certains patterns
# deux fois et les sur-apprendrait par rapport aux patterns rares.
df = df.drop_duplicates().reset_index(drop=True)

# ── Suppression des outliers (méthode IQR) ────────────────────────────────────
# Q1 = 25e percentile, Q3 = 75e percentile, IQR = écart interquartile.
# Borne basse : Q1 - 1.5×IQR  → supprime les vols anormalement bon marché
# Borne haute : Q3 + 3.0×IQR  → on utilise 3× (et non 1.5×) pour conserver
# les vols premium légitimement chers sans les traiter comme des outliers.
Q1, Q3 = df['price'].quantile(0.25), df['price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 3.0 * IQR)].copy()

# ── Échantillonnage pour Docker Desktop ───────────────────────────────────────
# Docker Desktop alloue par défaut 4-8 Go de RAM sur Mac. Entraîner un
# StackingRegressor avec 5 base learners × cv=5 sur 200 000 lignes crée
# des structures intermédiaires qui dépassent cette limite → OOM kill (code 137).
# On tire aléatoirement 150 000 lignes représentatives avec random_state=42
# pour garantir la reproductibilité.
# En production sur un vrai serveur, on supprimerait cette ligne et
# utiliserait le dataset complet (train_and_save.py fait ça).
SAMPLE_SIZE = 150_000
if len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    print(f"Échantillon : {SAMPLE_SIZE:,} lignes (dataset complet → utilise train_and_save.py)")

# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
# On extrait des signaux exploitables depuis les colonnes brutes (texte, dates).
# Chaque feature doit capturer un aspect indépendant qui influence le prix.
print("Feature engineering...")

# ── Features temporelles ──────────────────────────────────────────────────────
# La date brute n'est pas exploitable directement — les modèles n'interprètent
# pas les dates. On la décompose en composantes numériques indépendantes.
df['date_dt']      = pd.to_datetime(df['date'], format='%d-%m-%Y')
df['day']          = df['date_dt'].dt.day          # jour du mois (1-31)
df['month']        = df['date_dt'].dt.month        # mois (1-12) — effet saisonnier
df['day_of_week']  = df['date_dt'].dt.dayofweek    # lundi=0, dimanche=6
df['is_weekend']   = (df['day_of_week'] >= 5).astype(int)  # 1 si samedi/dimanche
df['week_of_year'] = df['date_dt'].dt.isocalendar().week.astype(int)  # semaine ISO
# days_left : jours écoulés depuis la date minimale du dataset.
# Capture la position dans la saison — les prix évoluent au fil du temps.
df['days_left']    = (df['date_dt'] - df['date_dt'].min()).dt.days

# ── Features jours fériés indiens ─────────────────────────────────────────────
# Les prix augmentent systématiquement avant et pendant les jours fériés.
# On encode 3 niveaux de proximité au prochain férié :
#   is_holiday   → le vol est exactement un jour férié
#   days_to_holiday → jours avant le prochain férié (signal continu)
#   near_holiday → 1 si à moins de 7 jours d'un férié (signal binaire)
indian_holidays = pd.to_datetime([
    '2022-01-26', '2022-02-16', '2022-03-01', '2022-03-14',
    '2022-03-17', '2022-03-18', '2022-03-19', '2022-04-10',
    '2022-04-14', '2022-04-15'
])
# set() pour O(1) à chaque lookup au lieu de O(n) avec une liste
holiday_set = set(indian_holidays.strftime('%Y-%m-%d'))
df['is_holiday'] = df['date_dt'].dt.strftime('%Y-%m-%d').isin(holiday_set).astype(int)

def days_to_holiday(d):
    """Retourne le nombre de jours avant le prochain jour férié, 99 si aucun."""
    diffs = [(h - d).days for h in indian_holidays if (h - d).days >= 0]
    return min(diffs) if diffs else 99

df['days_to_holiday'] = df['date_dt'].apply(days_to_holiday)
df['near_holiday']    = (df['days_to_holiday'] <= 7).astype(int)

# ── Features horaires ─────────────────────────────────────────────────────────
# La durée est en texte ('2h 15m') → conversion en minutes (135).
# Conversion directement exploitable par le modèle comme feature numérique.
def parse_duration(s):
    """Convertit '2h 15m' → 135. Retourne NaN si format invalide."""
    try:
        h = int(s.strip().split('h')[0])
        m = int(s.strip().split('h')[1].replace('m', '').strip())
        return h * 60 + m
    except:
        return np.nan

df['duration_minutes'] = df['time_taken'].apply(parse_duration)

# Extraction de l'heure entière depuis 'HH:MM' (ex: '06:30' → 6)
df['dep_hour'] = df['dep_time'].apply(
    lambda t: int(t.split(':')[0]) if ':' in str(t) else np.nan
)
df['arr_hour'] = df['arr_time'].apply(
    lambda t: int(t.split(':')[0]) if ':' in str(t) else np.nan
)

# dep_period : catégorisation de l'heure de départ en 4 périodes.
# Plus robuste qu'une heure exacte — un vol à 6h et à 7h partagent
# le même comportement tarifaire (vol du matin d'affaires).
df['dep_period'] = df['dep_hour'].apply(
    lambda h: 'Morning'   if 5  <= h < 12
    else ('Afternoon' if 12 <= h < 17
    else ('Evening'   if 17 <= h < 21
    else 'Night')) if not np.isnan(h) else 'Unknown'
)

# ── Features de route ─────────────────────────────────────────────────────────
# route : capture que Delhi-Mumbai a une structure tarifaire différente
# de Chennai-Kolkata (distance, concurrence, demande).
df['route'] = df['from'] + '_' + df['to']

# route_airline : IndiGo peut être compétitif sur Delhi-Mumbai mais pas
# sur Bangalore-Chennai. Cette feature capture ces spécificités.
df['route_airline'] = df['from'] + '_' + df['to'] + '_' + df['airline']

# ══════════════════════════════════════════════════════════════════════════════
# 3. ENCODAGE
# ══════════════════════════════════════════════════════════════════════════════
# Les modèles sklearn ne comprennent que des valeurs numériques.
# LabelEncoder assigne un entier à chaque valeur unique de la colonne.
# On sauvegarde un encodeur DISTINCT par colonne pour l'API — elle en a besoin
# pour encoder les nouvelles requêtes avec les mêmes entiers qu'à l'entraînement.
cat_cols = ['airline', 'from', 'to', 'stop', 'dep_period', 'route', 'route_airline']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    encoders[col] = le  # sauvegardé dans le dictionnaire pour le pkl

# ── Sélection des 19 features du modèle ──────────────────────────────────────
# Ces 19 features couvrent 5 dimensions du prix :
#   - Identité du vol  : airline, from, to, stop, route, route_airline
#   - Horaires         : dep_hour, arr_hour, dep_period, duration_minutes
#   - Calendrier       : day, month, day_of_week, is_weekend, week_of_year, days_left
#   - Fériés           : is_holiday, near_holiday, days_to_holiday
feature_cols = [
    'airline_encoded', 'from_encoded', 'to_encoded', 'stop_encoded',
    'dep_period_encoded', 'route_encoded', 'route_airline_encoded',
    'duration_minutes', 'dep_hour', 'arr_hour', 'day', 'month', 'day_of_week',
    'is_weekend', 'week_of_year', 'days_left',
    'is_holiday', 'near_holiday', 'days_to_holiday'
]

# dropna() supprime les lignes avec au moins un NaN dans les features.
# Ces NaN viennent de parse_duration ou de dep_hour/arr_hour sur des entrées malformées.
X = df[feature_cols].dropna().reset_index(drop=True)
y = df.loc[df[feature_cols].dropna().index, 'price'].reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# 4. SCALING & TRANSFORMATION LOG
# ══════════════════════════════════════════════════════════════════════════════

# StandardScaler : ramène chaque feature à moyenne=0 et écart-type=1.
# Obligatoire pour KNN (sensible à l'échelle des features — une feature en
# milliers dominerait les distances euclidiennes).
# Sans effet sur les arbres (RF, ET, HGB) mais on garde un pipeline uniforme.
# On utilise fit_transform ici (pas seulement transform) car c'est l'entraînement —
# le scaler apprend les statistiques sur ces données.
scaler   = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

# Transformation log sur la variable cible.
# La distribution brute du prix est fortement asymétrique à droite (right-skewed).
# log1p(x) = log(x+1) rend la distribution plus symétrique, ce qui améliore
# la convergence des modèles et produit des résidus plus homoscédastiques.
# À la prédiction, on inverse avec expm1(x) = exp(x)-1 pour repasser en INR.
y_log = np.log1p(y)

# Split 80% train / 20% test avec random_state=42 pour la reproductibilité.
# test_size=0.2 → ~30 000 lignes de test sur 150 000 — suffisant pour des
# métriques statistiquement stables.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_log, test_size=0.2, random_state=42
)
y_test_real = np.expm1(y_test)  # repasse en INR pour l'évaluation finale
print(f"Dataset : {X_scaled.shape} | Train : {X_train.shape[0]:,} | Test : {X_test.shape[0]:,}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. ENTRAÎNEMENT DU STACKINGREGRESSOR
# ══════════════════════════════════════════════════════════════════════════════
# Le Stacking combine plusieurs modèles complémentaires via un méta-modèle.
# Chaque base learner est entraîné sur X_train, puis ses prédictions out-of-fold
# (cv=3) servent de features d'entrée au méta-modèle.
#
# VERSION ALLÉGÉE pour Docker Desktop :
#   n_estimators réduits (50 au lieu de 100) → moins de RAM, build plus rapide
#   cv=3 au lieu de 5 → 40% moins d'entraînements intermédiaires
#   n_jobs=1 → pas de parallélisme pour éviter les OOM sur Docker Desktop
#   Pour les meilleures performances → utiliser train_and_save.py hors Docker
print("Entraînement du StackingRegressor (version allégée pour Docker)...")

estimators = [
    # Random Forest : bagging d'arbres, splits optimaux sur sous-ensemble de features
    ('rf',      RandomForestRegressor(n_estimators=50,  random_state=42, n_jobs=-1)),

    # Extra Trees : splits ENTIÈREMENT aléatoires → plus rapide que RF,
    # variance réduite grâce à la randomisation maximale
    ('et',      ExtraTreesRegressor(n_estimators=50,    random_state=42, n_jobs=-1)),

    # HistGradientBoosting : boosting séquentiel sur histogrammes, très efficace
    # sur données tabulaires, capture les non-linéarités fines
    ('hgb',     HistGradientBoostingRegressor(max_iter=50, random_state=42)),

    # KNN : raisonnement par voisinage géométrique — approche complètement
    # différente des arbres, apporte de la diversité au Stacking
    ('knn',     KNeighborsRegressor(n_neighbors=10, n_jobs=-1)),

    # Bagging(DT) : arbres de décision en bagging, profondeur limitée à 8
    # pour éviter le surapprentissage dans le contexte du Stacking
    ('bagging', BaggingRegressor(
                    estimator=DecisionTreeRegressor(max_depth=8),
                    n_estimators=20, random_state=42, n_jobs=-1)),
]

stack = StackingRegressor(
    estimators=estimators,

    # Méta-modèle : HGB plutôt qu'une régression linéaire car les relations
    # entre prédictions des base learners et prix final sont non-linéaires.
    # Le méta-modèle apprend "quand faire confiance à RF vs KNN".
    final_estimator=HistGradientBoostingRegressor(
        max_iter=100, learning_rate=0.05, random_state=42
    ),

    # cv=3 : les données de train sont divisées en 3 plis. Chaque base learner
    # prédit sur le pli qu'il n'a pas vu (out-of-fold). Ces prédictions OOF
    # constituent les features d'entrée du méta-modèle. Sans cv, le méta-modèle
    # serait entraîné sur des prédictions in-sample (trop optimistes → surapprentissage).
    cv=3,

    # n_jobs=1 : désactive la parallélisation pour éviter les OOM sur Docker Desktop.
    # Avec n_jobs=-1, Python spawne plusieurs processus qui se partagent la RAM —
    # sur un système à 4 Go alloués, ça dépasse facilement le seuil.
    n_jobs=1
)
stack.fit(X_train, y_train)

# ══════════════════════════════════════════════════════════════════════════════
# 6. ÉVALUATION SUR LE JEU DE TEST
# ══════════════════════════════════════════════════════════════════════════════
# On évalue sur les 20% de données jamais vues pendant l'entraînement.
# expm1 pour repasser les prédictions log → INR avant de calculer les métriques.
y_pred = np.expm1(stack.predict(X_test))

r2   = r2_score(y_test_real, y_pred)
mae  = mean_absolute_error(y_test_real, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred))

# R²  : proportion de la variance du prix expliquée (1.0 = parfait)
# MAE : erreur absolue moyenne en INR — directement interprétable
# RMSE : sensible aux grosses erreurs, plus élevé que MAE si erreurs catastrophiques
print(f"R2   : {r2:.4f}  → le modèle explique {r2*100:.1f}% de la variance du prix")
print(f"MAE  : {mae:.0f} INR  (≈ {mae/90:.0f} EUR)")
print(f"RMSE : {rmse:.0f} INR")

