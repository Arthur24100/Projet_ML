"""
main.py — Mission 2 & 3 MLOps
==============================
Rôle : API FastAPI qui expose le modèle de prédiction de prix de vols.

Quand un utilisateur envoie les caractéristiques d'un vol (compagnie, ville de
départ/arrivée, date, période de la journée), l'API :
  1. Valide les données reçues (Pydantic)
  2. Reconstruit toutes les features nécessaires au modèle (preprocessing)
  3. Encode et normalise avec les mêmes objets qu'à l'entraînement
  4. Prédit le prix via le StackingRegressor
  5. Retourne le prix estimé en INR et en EUR

L'API tourne dans le conteneur Docker "api", qui lit le modèle depuis le volume
partagé /shared_model/ — rempli au préalable par le conteneur "modele-builder".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GUIDE DE TEST POUR LE PROFESSEUR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Accès à la documentation interactive : http://localhost:8000/docs

COMPAGNIES DISPONIBLES (airline) :
  Air Asia, Air India, AirAsia, Go First, IndiGo,
  SpiceJet, StarAir, Trujet, Vistara

VILLES DE DÉPART ET D'ARRIVÉE DISPONIBLES :
  Bangalore, Chennai, Delhi, Hyderabad, Kolkata, Mumbai

P�RIODES DE DÉPART DISPONIBLES (dep_period) :
  Morning   → vol entre 05h00 et 11h59
  Afternoon → vol entre 12h00 et 16h59
  Evening   → vol entre 17h00 et 20h59
  Night     → vol entre 21h00 et 04h59

FORMAT DE DATE : JJ-MM-AAAA
  Le dataset couvre février à avril 2022.
  Exemples valides : 01-02-2022 | 15-03-2022 | 10-04-2022

EXEMPLE DE REQUÊTE /predict :
  {
    "airline":    "IndiGo",
    "from_city":  "Delhi",
    "to_city":    "Mumbai",
    "stop":       "non-stop",
    "date":       "15-03-2022",
    "dep_time":   "06:00",
    "arr_time":   "08:15",
    "time_taken": "2h 15m"
  }
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import numpy as np
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal

# ── Chemin du volume partagé ───────────────────────────────────────────────────
# Configuré via docker-compose.yml (environment: MODEL_DIR=/shared_model).
# Par défaut /shared_model si la variable n'est pas définie — utile pour
# tester l'API localement sans Docker (export MODEL_DIR=./data/model).
MODEL_DIR = os.environ.get("MODEL_DIR", "/shared_model")

# ── Chargement du modèle au DÉMARRAGE du conteneur ────────────────────────────
# Le chargement se fait une seule fois au démarrage, pas à chaque requête.
# Charger un StackingRegressor à chaque appel /predict prendrait plusieurs
# secondes — inacceptable en production. En chargeant au démarrage, chaque
# requête bénéficie du modèle déjà en mémoire RAM.
#
# Le try/except est indispensable : si un fichier .pkl est absent ou corrompu
# (mauvaise version numpy, builder pas encore terminé), on veut un message
# d'erreur explicite au démarrage plutôt qu'un crash cryptique au premier /predict.
print("Chargement du modèle...")
try:
    model        = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))        # StackingRegressor entraîné
    scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))       # StandardScaler fitté sur X_train
    encoders     = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))     # LabelEncoders par colonne
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl")) # Liste ordonnée des 19 features
    print("Modèle chargé avec succès !")
except Exception as e:
    print(f"ERREUR chargement modèle : {e}")
    raise

# ── Initialisation de l'application FastAPI ────────────────────────────────────
# FastAPI génère automatiquement une documentation interactive à /docs (Swagger UI).
# Le title/description apparaissent dans cette interface — utile pour le prof.
app = FastAPI(
    title="Flight Price Prediction API",
    description=(
        "Prédit le prix d'un vol économique indien (INR / EUR) "
        "à partir de la compagnie, des villes, de la date et des horaires.\n\n"
        "**Villes disponibles** : Bangalore, Chennai, Delhi, Hyderabad, Kolkata, Mumbai\n\n"
        "**Compagnies disponibles** : Air Asia, Air India, AirAsia, Go First, IndiGo, "
        "SpiceJet, StarAir, Trujet, Vistara\n\n"
        "**Format date** : JJ-MM-AAAA (ex: 15-03-2022) — période : fév à avr 2022\n\n"
        "Appelez **/options** pour la liste complète des valeurs valides."
    ),
    version="1.0.0"
)

# ── Données de référence extraites des LabelEncoders ──────────────────────────
# Ces listes correspondent exactement aux valeurs vues à l'entraînement.
# Une valeur hors de ces listes reçoit un encodage médiane arbitraire qui
# produirait une prédiction sans sens — mieux vaut guider l'utilisateur.
AIRLINES    = sorted(encoders['airline'].classes_.tolist())
FROM_CITIES = sorted(encoders['from'].classes_.tolist())
TO_CITIES   = sorted(encoders['to'].classes_.tolist())
STOPS       = sorted(encoders['stop'].classes_.tolist())

# ── Constantes de preprocessing ───────────────────────────────────────────────
# Jours fériés indiens sur la période du dataset (fév-avr 2022).
# Permettent de calculer les features is_holiday, near_holiday, days_to_holiday
# qui ont un impact mesurable sur le prix des billets.
INDIAN_HOLIDAYS = pd.to_datetime([
    '2022-01-26', '2022-02-16', '2022-03-01', '2022-03-14',
    '2022-03-17', '2022-03-18', '2022-03-19', '2022-04-10',
    '2022-04-14', '2022-04-15'
])
# Set pour O(1) au lieu de O(n) à chaque lookup — la requête fait cette
# vérification à chaque appel /predict, autant qu'elle soit rapide.
HOLIDAY_SET = set(INDIAN_HOLIDAYS.strftime('%Y-%m-%d'))

# Date minimale du dataset — utilisée pour calculer days_left.
# Doit correspondre exactement à la valeur de train_and_save.py.
DATE_MIN = pd.to_datetime('2022-02-01')

# ── Schéma de la requête ──────────────────────────────────────────────────────
# Pydantic valide automatiquement les types et lève une erreur 422 si un champ
# est manquant ou du mauvais type — avant même d'appeler preprocess().
# L'utilisateur reçoit un message d'erreur précis plutôt qu'un crash Python.
class FlightRequest(BaseModel):
    airline:    str   # Compagnie aérienne       — ex: "IndiGo"
    from_city:  str   # Ville de départ           — ex: "Delhi"
    to_city:    str   # Ville d'arrivée           — ex: "Mumbai"
    stop:       str   # Type d'escale             — ex: "non-stop" ou "1-stop"
    date:       str   # Date au format JJ-MM-AAAA — ex: "15-03-2022"
    dep_time:   str   # Heure de départ HH:MM     — ex: "06:00"
    arr_time:   str   # Heure d'arrivée HH:MM     — ex: "08:15"
    time_taken: str   # Durée au format XhYm      — ex: "2h 15m"

    class Config:
        # Exemple affiché dans /docs pour guider l'utilisateur
        json_schema_extra = {
            "example": {
                "airline":    "IndiGo",
                "from_city":  "Delhi",
                "to_city":    "Mumbai",
                "stop":       "non-stop",
                "date":       "15-03-2022",
                "dep_time":   "06:00",
                "arr_time":   "08:15",
                "time_taken": "2h 15m"
            }
        }

# ── Fonctions de preprocessing ────────────────────────────────────────────────

def parse_duration(s: str) -> float:
    """
    Convertit une durée texte en minutes numériques.
    Ex : '2h 15m' → 135.0 | '3h' → 180.0
    Retourne NaN si le format est invalide — détecté ensuite dans /predict.
    """
    try:
        h = int(s.strip().split('h')[0])
        m = int(s.strip().split('h')[1].replace('m', '').strip())
        return h * 60 + m
    except:
        return float('nan')

def encode_col(col_name: str, value: str) -> int:
    """
    Encode une valeur catégorielle avec le LabelEncoder correspondant.

    Si la valeur est connue (vue à l'entraînement) → encodage exact.
    Si la valeur est inconnue → retourne la médiane de l'encodeur.
    Ce comportement dégradé gracieux évite un crash sur une compagnie ou
    une ville inconnue, tout en signalant implicitement le problème dans
    la qualité de la prédiction.
    """
    le = encoders[col_name]
    if value in le.classes_:
        return int(le.transform([value])[0])
    # Valeur inconnue : encodage médiane pour ne pas crasher
    return int(len(le.classes_) // 2)

def preprocess(req: FlightRequest) -> list:
    """
    Reconstruit le vecteur de 19 features attendu par le modèle
    à partir des 8 champs fournis par l'utilisateur.

    Toutes les features sont calculées de façon identique à train_and_save.py
    — toute divergence produirait des prédictions fausses sans erreur visible.
    """
    # ── Features temporelles ──────────────────────────────────────────────────
    # On décompose la date en plusieurs signaux indépendants car le modèle
    # ne sait pas interpréter une date brute — il a besoin de composantes numériques.
    date_dt      = pd.to_datetime(req.date, format='%d-%m-%Y')
    day          = date_dt.day          # jour du mois (1-31)
    month        = date_dt.month        # mois (1-12)
    day_of_week  = date_dt.dayofweek    # lundi=0, dimanche=6
    is_weekend   = int(day_of_week >= 5) # 1 si samedi ou dimanche
    week_of_year = int(date_dt.isocalendar()[1])  # semaine ISO (1-53)
    days_left    = (date_dt - DATE_MIN).days       # position dans la saison

    # ── Features jours fériés ─────────────────────────────────────────────────
    # Les prix augmentent autour des jours fériés — ces 3 features capturent
    # l'intensité de cet effet (exact, proche, ou loin du prochain férié).
    date_str     = date_dt.strftime('%Y-%m-%d')
    is_holiday   = int(date_str in HOLIDAY_SET)   # 1 si le vol est un jour férié
    diffs        = [(h - date_dt).days for h in INDIAN_HOLIDAYS if (h - date_dt).days >= 0]
    days_to_hol  = min(diffs) if diffs else 99     # jours avant le prochain férié
    near_holiday = int(days_to_hol <= 7)           # 1 si à moins d'une semaine d'un férié

    # ── Features horaires ─────────────────────────────────────────────────────
    # La durée est convertie de '2h 15m' en 135 minutes.
    # Les heures de départ/arrivée sont extraites en entier (0-23).
    # dep_period est dérivé de dep_hour : Morning/Afternoon/Evening/Night.
    duration_minutes = parse_duration(req.time_taken)
    dep_hour = int(req.dep_time.split(':')[0]) if ':' in req.dep_time else float('nan')
    arr_hour = int(req.arr_time.split(':')[0]) if ':' in req.arr_time else float('nan')

    # Catégorisation de l'heure de départ en 4 périodes
    # (même logique que dans train_and_save.py — doit être identique)
    if not isinstance(dep_hour, float):
        if   5  <= dep_hour < 12: dep_period = 'Morning'
        elif 12 <= dep_hour < 17: dep_period = 'Afternoon'
        elif 17 <= dep_hour < 21: dep_period = 'Evening'
        else:                     dep_period = 'Night'
    else:
        dep_period = 'Unknown'

    # ── Features de route ─────────────────────────────────────────────────────
    # route_airline capture les tarifs spécifiques d'une compagnie sur une route donnée.
    # Ex : IndiGo est cheap sur Delhi-Mumbai mais pas forcément sur Chennai-Kolkata.
    route         = f"{req.from_city}_{req.to_city}"
    route_airline = f"{req.from_city}_{req.to_city}_{req.airline}"

    # ── Construction du vecteur dans l'ordre exact de feature_cols ───────────
    # L'ordre doit correspondre EXACTEMENT à celui de train_and_save.py.
    # sklearn reconstruit X colonne par colonne — un ordre différent produit
    # des prédictions fausses sans lever d'erreur.
    return [
        encode_col('airline',       req.airline),
        encode_col('from',          req.from_city),
        encode_col('to',            req.to_city),
        encode_col('stop',          req.stop),
        encode_col('dep_period',    dep_period),
        encode_col('route',         route),
        encode_col('route_airline', route_airline),
        duration_minutes,
        dep_hour,
        arr_hour,
        day, month, day_of_week,
        is_weekend, week_of_year, days_left,
        is_holiday, near_holiday, days_to_hol,
    ]

# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    """Point d'entrée — renvoie les URLs utiles pour démarrer."""
    return {
        "message": "Flight Price Prediction API",
        "docs":    "http://localhost:8000/docs",
        "options": "http://localhost:8000/options"
    }

@app.get("/health")
def health():
    """
    Endpoint de healthcheck — utilisé par Docker pour vérifier que le conteneur
    est vivant et prêt à recevoir des requêtes.
    Retourne toujours 200 OK si le service est opérationnel.
    """
    return {"status": "ok"}

@app.get("/options")
def options():
    """
    Retourne toutes les valeurs valides pour remplir une requête /predict.

    À appeler avant /predict pour connaître exactement quelles compagnies,
    villes et types d'escale sont acceptés par le modèle.
    Ces listes sont extraites directement des LabelEncoders — elles correspondent
    aux valeurs vues à l'entraînement.
    """
    return {
        "airlines":    AIRLINES,
        "from_cities": FROM_CITIES,
        "to_cities":   TO_CITIES,
        "stops":       STOPS,
        "dep_periods": {
            "Morning":   "Départ entre 05h00 et 11h59 → dep_time ex: '06:00'",
            "Afternoon": "Départ entre 12h00 et 16h59 → dep_time ex: '14:30'",
            "Evening":   "Départ entre 17h00 et 20h59 → dep_time ex: '18:00'",
            "Night":     "Départ entre 21h00 et 04h59 → dep_time ex: '22:00'"
        },
        "date_format":  "JJ-MM-AAAA",
        "date_exemples": ["01-02-2022", "15-03-2022", "10-04-2022"],
        "time_taken_format": "XhYm (ex: '2h 15m', '3h 0m')",
        "exemple_requete": {
            "airline":    "IndiGo",
            "from_city":  "Delhi",
            "to_city":    "Mumbai",
            "stop":       "non-stop",
            "date":       "15-03-2022",
            "dep_time":   "06:00",
            "arr_time":   "08:15",
            "time_taken": "2h 15m"
        }
    }

@app.post("/predict")
def predict(req: FlightRequest):
    """
    Endpoint principal de prédiction.

    Reçoit les caractéristiques complètes d'un vol et retourne le prix estimé.

    Erreurs possibles :
    - 422 : données invalides (NaN dans les features, format de date incorrect,
            heure mal formatée, durée illisible)
    - 500 : erreur interne inattendue dans le preprocessing ou le modèle
    """
    try:
        # ── 1. Reconstruction des 19 features ─────────────────────────────────
        features = preprocess(req)

        # ── 2. Détection des NaN ───────────────────────────────────────────────
        # Un NaN dans le vecteur de features passerait silencieusement dans sklearn
        # et produirait NaN en sortie de prédiction, sans erreur Python.
        # On le détecte ici et on retourne une erreur 422 explicite.
        # Cause typique : format dep_time ou time_taken invalide.
        if any(isinstance(v, float) and np.isnan(v) for v in features):
            raise HTTPException(
                status_code=422,
                detail=(
                    "Données invalides : valeurs manquantes détectées. "
                    "Vérifiez : date (JJ-MM-AAAA), dep_time/arr_time (HH:MM), "
                    "time_taken (XhYm ex: '2h 15m')."
                )
            )

        # ── 3. Construction du DataFrame ───────────────────────────────────────
        # sklearn nécessite un DataFrame nommé (pas une liste brute) pour que
        # les noms de colonnes correspondent à ceux vus à l'entraînement.
        X = pd.DataFrame([features], columns=feature_cols)

        # ── 4. Normalisation ──────────────────────────────────────────────────
        # On utilise .transform() et NON .fit_transform() — le scaler est déjà
        # fitté sur X_train dans train_and_save.py. Re-fitter ici sur une seule
        # ligne produirait une normalisation complètement différente (data leakage).
        X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols)

        # ── 5. Prédiction + transformation inverse ────────────────────────────
        # Le modèle prédit log(prix + 1) — on repasse en INR avec expm1.
        # expm1(x) = exp(x) - 1 est l'inverse mathématique exact de log1p(x).
        y_log_pred = model.predict(X_scaled)[0]
        prix_inr   = float(np.expm1(y_log_pred))

        # Conversion INR → EUR (taux approximatif 1 EUR ≈ 90 INR)
        prix_eur = round(prix_inr / 90, 2)

        return {
            "prix_estime_inr": round(prix_inr, 0),
            "prix_estime_eur": prix_eur,
            "vol":             f"{req.from_city} → {req.to_city}",
            "compagnie":       req.airline,
            "date":            req.date,
            "periode":         "Morning" if int(req.dep_time.split(':')[0]) < 12
                               else "Afternoon" if int(req.dep_time.split(':')[0]) < 17
                               else "Evening" if int(req.dep_time.split(':')[0]) < 21
                               else "Night"
        }

    except HTTPException:
        # On laisse passer les HTTPException levées explicitement (422, etc.)
        raise
    except Exception as e:
        # Toute autre exception → 500 avec le message d'erreur.
        # En production, on logguerait le traceback complet dans un système
        # de monitoring (Sentry, Datadog) avant de retourner le 500.
        raise HTTPException(status_code=500, detail=str(e))