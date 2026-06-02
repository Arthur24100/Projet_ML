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
partagé /shared_model/ — rempli au préalable par le conteneur "model-builder".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GUIDE DE TEST POUR LE PROFESSEUR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Accès à la documentation interactive : http://localhost:8000/docs
Liste complète des valeurs valides  : http://localhost:8000/options

COMPAGNIES DISPONIBLES :
  Air Asia, Air India, AirAsia, Go First, IndiGo,
  SpiceJet, StarAir, Trujet, Vistara

VILLES DE DÉPART ET D'ARRIVÉE DISPONIBLES :
  Bangalore, Chennai, Delhi, Hyderabad, Kolkata, Mumbai

P�RIODES DE DÉPART DISPONIBLES :
  Morning   → vol entre 05h00 et 11h59
  Afternoon → vol entre 12h00 et 16h59
  Evening   → vol entre 17h00 et 20h59
  Night     → vol entre 21h00 et 04h59

FORMAT DE DATE : JJ-MM-AAAA (ex: 15-03-2022)
  Le dataset couvre la période février à avril 2022.
  Exemples de dates valides : 01-02-2022, 15-03-2022, 10-04-2022

EXEMPLE DE REQUÊTE /predict :
  {
    "airline":    "IndiGo",
    "from_city":  "Delhi",
    "to_city":    "Mumbai",
    "date":       "15-03-2022",
    "dep_period": "Morning"
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
# Charger un StackingRegressor de plusieurs centaines de Mo à chaque appel /predict
# prendrait plusieurs secondes — inacceptable en production.
# En chargeant au démarrage, chaque requête bénéficie du modèle déjà en mémoire.
#
# Le try/except est indispensable : si le modèle est absent (builder pas encore terminé,
# fichier corrompu, mauvaise version numpy), on veut un message d'erreur clair au
# démarrage plutôt qu'un crash cryptique au premier appel /predict.
print("Chargement du modèle...")
try:
    model        = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    encoders     = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    print("Modèle chargé avec succès !")
except Exception as e:
    print(f"ERREUR chargement modèle : {e}")
    raise

# ── Initialisation de l'application FastAPI ────────────────────────────────────
# FastAPI génère automatiquement une documentation interactive à /docs (Swagger UI).
# Le title/description/version apparaissent dans cette doc.
app = FastAPI(
    title="Flight Price Prediction API",
    description=(
        "Prédit le prix d'un vol économique indien (INR / EUR) "
        "à partir de la compagnie, des villes, de la date et de la période de départ.\n\n"
        "**Villes disponibles** : Bangalore, Chennai, Delhi, Hyderabad, Kolkata, Mumbai\n\n"
        "**Compagnies disponibles** : Air Asia, Air India, AirAsia, Go First, IndiGo, "
        "SpiceJet, StarAir, Trujet, Vistara\n\n"
        "**Périodes** : Morning (5h-12h), Afternoon (12h-17h), Evening (17h-21h), Night (21h-5h)\n\n"
        "Appelez **/options** pour la liste complète des valeurs valides."
    ),
    version="1.0.0"
)

# ── Données de référence extraites des LabelEncoders ──────────────────────────
# Ces listes correspondent exactement aux valeurs vues à l'entraînement.
# Une valeur hors de ces listes recevrait un encodage arbitraire (médiane)
# qui produirait une prédiction sans sens.
AIRLINES    = sorted(encoders['airline'].classes_.tolist())
FROM_CITIES = sorted(encoders['from'].classes_.tolist())
TO_CITIES   = sorted(encoders['to'].classes_.tolist())

# ── Correspondance période → heure de départ et d'arrivée ─────────────────────
# Chaque période est traduite en heure numérique représentative pour le modèle.
# Ces valeurs correspondent aux centres des plages horaires du dataset.
PERIOD_TO_HOURS = {
    "Morning":   {"dep_hour": 7,  "arr_hour": 9,  "dep_period": "Morning"},
    "Afternoon": {"dep_hour": 14, "arr_hour": 16, "dep_period": "Afternoon"},
    "Evening":   {"dep_hour": 18, "arr_hour": 20, "dep_period": "Evening"},
    "Night":     {"dep_hour": 22, "arr_hour": 0,  "dep_period": "Night"},
}

# ── Constantes de preprocessing ───────────────────────────────────────────────
# Jours fériés indiens de la période couverte par le dataset (fév-avr 2022).
# Permettent de calculer is_holiday, near_holiday et days_to_holiday.
INDIAN_HOLIDAYS = pd.to_datetime([
    '2022-01-26', '2022-02-16', '2022-03-01', '2022-03-14',
    '2022-03-17', '2022-03-18', '2022-03-19', '2022-04-10',
    '2022-04-14', '2022-04-15'
])
HOLIDAY_SET = set(INDIAN_HOLIDAYS.strftime('%Y-%m-%d'))

# Date minimale du dataset — utilisée pour calculer days_left (position dans la saison).
# Doit correspondre exactement à la valeur utilisée dans train_and_save.py.
DATE_MIN = pd.to_datetime('2022-02-01')

# ── Schéma de la requête ──────────────────────────────────────────────────────
# Pydantic valide automatiquement les types et lève une erreur 422 si un champ
# est manquant ou du mauvais type — avant même d'appeler preprocess().
#
# dep_period est un Literal — Pydantic rejette toute valeur hors de
# {"Morning", "Afternoon", "Evening", "Night"} avec un message d'erreur clair.
# C'est plus robuste qu'un simple str qui accepterait n'importe quelle valeur.
class FlightRequest(BaseModel):
    airline:    str                                              # ex: "IndiGo"
    from_city:  str                                              # ex: "Delhi"
    to_city:    str                                              # ex: "Mumbai"
    date:       str                                              # ex: "15-03-2022"
    dep_period: Literal["Morning", "Afternoon", "Evening", "Night"]  # période de départ

    class Config:
        json_schema_extra = {
            "example": {
                "airline":    "IndiGo",
                "from_city":  "Delhi",
                "to_city":    "Mumbai",
                "date":       "15-03-2022",
                "dep_period": "Morning"
            }
        }

# ── Fonctions de preprocessing ────────────────────────────────────────────────
def encode_col(col_name: str, value: str) -> int:
    """
    Encode une valeur catégorielle avec le LabelEncoder correspondant.
    Si la valeur est inconnue (pas vue à l'entraînement), retourne la valeur médiane
    de l'encodeur plutôt que de planter — comportement dégradé gracieux.
    """
    le = encoders[col_name]
    if value in le.classes_:
        return int(le.transform([value])[0])
    return int(len(le.classes_) // 2)

def preprocess(req: FlightRequest) -> list:
    """
    Reconstruit le vecteur de 19 features attendu par le modèle à partir
    des 5 champs fournis par l'utilisateur.

    Hypothèses fixes (non fournies par l'utilisateur) :
    - duration_minutes : 120 min (2h) — durée médiane des vols courts indiens
    - stop : 'non-stop' — scénario le plus courant
    Les heures dep_hour/arr_hour sont déduites de dep_period.
    """
    # ── Features temporelles ──────────────────────────────────────────────────
    date_dt      = pd.to_datetime(req.date, format='%d-%m-%Y')
    day          = date_dt.day
    month        = date_dt.month
    day_of_week  = date_dt.dayofweek
    is_weekend   = int(day_of_week >= 5)
    week_of_year = int(date_dt.isocalendar()[1])
    days_left    = (date_dt - DATE_MIN).days

    # ── Features jours fériés ─────────────────────────────────────────────────
    date_str     = date_dt.strftime('%Y-%m-%d')
    is_holiday   = int(date_str in HOLIDAY_SET)
    diffs        = [(h - date_dt).days for h in INDIAN_HOLIDAYS if (h - date_dt).days >= 0]
    days_to_hol  = min(diffs) if diffs else 99
    near_holiday = int(days_to_hol <= 7)

    # ── Features horaires déduites de dep_period ──────────────────────────────
    # On traduit la période choisie en heures numériques représentatives.
    # Morning → 7h départ / 9h arrivée, Afternoon → 14h/16h, etc.
    hours        = PERIOD_TO_HOURS[req.dep_period]
    dep_hour     = hours["dep_hour"]
    arr_hour     = hours["arr_hour"]
    dep_period   = hours["dep_period"]

    # ── Durée et escale (hypothèses fixes) ───────────────────────────────────
    duration_minutes = 120.0   # durée médiane des vols courts indiens
    stop             = 'non-stop'

    # ── Features de route ─────────────────────────────────────────────────────
    route         = f"{req.from_city}_{req.to_city}"
    route_airline = f"{req.from_city}_{req.to_city}_{req.airline}"

    # ── Construction du vecteur dans l'ordre exact de feature_cols ───────────
    # L'ordre est critique — sklearn est strict sur l'ordre des colonnes.
    return [
        encode_col('airline',       req.airline),
        encode_col('from',          req.from_city),
        encode_col('to',            req.to_city),
        encode_col('stop',          stop),
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
    """Point d'entrée — redirige vers /docs pour la documentation interactive."""
    return {
        "message": "Flight Price Prediction API",
        "docs": "http://localhost:8000/docs",
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

    À appeler avant /predict pour connaître :
    - Les compagnies disponibles (airline)
    - Les villes de départ disponibles (from_city)
    - Les villes d'arrivée disponibles (to_city)
    - Les périodes de départ (dep_period)
    - Le format de date attendu

    Ces valeurs sont extraites directement des LabelEncoders sauvegardés —
    ce sont exactement les valeurs vues à l'entraînement.
    """
    return {
        "airlines": AIRLINES,
        "from_cities": FROM_CITIES,
        "to_cities": TO_CITIES,
        "dep_periods": {
            "Morning":   "Départ entre 05h00 et 11h59",
            "Afternoon": "Départ entre 12h00 et 16h59",
            "Evening":   "Départ entre 17h00 et 20h59",
            "Night":     "Départ entre 21h00 et 04h59"
        },
        "date_format": "JJ-MM-AAAA",
        "date_exemples": ["01-02-2022", "15-03-2022", "10-04-2022"],
        "exemple_requete": {
            "airline":    "IndiGo",
            "from_city":  "Delhi",
            "to_city":    "Mumbai",
            "date":       "15-03-2022",
            "dep_period": "Morning"
        },
        "note": (
            "La durée (~2h) et le type d'escale (non-stop) sont inférés automatiquement. "
            "Seuls les 5 champs ci-dessus sont nécessaires."
        )
    }

@app.post("/predict")
def predict(req: FlightRequest):
    """
    Endpoint principal de prédiction.

    Reçoit les caractéristiques d'un vol et retourne le prix estimé en INR et EUR.

    Valeurs acceptées pour dep_period : Morning, Afternoon, Evening, Night

    Erreurs possibles :
    - 422 : données invalides (dep_period inconnu, format de date incorrect, etc.)
    - 500 : erreur interne (bug dans le preprocessing ou le modèle)
    """
    try:
        # ── 1. Reconstruction des 19 features ─────────────────────────────────
        features = preprocess(req)

        # ── 2. Validation : aucun NaN dans le vecteur de features ─────────────
        # Un NaN passerait silencieusement dans sklearn et produirait NaN en sortie.
        # On détecte et rejette explicitement avec un message d'erreur utile.
        if any(isinstance(v, float) and np.isnan(v) for v in features):
            raise HTTPException(
                status_code=422,
                detail="Données invalides : valeurs manquantes. Vérifiez le format de la date (JJ-MM-AAAA)."
            )

        # ── 3. Construction du DataFrame avec les bons noms de colonnes ────────
        # sklearn nécessite un DataFrame (pas une liste) pour que les noms
        # de colonnes correspondent à ceux vus à l'entraînement.
        X = pd.DataFrame([features], columns=feature_cols)

        # ── 4. Normalisation avec le scaler fitté à l'entraînement ────────────
        # On utilise .transform() et non .fit_transform() — le scaler est déjà
        # fitté sur X_train. Re-fitter ici introduirait du data leakage.
        X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols)

        # ── 5. Prédiction et transformation inverse du log ────────────────────
        # Le modèle prédit log(prix + 1). expm1 = exp(x) - 1 est l'inverse exact
        # de log1p — on repasse en INR.
        y_log_pred = model.predict(X_scaled)[0]
        prix_inr   = float(np.expm1(y_log_pred))
        prix_eur   = round(prix_inr / 90, 2)

        return {
            "prix_estime_inr": round(prix_inr, 0),
            "prix_estime_eur": prix_eur,
            "vol":             f"{req.from_city} → {req.to_city}",
            "compagnie":       req.airline,
            "date":            req.date,
            "periode":         req.dep_period,
            "note":            "Prix estimé pour un vol non-stop d'environ 2h."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))