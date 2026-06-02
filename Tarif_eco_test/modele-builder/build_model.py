"""
build_model.py — Mission 1 MLOps
=================================
Rôle : ce script tourne dans le conteneur Docker "model-builder".
Il ne ré-entraîne PAS le modèle (l'entraînement se fait en dehors de Docker
via train_and_save.py). Son seul travail est de copier les fichiers .pkl
depuis le dossier monté /data/model/ vers le volume Docker partagé /shared_model/,
pour que le conteneur "api" puisse y accéder au démarrage.

Flux complet :
  1. Tu lances train_and_save.py sur ta machine → génère les .pkl dans data/model/
  2. Docker monte data/model/ dans ce conteneur sous /data/model/
  3. Ce script copie les.pkl dans /shared_model/ (volume Docker partagé)
  4. Le conteneur "api" lit les .pkl depuis /shared_model/ et démarre FastAPI
"""

import os
import shutil
import joblib

# ── Variables d'environnement ──────────────────────────────────────────────────
# Ces deux chemins sont configurables via docker-compose.yml (section environment).
# Par défaut :
#   SRC_DIR    = /data/model     → là où Docker monte tes .pkl locaux
#   OUTPUT_DIR = /shared_model   → le volume partagé entre les deux conteneurs
#
# Utiliser des variables d'environnement (plutôt que des chemins en dur)
# permet de changer les chemins sans modifier le code — utile si on déploie
# sur un autre environnement (serveur de prod, CI/CD, etc.)
SRC_DIR    = os.environ.get("SRC_DIR",   "/data/model")
OUTPUT_DIR = os.environ.get("MODEL_DIR", "/shared_model")

# Crée /shared_model/ s'il n'existe pas encore (premiers run sur un volume vide)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Liste des fichiers à copier ────────────────────────────────────────────────
# Ces 4 fichiers forment le "package modèle" complet :
#   model.pkl        → le StackingRegressor entraîné
#   scaler.pkl       → le StandardScaler fitté sur X_train (obligatoire pour normaliser
#                      les nouvelles données avec les mêmes stats qu'à l'entraînement)
#   encoders.pkl     → les LabelEncoders par colonne catégorielle (airline, route, etc.)
#   feature_cols.pkl → la liste ordonnée des 19 features (sklearn est strict sur l'ordre)
FILES = ["model.pkl", "scaler.pkl", "encoders.pkl", "feature_cols.pkl"]

print("Copie des modèles vers le volume partagé...")
for f in FILES:
    src = os.path.join(SRC_DIR, f)
    dst = os.path.join(OUTPUT_DIR, f)

    # Vérification explicite : si un fichier manque, on arrête tout avec un message clair.
    # Sans cette vérification, joblib.load() planterait plus tard avec une erreur cryptique.
    # Le message indique exactement quoi faire (lancer train_and_save.py) plutôt que
    # de laisser l'utilisateur chercher pourquoi le conteneur crash.
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"Fichier manquant : {src}\n"
            "Lance d'abord train_and_save.py sur ta machine !"
        )

    # shutil.copy2 copie le fichier ET ses métadonnées (date de modification, permissions).
    # Préféré à shutil.copy qui ne copie pas les métadonnées — utile pour tracer
    # quelle version du modèle a été déployée.
    shutil.copy2(src, dst)
    print(f"  ✓ {f}")

# ── Validation finale ──────────────────────────────────────────────────────────
# On charge le modèle pour vérifier que le fichier n'est pas corrompu et que
# les versions de scikit-learn et numpy sont compatibles entre ta machine et Docker.
# C'est ici que le classique "PCG64 is not a known BitGenerator" apparaît si les
# versions de numpy ne correspondent pas — mieux vaut le détecter dans le builder
# que de laisser l'API crasher au premier appel /predict.
model = joblib.load(os.path.join(OUTPUT_DIR, "model.pkl"))
print(f"\nModèle validé : {type(model).__name__}")
print("Volume prêt pour l'API !")