# Dockerfile — conteneur model-builder (version entraînement dans Docker)
# =========================================================================
# Rôle : construire l'image Docker du conteneur "model-builder".
# Ce conteneur entraîne le StackingRegressor depuis le CSV et sauvegarde
# les .pkl dans le volume partagé /shared_model/.
# Il s'exécute UNE SEULE FOIS au démarrage, puis se termine — le conteneur
# "api" prend le relais grâce à "condition: service_completed_successfully".

# ── Image de base ──────────────────────────────────────────────────────────────
# python:3.9-slim : image Python allégée (~125 Mo vs ~900 Mo pour python:3.9).
# "slim" exclut les outils de compilation C, les headers, la documentation —
# tout ce qui n'est pas nécessaire pour exécuter du Python en production.
# Version 3.9 pour correspondre à la version Python locale — garantit que
# les .pkl générés sont compatibles avec l'API qui tourne aussi sur Python 3.9.
FROM python:3.9-slim

# ── Répertoire de travail ──────────────────────────────────────────────────────
# /app est le répertoire de travail à l'intérieur du conteneur.
# Tous les COPY et CMD s'exécutent depuis /app.
# Sans WORKDIR, les fichiers atterriraient à la racine / — mauvaise pratique
# qui mélange code applicatif et fichiers système Linux.
WORKDIR /app

# ── Installation des dépendances ──────────────────────────────────────────────
# On copie requirements.txt EN PREMIER, avant build_model.py.
# Le cache Docker fonctionne couche par couche. Si requirements.txt n'a pas
# changé entre deux builds, Docker réutilise la couche pip depuis le cache
# sans réinstaller — le build passe de ~60s à ~2s.
# Si on copiait tout d'un coup (COPY . .), la moindre modif de build_model.py
# invaliderait le cache pip et forcerait une réinstallation complète.
COPY requirements.txt .

# pip install avec --no-cache-dir : ne conserve pas les wheels téléchargées
# dans l'image finale (~50-100 Mo économisés inutilement).
RUN pip install --no-cache-dir -r requirements.txt

# ── Copie du script principal ──────────────────────────────────────────────────
# build_model.py copié après requirements.txt pour exploiter le cache Docker.
# Le point final signifie "dans le WORKDIR courant" (/app).
COPY build_model.py .

# ── Commande de démarrage ──────────────────────────────────────────────────────
# Lance build_model.py au démarrage du conteneur.
# Syntaxe JSON (["python", "build_model.py"]) plutôt que string pour que
# Python soit PID 1 — garantit une gestion correcte des signaux d'arrêt
# (SIGTERM, SIGINT) envoyés par Docker lors d'un docker-compose down.
CMD ["python", "build_model.py"]