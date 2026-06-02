# Dockerfile — conteneur api
# ============================
# Rôle : construire l'image Docker du conteneur "api".
# Ce conteneur fait tourner l'API FastAPI qui expose le modèle de prédiction
# sur le port 8000. Il démarre APRÈS le conteneur "model-builder" (condition:
# service_completed_successfully dans docker-compose.yml) pour s'assurer que
# les .pkl sont disponibles dans /shared_model/ avant le chargement.

# ── Image de base ──────────────────────────────────────────────────────────────
# Même image que le model-builder pour la cohérence des versions Python.
# python:3.9-slim : ~125 Mo au lieu de ~900 Mo pour python:3.9 — on évite
# les outils de compilation C, les headers, les docs, etc., inutiles en prod.
FROM python:3.9-slim

# ── Répertoire de travail ──────────────────────────────────────────────────────
# Tous les fichiers de l'API atterrissent dans /app à l'intérieur du conteneur.
WORKDIR /app

# ── Installation des dépendances ──────────────────────────────────────────────
# requirements.txt copié EN PREMIER pour exploiter le cache Docker.
# Si on modifie main.py mais pas requirements.txt, Docker réutilise
# la couche pip depuis le cache — le build passe de ~60s à ~2s.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copie du code source ───────────────────────────────────────────────────────
# main.py copié après les dépendances pour optimiser le cache Docker.
COPY main.py .

# ── Port exposé ───────────────────────────────────────────────────────────────
# EXPOSE 8000 documente que le conteneur écoute sur le port 8000.
# C'est informatif — le vrai mapping de port se fait dans docker-compose.yml
# (ports: "8000:8000"). Sans ce mapping, le port serait accessible à l'intérieur
# du réseau Docker mais pas depuis ta machine hôte.
EXPOSE 8000

# ── Commande de démarrage ──────────────────────────────────────────────────────
# Lance uvicorn (serveur ASGI) avec FastAPI.
# --host 0.0.0.0 : écoute sur toutes les interfaces réseau du conteneur,
#   pas seulement localhost. Sans ça, l'API ne serait pas accessible depuis
#   l'extérieur du conteneur même avec le port mapping.
# --port 8000    : port d'écoute à l'intérieur du conteneur.
# Syntaxe JSON pour que Python soit PID 1 (gestion correcte des signaux d'arrêt).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]