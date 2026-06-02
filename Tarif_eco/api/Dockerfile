# Dockerfile — conteneur api (Tarif_eco)
# =======================================
# Rôle : construire l'image Docker du conteneur "api".
# Ce conteneur fait tourner l'API FastAPI sur le port 8000.
# Il démarre APRÈS "model-builder" (condition: service_completed_successfully
# dans docker-compose.yml) pour s'assurer que les .pkl sont dans /shared_model/.

# ── Image de base ──────────────────────────────────────────────────────────────
# python:3.9-slim : version allégée (~125 Mo vs ~900 Mo pour python:3.9).
# On évite les outils de compilation C, headers, docs — inutiles en prod.
# Version 3.9 pour correspondre à la version Python locale (compatibilité .pkl).
FROM python:3.9-slim

# ── Répertoire de travail ──────────────────────────────────────────────────────
# Tous les fichiers de l'API atterrissent dans /app à l'intérieur du conteneur.
# Sans WORKDIR, les fichiers iraient à la racine / — mauvaise pratique.
WORKDIR /app

# ── Installation des dépendances ──────────────────────────────────────────────
# requirements.txt copié EN PREMIER pour exploiter le cache Docker par couches.
# Si main.py change mais pas requirements.txt, Docker réutilise la couche pip
# depuis le cache — le build passe de ~60s à ~2s.
COPY requirements.txt .

# --no-cache-dir : ne stocke pas les fichiers de cache pip dans l'image
# (~50-100 Mo économisés inutilement sans cette option).
RUN pip install --no-cache-dir -r requirements.txt

# ── Copie du code source ───────────────────────────────────────────────────────
# Après les dépendances pour maximiser l'usage du cache Docker.
COPY main.py .

# ── Port exposé ───────────────────────────────────────────────────────────────
# EXPOSE 8000 documente que le conteneur écoute sur ce port.
# Le vrai mapping se fait dans docker-compose.yml (ports: "8000:8000").
# Sans ce mapping, le port est accessible dans le réseau Docker
# mais pas depuis ta machine hôte (localhost:8000).
EXPOSE 8000

# ── Commande de démarrage ──────────────────────────────────────────────────────
# Lance uvicorn (serveur ASGI) qui sert l'application FastAPI.
# --host 0.0.0.0 : écoute sur toutes les interfaces réseau du conteneur,
#   pas seulement localhost. Sans ça, l'API n'est pas accessible depuis l'hôte
#   même avec le port mapping docker-compose.
# --port 8000    : port d'écoute interne au conteneur.
# Syntaxe JSON : Python devient PID 1 (gestion correcte des signaux d'arrêt).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]