# Dockerfile — conteneur model-builder
# =====================================
# Rôle : construire l'image Docker du conteneur "model-builder".
# Ce conteneur a une seule responsabilité : copier les .pkl dans le volume
# partagé pour que l'API puisse les lire. Il s'exécute une fois, termine,
# et le conteneur "api" prend le relais (condition: service_completed_successfully).

# ── Image de base ──────────────────────────────────────────────────────────────
# python:3.9-slim : version allégée de Python 3.9 (sans les outils de compilation,
# les headers C, etc.). "slim" est 3x plus légère que l'image standard python:3.9
# (~125 Mo vs ~900 Mo), ce qui accélère le pull et le build.
# On choisit 3.9 pour correspondre à la version Python utilisée localement
# — garantit la compatibilité des .pkl sérialisés.
FROM python:3.9-slim

# ── Répertoire de travail ──────────────────────────────────────────────────────
# Tous les COPY et CMD suivants s'exécutent depuis /app à l'intérieur du conteneur.
# Sans WORKDIR, les fichiers atterriraient à la racine / — mauvaise pratique
# qui mélange les fichiers applicatifs avec les fichiers système Linux.
WORKDIR /app

# ── Installation des dépendances ──────────────────────────────────────────────
# On copie requirements.txt EN PREMIER, avant le code source.
# Raison : le cache Docker fonctionne couche par couche. Si requirements.txt
# n'a pas changé, Docker réutilise la couche d'installation depuis le cache
# sans relancer pip — le build passe de 60s à 1s. Si on copiait tout d'un coup,
# la moindre modification de build_model.py forcerait une réinstallation complète.
COPY requirements.txt .

# --no-cache-dir : ne stocke pas les fichiers de cache pip dans l'image.
# Sans cette option, pip garde les wheels téléchargées (~50-100 Mo supplémentaires)
# qui ne servent à rien une fois l'installation terminée.
RUN pip install --no-cache-dir -r requirements.txt

# ── Copie du script principal ──────────────────────────────────────────────────
# On copie build_model.py après les dépendances pour profiter du cache Docker.
# Le . final signifie "dans le WORKDIR courant" (/app).
COPY build_model.py .

# ── Commande de démarrage ──────────────────────────────────────────────────────
# CMD définit ce qui s'exécute quand le conteneur démarre.
# Syntaxe JSON (["python", "build_model.py"]) plutôt que string ("python build_model.py")
# pour éviter un shell intermédiaire — le process Python devient directement PID 1,
# ce qui garantit une gestion correcte des signaux d'arrêt (SIGTERM, SIGINT).
CMD ["python", "build_model.py"]
