# Déployer la démo sur Hugging Face Spaces

> Procédure pas-à-pas pour déployer `src/app.py` sur HF Spaces (gratuit, recommandé).

## 1. Créer le Space

1. Aller sur https://huggingface.co/new-space
2. Configurer :
   - **Space name** : `mediboussole`
   - **License** : `cc-by-4.0`
   - **SDK** : `Streamlit`
   - **Hardware** : `CPU basic` (gratuit, suffisant en mode démo sans Ollama)
   - **Visibility** : Public

## 2. README du Space (à coller en haut du README.md du Space)

```yaml
---
title: MediBoussole
emoji: 🩺
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.40.0"
app_file: src/app.py
pinned: false
license: cc-by-4.0
short_description: Triage IMCI hors-ligne pour agents de santé communautaires (Gemma 4)
---
```

## 3. Push du code

```bash
# Cloner le space créé
git clone https://huggingface.co/spaces/<votre-user>/mediboussole hf-space
cd hf-space

# Copier les fichiers du repo principal
cp -r ../TheGemma4GoodHackathon/{src,scripts,docs,data,requirements.txt,LICENSE,.streamlit} .

# Adapter le README avec le YAML ci-dessus
# (HF Spaces lit le YAML pour configurer le Space)

# Commit + push
git add .
git commit -m "Initial deploy of MediBoussole demo"
git push
```

## 4. Comportement attendu

- Le Space lance `src/app.py`
- **Pas d'Ollama disponible** sur HF Spaces (CPU basic) → l'app passe automatiquement en **mode démo** (réponses simulées + RAG réel)
- Le RAG sur l'index FAISS WHO IMCI fonctionne (CPU only)
- Le sidebar affichera `Ollama indisponible — mode démo activé`

## 5. Pour avoir Gemma 4 actif sur le Space

Deux options :

**A — Hardware Inference Endpoint** (HF, payant)

Créer un endpoint dédié pour Gemma 4 sur HF, puis configurer l'app pour appeler l'endpoint au lieu d'Ollama local. Demande modif de code dans `call_gemma4()`.

**B — Ollama externe via tunnel** (gratuit, fragile)

Faire tourner Ollama sur votre Mac M1 Max, exposer via Cloudflare Tunnel ou ngrok, configurer `OLLAMA_HOST` dans les Secrets du Space pour pointer dessus. Marche pour les démos juges, mais dépend de votre Mac qui doit rester allumé.

**C — Mode démo + vrai notebook Kaggle** (RECOMMANDÉ)

Garder le Space en mode démo (RAG + sortie simulée pédagogique) ; pointer les juges vers le notebook Kaggle pour la preuve technique avec Gemma 4 actif. Le mode démo sur Space montre l'UX, le notebook prouve le tech.

## 6. URL finale

`https://huggingface.co/spaces/<votre-user>/mediboussole`

À copier dans la section "Live Demo" du writeup Kaggle.
