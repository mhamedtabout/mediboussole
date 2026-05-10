# Publication — guide étape par étape

> Toutes les actions ici nécessitent **votre authentification** (GitHub, YouTube, Kaggle…).
> Je ne peux pas les faire à votre place. Mais voici les commandes exactes.
>
> **Temps estimé total : 30-45 minutes.**

---

## 1. Push sur GitHub (10 min)

### A. Authentification (une seule fois)

```bash
gh auth login
```

Choisir :
- **GitHub.com**
- **HTTPS** (pas SSH, plus simple)
- **Login with a web browser** → un code à coller s'affiche, le navigateur s'ouvre

Une fois authentifié, vérifier :
```bash
gh auth status
```

### B. Créer + push le repo

```bash
cd /Users/mhamedtabout/Documents/TheGemma4GoodHackathon
gh repo create mediboussole \
  --public \
  --source=. \
  --remote=origin \
  --push \
  --description "Offline IMCI triage assistant powered by Gemma 4. The Gemma 4 Good Hackathon 2026."
```

À la fin, l'URL de votre repo s'affiche, copiez-la → vous en aurez besoin pour le writeup Kaggle.

### C. Activer GitHub Pages pour la cover (optionnel)

```bash
# La cover.png sera accessible via raw GitHub :
# https://raw.githubusercontent.com/<vous>/mediboussole/main/assets/cover.png
```

---

## 2. Upload de la vidéo sur YouTube (5 min)

YouTube n'a pas de bonne CLI gratuite, c'est manuel.

1. Aller sur https://studio.youtube.com
2. Se connecter avec votre compte Google
3. Cliquer **CRÉER → Mettre une vidéo en ligne**
4. Choisir `assets/final/mediboussole-3min.mp4`
5. Remplir :
   - **Titre** : `MediBoussole — Offline IMCI Triage with Gemma 4`
   - **Description** : copier le résumé du writeup (paragraphes 1 et 2 de `docs/kaggle-writeup.md`)
   - **Vignette** : uploader `assets/cover.png`
   - **Public** : **Non listé** d'abord (test), puis **Public** avant la deadline
   - **Pas pour les enfants** (Made for kids = NON, pas un dispositif jouet)
   - **Sous-titres** : ajouter `assets/srt/subtitles-fr.srt` ET `assets/srt/subtitles-en.srt`
   - **Catégorie** : Science et technologie
6. Copier l'URL → besoin pour writeup Kaggle

⚠️ **Important** : les juges vérifient que la vidéo est accessible **sans login**. "Non listé" = OK, "Privé" = NON.

---

## 3. Déployer la démo live sur Hugging Face Spaces (10 min)

Recommandé : Hugging Face Spaces, gratuit, simple.

### A. Créer un compte / Space

1. https://huggingface.co/join (si pas de compte)
2. Créer un token : https://huggingface.co/settings/tokens (rôle : `write`)
3. Aller sur https://huggingface.co/new-space
4. Configurer :
   - **Space name** : `mediboussole`
   - **License** : `cc-by-4.0`
   - **SDK** : Streamlit
   - **Hardware** : CPU basic (gratuit)
   - **Visibility** : Public

### B. Push du code

```bash
cd /tmp  # ou un autre dossier de travail
huggingface-cli login --token <VOTRE_TOKEN_HF>

git clone https://huggingface.co/spaces/<VOTRE_USERNAME_HF>/mediboussole hf-space
cd hf-space

# Copier les fichiers nécessaires depuis le repo principal
cp -r ~/Documents/TheGemma4GoodHackathon/{src,scripts,docs,data,requirements.txt,LICENSE,.streamlit,assets/cover.png} .

# Créer le README avec metadata YAML obligatoire
cat > README.md <<'EOF'
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

# MediBoussole

Voir le code source complet : <URL_GITHUB>
EOF

git add . && git commit -m "Initial deploy of MediBoussole demo"
git push
```

URL finale : `https://huggingface.co/spaces/<vous>/mediboussole`

⚠️ Sur HF Spaces gratuit, **Ollama n'est pas disponible** (pas de daemon). L'app passera en mode démo (RAG réel + sortie LLM simulée). Indiquez ce dans la description du Space que la démo Live montre l'UX et le RAG ; la preuve technique avec Gemma 4 actif est dans le notebook Kaggle.

---

## 4. Soumettre sur Kaggle (10 min)

### A. Convertir le notebook au format Kaggle

```bash
cd /Users/mhamedtabout/Documents/TheGemma4GoodHackathon
.venv/bin/jupytext --to notebook notebook/medi-boussole.py
# Crée notebook/medi-boussole.ipynb
```

### B. Uploader le notebook sur Kaggle

1. https://www.kaggle.com/competitions/gemma-4-good-hackathon → **New Notebook**
2. **File → Upload Notebook** → choisir `notebook/medi-boussole.ipynb`
3. Settings → Internet : **On** (nécessaire pour Ollama si vous le branchez)
4. Settings → Accelerator : **GPU T4** (gratuit, 30h/semaine)
5. Lancer un Run pour valider, puis **Save Version → Publish**
6. Copier l'URL du notebook publié

### C. Créer le Writeup

1. Sur la page de la compétition → **New Writeup**
2. **Title** : `MediBoussole — Offline IMCI Triage with Gemma 4`
3. **Subtitle** : `Un assistant qui amplifie 3,5 millions d'ASC, 100% offline, ancré sur WHO IMCI`
4. **Track** : sélectionner **Health & Sciences** (et Main Track si possible)
5. **Cover image** : uploader `assets/cover.png` (1920×1080, 479 KB)
6. **Body** : copier-coller le contenu de `docs/kaggle-writeup.md`
   - **Bien remplir les liens à la fin** :
     - 🎬 Vidéo YouTube : `https://youtu.be/...`
     - 💻 Repo GitHub : `https://github.com/<vous>/mediboussole`
     - 🌐 Démo live : `https://huggingface.co/spaces/<vous>/mediboussole`
     - 📓 Notebook Kaggle : `https://www.kaggle.com/code/<vous>/...`
7. **Project Links / Attachments** :
   - Vidéo (lien YouTube)
   - Notebook (lien Kaggle)
   - Repo (lien GitHub)
   - Démo live (lien HF Spaces)
8. **Save** puis **SUBMIT** (le bouton apparaît en haut à droite)

⚠️ **CRUCIAL** : "Save" ne suffit pas. Vous devez cliquer **SUBMIT** avant la deadline (2026-05-18 23:59 UTC).

---

## Checklist finale avant submission

- [ ] `git push` GitHub réussi, repo public
- [ ] Vidéo uploadée sur YouTube en non-listé puis public, sous-titres FR + EN ajoutés
- [ ] HF Space déployé et accessible sans login
- [ ] Notebook Kaggle publié (Save Version)
- [ ] Writeup Kaggle créé avec cover + body + 4 liens
- [ ] **Bouton SUBMIT cliqué**
- [ ] Vérifier sur https://www.kaggle.com/competitions/gemma-4-good-hackathon/submissions que votre soumission apparaît

## Si quelque chose casse

- HF Space en erreur : `huggingface-cli logs <user>/mediboussole` pour voir les logs
- GitHub rejette le push : vérifier `gh auth status`, peut-être ré-auth
- Kaggle notebook ne s'exécute pas : enlever les imports qui ne marchent pas sur Kaggle (Ollama notamment), basculer en mode "démo simulée" pour la cellule de démo
- Vidéo YouTube refusée : vérifier le ContentID musical (si vous avez ajouté de la musique externe)

---

*Bon courage pour la deadline.* 🚀
