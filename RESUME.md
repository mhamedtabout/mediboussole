# Reprise du projet MediBoussole

> Document destiné à vous (ou à une future session Claude Code) pour reprendre exactement où on s'est arrêté.

---

## État actuel (2026-05-03)

### Livrables finalisés (10/10 fichiers prêts)

- ✅ `README.md`, `LICENSE` (CC-BY 4.0), `requirements.txt`
- ✅ `docs/gemma4-schema.svg` — schéma de la famille Gemma 4
- ✅ `docs/architecture.svg` — pipeline MediBoussole
- ✅ `docs/mathematical-foundation.md` — RAG, Q4_K_M, multimodal (math rigoureuse)
- ✅ `docs/video-storyboard.md` — 20 prompts IA pour la vidéo
- ✅ `docs/kaggle-writeup.md` — writeup ≤1500 mots (1304 actuels)
- ✅ `docs/handoff-video-production.md` — guide tournage / montage
- ✅ `notebook/medi-boussole.py` — notebook reproductible (jupytext)
- ✅ `src/app.py` — démo Streamlit
- ✅ `scripts/generate_images.py` — batch génération d'images
- ✅ `scripts/sanity_check.py` — test end-to-end pipeline

### Données prêtes

- ✅ `data/raw/imci_chart_booklet.pdf` (5.1 MB, 80 pages, vrai PDF WHO)
- ✅ `data/index/imci.faiss` (642 KB) + `imci_chunks.pkl` (145 KB) — 214 chunks indexés
- ✅ `.venv/` avec toutes les deps installées (sentence-transformers, faiss, streamlit, ollama, pymupdf, pdfplumber)

### Modèle Gemma 4

- 🔄 **EN COURS DE TÉLÉCHARGEMENT** : `gemma4:e4b-it-q4_K_M` (~9.6 GB)
- Vitesse observée : 2.6 MB/s → ETA ~1 heure
- ⚠️ **Le pull tourne dans la session Claude Code en cours.** Si vous fermez Claude Code, il peut s'arrêter.

---

## Si le pull s'est arrêté → relancer

```bash
# Dans un Terminal natif (pas Claude Code), pour qu'il survive :
ollama pull gemma4:e4b-it-q4_K_M
# Ollama reprend automatiquement là où le téléchargement s'est arrêté
```

## Si le pull a réussi → tester end-to-end

```bash
cd /Users/mhamedtabout/Documents/TheGemma4GoodHackathon

# Vérifier que le modèle est dispo
ollama list | grep gemma4

# Sanity check pipeline complet (RAG + LLM + function calling)
.venv/bin/python scripts/sanity_check.py

# Si tout est ✓, lancer la démo web
.venv/bin/streamlit run src/app.py
# → ouvre http://localhost:8501
```

Sortie attendue du `sanity_check.py` :

```
1. Ollama health + modèle disponible          ✓
2. Index FAISS chargeable                     ✓
3. Retrieval RAG sur 4 queries
   ✓ in_scope « Bébé 14 mois fièvre 39 léthargie » sim=0.62
   ✓ in_scope « Toux respiration tirage »          sim=0.52
   ✓ in_scope « Diarrhée sang dans les selles »    sim=0.65
   ✓ out_of_scope « Lymphome Hodgkin ABVD »        sim=0.36 → abstention ✓
4. Gemma 4 — sortie JSON structurée
   ✓ triage=ROUGE, confiance=...
5. Function calling
   ✓ send_referral_sms({...})
VERDICT : ✓ Pipeline opérationnel
```

---

## Reste à faire avant la deadline (2026-05-18 23:59 UTC)

### Côté humain (15 jours dispo, je ne peux pas faire à votre place)

1. **Générer les 20 images** — `python3 scripts/generate_images.py` (clé API Imagen depuis aistudio.google.com → ~5 min)
2. **Filmer 5 plans authentiques** chez votre famille (scènes 1, 2, 6, 16, 19 de `docs/video-storyboard.md`)
3. **Enregistrer la voix-off** (votre voix ou ElevenLabs / Gemini TTS — script dans le storyboard)
4. **Monter la vidéo** ≤ 3:00 dans DaVinci Resolve ou CapCut (cf. `docs/handoff-video-production.md`)
5. **Sous-titres** FR + EN baked-in
6. **Upload YouTube** non-listé (test) puis public (avant deadline)
7. **Publier le repo GitHub** (push de tout le dossier sauf `.venv/` — penser au `.gitignore`)
8. **Déployer la démo live** sur Streamlit Cloud / HF Spaces / Render
9. **Convertir le notebook** : `jupytext --to notebook notebook/medi-boussole.py` puis publier sur Kaggle
10. **Soumettre le writeup** Kaggle avec liens vidéo + repo + démo + cover image (proposez la scène 19)

### Côté technique (je peux faire si vous me le demandez en revenant)

- [ ] Créer un script de déploiement Streamlit Cloud
- [ ] Étendre le set d'évaluation (8 → 50 cas) dans le notebook
- [ ] Calibrer τ sur le set étendu pour optimiser F2-score
- [ ] Ajouter une cellule de benchmarks reproductibles dans le notebook
- [ ] Préparer un `.gitignore` propre + initialiser le repo Git
- [ ] Rédiger un script de voix-off plus léché (3 min, FR documentaire)
- [ ] Ajouter une page "Mode hors-ligne" dans l'app pour preuve visuelle (mode avion)

---

## Pour reprendre une session avec moi

Lancez Claude Code dans `/Users/mhamedtabout/Documents/TheGemma4GoodHackathon` et dites simplement :

> « Reprends MediBoussole — où on en est ? »

Je relirai ce fichier + ma mémoire et je continuerai exactement où on s'est arrêté.

---

*Dernière mise à jour : 2026-05-03*
