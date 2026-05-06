# Handoff — Production de la vidéo (tâches 4, 5, 6)

> Ce document détaille les **3 actions physiques** que vous devez réaliser vous-même.
> Claude Code (terminal) ne peut ni générer d'images, ni filmer, ni monter une vidéo.

---

## Tâche 4 — Générer les 20 images IA (1 heure)

**Outil recommandé** : [Google AI Studio](https://aistudio.google.com) → Imagen 4 (gratuit, quota généreux).
**Alternative gratuite** : Flux Schnell sur fal.ai · [Pollinations.ai](https://pollinations.ai) (zéro signup).
**Alternative payante** : Midjourney v6 (~10€/mois, qualité supérieure).

### Procédure

1. Ouvrir `docs/video-storyboard.md` dans deux fenêtres : prompts à gauche, app de génération à droite.
2. Pour chaque scène 1 → 20 :
   - Copier-coller le bloc `Prompt IA` dans Imagen
   - Forcer le ratio **16:9** dans les options
   - Générer 4 variantes, garder la meilleure
   - Sauvegarder dans `assets/images/scene-NN.png` (ex: `scene-01.png`)
3. Vérifier la cohérence visuelle (même style, mêmes couleurs ocre/vert) — re-générer si dissonance.

### Conseil de cohérence

Ajoutez ce **suffixe systématique** à chaque prompt pour homogénéiser le style :
```
, photorealistic documentary cinematography style, warm earth tones with medical green accents, 16:9 aspect ratio
```

### Si vous voulez automatiser

L'API Imagen est disponible via Google AI Studio. Petit script Python que vous pouvez lancer :

```python
# scripts/generate_images.py
import google.genai as genai
import re, pathlib

# 1. Coller votre clé API (gratuite via aistudio.google.com)
client = genai.Client(api_key="VOTRE_CLE_API")

# 2. Extraire les 20 prompts du storyboard
storyboard = pathlib.Path("docs/video-storyboard.md").read_text()
prompts = re.findall(r"\*\*Prompt IA\*\* :\n```\n(.*?)\n```", storyboard, re.S)

# 3. Générer les 20 images
out = pathlib.Path("assets/images")
out.mkdir(parents=True, exist_ok=True)
for i, prompt in enumerate(prompts, 1):
    response = client.models.generate_images(
        model="imagen-4.0-generate-001",
        prompt=prompt,
        config={"number_of_images": 1, "aspect_ratio": "16:9"},
    )
    response.generated_images[0].image.save(out / f"scene-{i:02d}.png")
    print(f"Scene {i}/20 done")
```

À lancer : `python scripts/generate_images.py` (~5 minutes total).

---

## Tâche 5 — Filmer 5 plans authentiques chez votre famille (1 demi-journée)

**Pourquoi c'est crucial** : 40% du score est sur l'authenticité (« Impact & Vision »). Les juges détectent les vidéos 100% IA — fades, sans âme. Un plan tourné en vrai, *même au smartphone*, change la perception du jury.

### Plans à tourner sur place

| Scène | Plan | Durée | Difficulté |
|---|---|---|---|
| **1** | Lever du soleil sur la région rurale | 8s | facile (réveil tôt) |
| **2** | Une personne marche sur un chemin (votre famille, dos à la caméra, pas besoin de visage) | 6s | facile |
| **6** | Gros plan sur des yeux fatigués (un acteur bénévole adulte, pas un enfant malade) | 9s | moyen |
| **16** | Une moto qui démarre / s'éloigne sur un chemin | 7s | facile |
| **19** | Portrait au regard caméra, sourire (votre famille, consentement signé) | 12s | moyen |

### Matériel minimum

- **Smartphone** : iPhone 12+ ou Android haut de gamme (4K si possible)
- **Stabilisateur** : main appuyée sur surface, ou DJI Osmo Mobile (~70€)
- **Audio** : non utilisé (la voix-off remplace le son ambiant)
- **Heure** : "golden hour" — 1h après lever ou avant coucher du soleil

### Consentements (obligatoire)

Toute personne identifiable à l'écran doit signer un **release**. Modèle minimaliste FR :

```
Je soussigné(e) [Nom Prénom], autorise [Mhamed Tabout] à utiliser mon image
filmée le [date] à [lieu] dans le cadre du projet MediBoussole, soumis à
The Gemma 4 Good Hackathon (kaggle.com). L'usage est non-commercial et
les images peuvent être publiées sous licence CC-BY 4.0.

Signature : ____________________  Date : ____________________
```

⚠️ **Mineurs** : signature des deux parents obligatoire. **Évitez de filmer des enfants malades**, même avec consentement. Pour la scène "bébé fiévreux" (scène 3), utilisez l'IA générative.

### Réglages caméra rapides

- iPhone : Cinematic Mode 24fps · résolution 4K · auto-focus tap-to-track
- Android : mode Pro · ISO ≤400 · vitesse 1/50 · balance des blancs "ensoleillé"

---

## Tâche 6 — Monter la vidéo (1 demi-journée)

**Outil recommandé** : [DaVinci Resolve](https://www.blackmagicdesign.com/products/davinciresolve) (gratuit, professionnel).
**Alternative simple** : [CapCut Desktop](https://www.capcut.com) (gratuit, plus rapide à apprendre).

### Structure du projet (à créer dans `assets/`)

```
assets/
├── images/             # 20 PNG depuis Imagen (tâche 4)
├── footage/            # 5 clips MP4 depuis votre caméra (tâche 5)
├── audio/
│   ├── voice-over.wav  # Voix-off (à enregistrer ci-dessous)
│   └── music.mp3       # Musique libre (Pixabay / YouTube Audio Library)
├── srt/
│   ├── subs-fr.srt     # Sous-titres français
│   └── subs-en.srt     # Sous-titres anglais
└── final/
    └── mediboussole-3min.mp4
```

### Étapes (DaVinci Resolve)

1. **Créer le projet** : 1920×1080, 30fps, durée 3:00.
2. **Importer** : drag-drop tous les médias dans le Media Pool.
3. **Timeline** :
   - Couche vidéo principale : assembler les 20 plans dans l'ordre du storyboard
   - Couche vidéo secondaire : substituer les plans tournés (1, 2, 6, 16, 19) aux PNG correspondants
   - Couche audio 1 : voix-off
   - Couche audio 2 : musique (volume -18 dB pendant la voix, -12 dB ailleurs)
4. **Mouvements de caméra IA** : ajouter un Ken Burns subtil sur chaque PNG (zoom 100% → 105% sur la durée du plan)
5. **Étalonnage** : applicquer un LUT cohérent (Filmic, ou Resolve "Standard") sur tous les plans pour homogénéité couleur
6. **Texte** :
   - Scène 5 : statistique OMS en grand
   - Scène 8 : "100% offline"
   - Scène 18 : "3,5 millions d'ASC"
   - Scène 20 : carton final (logo + URLs + "CC-BY 4.0")
7. **Sous-titres** : importer les .srt, vérifier le timing
8. **Export** : H.264, 1080p30, débit 10 Mbps, AAC 192k → `mediboussole-3min.mp4`

### Voix-off — comment l'enregistrer

**Option A** : votre voix avec un téléphone collé à 10 cm, dans une chambre tapissée (pas d'écho). Logiciel : [Audacity](https://www.audacityteam.org) gratuit.

**Option B** : voix synthétique professionnelle gratuite via [ElevenLabs](https://elevenlabs.io) (10 min gratuites/mois) ou [Gemini TTS](https://aistudio.google.com) (gratuit).

Le script de voix-off est déjà dans `docs/video-storyboard.md` (champ "Voix-off" sur chaque scène). Total ~ 90 secondes de parole pour 3 minutes de vidéo (le reste = musique + texte écran).

### Musique — sources libres

- [YouTube Audio Library](https://studio.youtube.com/channel/UC.../music) (créer chaîne YouTube vide, accès gratuit)
- [Pixabay Music](https://pixabay.com/music) (CC0)
- [Free Music Archive](https://freemusicarchive.org)

Recherche : "minimal piano", "documentary african", "hopeful uplifting". **Vérifier ContentID YouTube** avant l'upload.

### Sous-titres bilingues

Générer auto puis relire :

```bash
# Whisper local (déjà installé via requirements.txt si vous décommentez la ligne)
whisper assets/audio/voice-over.wav --language fr --output_format srt
# Puis traduire vers EN avec Gemini ou DeepL
```

### Upload YouTube

1. **Privé** d'abord pour test
2. Vérifier sous-titres affichés correctement
3. **Non-listé** pour les juges (ne pas exiger login)
4. Titre : "MediBoussole — Offline IMCI Triage with Gemma 4"
5. Description : copier le résumé de `docs/kaggle-writeup.md` §1-2

---

## Checklist finale avant submission Kaggle

- [ ] **20 images** générées dans `assets/images/`
- [ ] **5 clips tournés** dans `assets/footage/`
- [ ] **Voix-off** enregistrée + transcrite en .srt FR + traduite EN
- [ ] **Vidéo finale** uploadée sur YouTube (non-listé), durée ≤ 3:00
- [ ] **Repo GitHub** publié, README à jour avec liens
- [ ] **Démo live** déployée (Streamlit Cloud / HF Spaces / Render)
- [ ] **Notebook Kaggle** publié avec PDF WHO indexé
- [ ] **Writeup Kaggle** rédigé à partir de `docs/kaggle-writeup.md`
- [ ] **Cover image** (image hero pour le writeup) — utilisez la scène 7 ou 19
- [ ] **Submit** sur kaggle.com/competitions/gemma-4-good-hackathon avant **2026-05-18 23:59 UTC**

---

## Si vous bloquez

Demandez-moi :
- "Génère le script Python pour batch-Imagen" (déjà fait ci-dessus, j'ajuste)
- "Aide-moi à écrire la voix-off" (je peux raffiner le ton/rythme)
- "Ajuste le storyboard pour [contrainte]" (météo, langue, couleur de peau, etc.)
- "Adapte la vidéo si je n'ai que 5 plans IA et 0 plan tourné" (mode tout-IA possible mais moins fort)
