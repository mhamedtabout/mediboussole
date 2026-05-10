# MediBoussole

> **Assistant de triage hors-ligne pour agents de santé communautaires.**
> Soumission pour **The Gemma 4 Good Hackathon** — Kaggle / Google DeepMind, 2026.

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)
![Stack](https://img.shields.io/badge/stack-Gemma%204%20E4B%20%2B%20Ollama-2e7d32)
![Offline](https://img.shields.io/badge/offline-100%25-blue)

---

## Le problème

3,5 millions d'agents de santé communautaires (ASC) gèrent les soins de première ligne en zone rurale avec des **protocoles papier** de 200 pages. L'internet est rare, les médecins sont à des heures de route, et un mauvais triage tue.

D'après l'OMS, un enfant meurt toutes les minutes d'une cause évitable que l'IMCI (*Integrated Management of Childhood Illness*) sait éviter — quand les protocoles sont appliqués correctement.

## La solution

Un assistant qui tourne **100 % hors-ligne** sur un téléphone Android à 150 € :

- 🎤 **Voix multilingue** (français, wolof, bambara, peul) → `whisper.cpp` local
- 📷 **Photo** de l'enfant (signes cliniques, MUAC, lésions) → vision encoder Gemma 4
- 🔍 **RAG** ancré sur les protocoles WHO IMCI publics (CC-BY)
- 🛡️ **Garde-fous calibrés** : abstention si similarité < seuil → "référer immédiatement"
- 📨 **Function calling natif** : SMS de référence + note clinique SOAP auto-générés
- 🔒 **Aucune donnée ne quitte le téléphone**

Modèle : **Gemma 4 E4B**, quantifié **Q4_K_M** (~2,3 GB), servi via **Ollama**.

## Tracks visés

- 🏆 **Main Track** (Impact & Vision · Storytelling · Tech depth)
- 🩺 **Health & Sciences** (Impact Track)
- 🧠 **Ollama** (Special Technology Track)

## Liens publics

- 🎬 **Vidéo (3 min)** : https://youtu.be/JMaf947uMTM
- 🌐 **Démo live** : https://regression-wooden-pine-educators.trycloudflare.com

## Architecture en une image

Voir `docs/architecture.svg` pour le schéma complet, et `docs/gemma4-schema.svg` pour la famille de modèles Gemma 4.

## Structure du dépôt

```
TheGemma4GoodHackathon/
├── README.md                         # ce fichier
├── LICENSE                           # CC-BY 4.0
├── requirements.txt                  # dépendances Python
├── docs/
│   ├── gemma4-schema.svg             # schéma de la famille Gemma 4
│   ├── architecture.svg              # pipeline MediBoussole complet
│   ├── mathematical-foundation.md    # RAG, Q4_K_M, fusion multimodale (math rigoureuse)
│   └── video-storyboard.md           # 20 plans + prompts IA pour la vidéo
├── notebook/
│   └── medi-boussole.py              # notebook reproductible (jupytext format)
├── src/
│   ├── app.py                        # démo Streamlit (web app)
│   └── ...                           # modules de l'app
└── data/
    ├── raw/                          # PDF WHO IMCI à télécharger
    └── index/                        # index FAISS persisté
```

## Démarrage rapide

### Prérequis

- Python ≥ 3.10
- [Ollama](https://ollama.com) installé et en cours d'exécution
- Mac M-series, Linux ou Windows (CPU ou GPU)

### Installation

```bash
git clone <ce-repo>
cd TheGemma4GoodHackathon

# Dépendances Python
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Modèle Gemma 4 (E4B instruction-tuned, Q4_K_M — ~2.5 GB)
ollama pull gemma4:e4b-it-q4_K_M

# Optionnel : variante MLX pour Apple Silicon (perfs natives M-series)
# ollama pull gemma4:e4b-mlx-bf16
```

### Notebook (Kaggle / local)

```bash
# Convertir le .py jupytext en .ipynb
jupytext --to notebook notebook/medi-boussole.py
jupyter notebook notebook/medi-boussole.ipynb
```

### App web (démo live)

```bash
streamlit run src/app.py
# Ouvre http://localhost:8501
```

### Corpus WHO IMCI (optionnel mais recommandé)

```bash
# Télécharger le chart booklet officiel
curl -o data/raw/imci_chart_booklet.pdf \
  https://www.who.int/publications/i/item/9789241506823
```

> ⚠️ Vérifier l'URL exacte sur le site WHO. Le chart booklet est sous licence ouverte WHO.

## Reproductibilité

Tous les artefacts (notebook, app, schémas, math) sont versionnés et exécutables.
Hardware de référence pour les benchmarks : **Mac M1 Max 32 GB**, macOS 14+.
Pour reproduire sur Kaggle (CPU ou T4) : exécuter le notebook tel quel ; Ollama doit tourner via les *Kaggle Datasets* qui hébergent le binaire (voir `notebook/medi-boussole.py` §1).

## Sécurité, scope et éthique

MediBoussole est un **outil d'aide à la décision**. Il ne remplace pas l'agent de santé ni le médecin référent.

**Scope verrouillé** : IMCI, enfants 2 mois — 5 ans. Toute requête hors scope (oncologie, chirurgie, pathologies adultes) déclenche une réponse "référer". Voir `docs/architecture.svg` panneau "Garde-fous" pour le détail des 4 garde-fous calibrés.

**Données patient** : aucune donnée patient réelle n'est utilisée dans ce dépôt. Les cas dans le notebook et l'app sont synthétiques.

## Licence

- **Code, schémas, écrits** : [CC-BY 4.0](LICENSE) (conforme aux règles du hackathon)
- **Modèle Gemma 4** : licence Gemma de Google ([conditions](https://ai.google.dev/gemma/terms))
- **Corpus WHO IMCI** : licence ouverte WHO ([source](https://www.who.int/publications/i/item/9789241506823))
- **Embeddings multilingues** : `paraphrase-multilingual-mpnet-base-v2` sous Apache 2.0

## Citer ce travail

Si vous reprenez ce projet :

```bibtex
@misc{mediboussole2026,
  title  = {MediBoussole: Offline triage assistant for community health workers, powered by Gemma 4},
  author = {Tabout, Mhamed},
  year   = {2026},
  note   = {The Gemma 4 Good Hackathon submission},
  url    = {https://kaggle.com/competitions/gemma-4-good-hackathon}
}
```

## Remerciements

À ma famille en zone rurale, pour qui ce projet a un sens concret.
À l'équipe Gemma de Google DeepMind pour l'ouverture des modèles.
À l'OMS et aux ministères de la santé qui publient leurs protocoles en ouvert.
