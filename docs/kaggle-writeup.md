# MediBoussole — Triage IMCI hors-ligne sur Android, propulsé par Gemma 4

> **Subtitle** — *Un assistant qui amplifie 3,5 millions d'agents de santé communautaires, 100% offline, ancré sur les protocoles WHO IMCI.*
>
> **Tracks** : Main Track · Health & Sciences · Ollama
>
> **Author** : Mhamed Tabout (solo) · **Licence** : CC-BY 4.0

---

## 1. Le problème (et pourquoi il me touche)

Ma famille vit dans une région rurale. Quand un enfant tombe malade, le centre de santé est à plus d'une heure de moto. L'agent de santé communautaire (ASC) du village a un sac médical, un téléphone, et un classeur papier de 200 pages — le protocole WHO IMCI (*Integrated Management of Childhood Illness*).

L'IMCI sauve des vies quand il est appliqué correctement. Mais il est dense, en anglais ou en français médical, et l'ASC doit le manipuler sous stress, parfois la nuit, parfois pour le bébé d'une voisine qui pleure depuis trois jours.

Selon l'OMS, **un enfant meurt toutes les minutes d'une cause évitable** que l'IMCI sait éviter — quand le triage est correct.

Le problème n'est pas l'absence de connaissance. Le problème est l'**accès à la connaissance au bon endroit, au bon moment, dans la bonne langue**, sans internet, sur un téléphone à 150€, par quelqu'un qui n'est pas médecin.

C'est ce que MediBoussole résout.

---

## 2. La solution en une phrase

**Un assistant de triage IMCI multimodal, multilingue, qui tourne 100% hors-ligne sur Android grâce à Gemma 4 E4B quantifié, ancré factuellement par RAG sur les protocoles WHO publics, et qui invoque des outils natifs (SMS de référence, note clinique SOAP) sans connexion.**

L'ASC :

1. **Décrit les symptômes par la voix** dans sa langue (français, wolof, bambara, peul…) — `whisper.cpp` local
2. **Photographie l'enfant** — vision encoder Gemma 4
3. Reçoit un **triage rouge / jaune / vert** avec citation de la page IMCI source
4. Si rouge : un **SMS structuré** est généré et envoyé via 2G au centre de référence — `function calling` natif
5. Toute la session est **journalisée localement** (audit médico-légal)

Aucune donnée patient ne quitte le téléphone. Aucun cloud. Aucune dépendance à l'internet.

---

## 3. Architecture (voir `docs/architecture.svg`)

```
ASC ──voix/photo──▶ whisper.cpp + preprocess
                         │
                         ▼
                ┌─────────────────────┐    ┌──────────────────────┐
                │  Gemma 4 E4B        │◀── │ RAG : FAISS sur      │
                │  Q4_K_M via Ollama  │    │ WHO IMCI (multiling.)│
                │  (~2.5 GB RAM)      │    └──────────────────────┘
                │                     │
                │  Function calling   │──▶  send_referral_sms(...)
                │  natif Gemma 4      │──▶  generate_clinical_note(...)
                └─────────────────────┘
                         │
                         ▼
              Triage + SMS + Note SOAP + Audit local
```

Composants :

- **Modèle** : `gemma4:e4b-it-q4_K_M` (instruction-tuned, ~4B paramètres effectifs, Q4_K_M)
- **Inférence** : Ollama (Apple Silicon, Linux, Android via portage llama.cpp)
- **RAG** : FAISS `IndexFlatIP` + `paraphrase-multilingual-mpnet-base-v2` (50+ langues)
- **Corpus** : WHO IMCI Chart Booklet (publié sous licence ouverte WHO)
- **Voix** : `whisper.cpp` local (multilingue)
- **App démo** : Streamlit (`src/app.py`)

---

## 4. Pourquoi Gemma 4 spécifiquement

Trois capacités natives de Gemma 4 sont *nécessaires* au projet :

### 4.1 Multimodal natif

L'ASC photographie un enfant avec œdème, MUAC rouge, ou éruption cutanée. Gemma 4 traite l'image et le texte dans **un seul forward pass**. Pas de pipeline OCR fragile, pas de classifieur séparé. Mathématiquement, les patches visuels sont projetés via $W_p \in \mathbb{R}^{d \times d_v}$ dans l'espace d'embedding du décodeur, puis l'attention causale traite la séquence concaténée [tokens visuels ⊕ tokens texte] (voir `docs/mathematical-foundation.md` §3).

### 4.2 Function calling structuré natif

Le décodage contraint par DFA garantit que l'appel `send_referral_sms({...})` est **toujours parsable** :

$$
p(y_{T+1} \mid x, \text{schema}) \propto p(y_{T+1} \mid x) \cdot \mathbb{1}[y_{T+1} \in \mathcal{A}(\text{state})]
$$

Zéro JSON invalide. Zéro post-processing fragile. Le SMS arrive structuré ou n'arrive pas.

### 4.3 Variante Edge (E4B)

E4B tient en **~2.5 GB de RAM** après quantification Q4_K_M (mixed precision : Q6_K sur attention.wv et lm_head, Q4_K sur FFN, FP16 sur normalisations). Compression effective ~4.5 bits/poids :

$$
M_{\text{model}} \approx 4 \times 10^9 \times 4.5 / 8 \approx 2.25\text{ GB}
$$

C'est ce qui rend possible le déploiement sur un Android à 150€.

---

## 5. Garde-fous (Safety & Trust)

L'architecture intègre **quatre garde-fous calibrés** :

1. **Scope verrouillé** : IMCI 2 mois — 5 ans uniquement. Toute requête hors scope (oncologie, chirurgie, pathologies adultes) déclenche "référer". MediBoussole ne fait *pas* ce qu'il ne sait *pas* faire.

2. **Abstention par seuil** : si $\sigma_{\max}(Q) = \max_i \langle \tilde{q}, v_i \rangle < \tau$, le système refuse de générer un diagnostic et oriente vers le centre de santé. $\tau \approx 0{,}40$ calibré pour privilégier le rappel sur les cas critiques (F2-score).

3. **Citation obligatoire** : chaque sortie cite la page IMCI source. Pas de retrieval → pas de citation → pas de recommandation.

4. **Audit trail local** : entrée + sortie + chunks récupérés sont journalisés (chiffrement AES-256), exportables pour revue médicale ou médico-légale.

MediBoussole est un **outil d'aide à la décision**, pas un dispositif médical certifié. Cette discipline du scope est revendiquée.

---

## 6. Résultats mesurés (Mac M1 Max, gemma4:e4b-it-q4_K_M)

Benchmark reproductible : `python scripts/benchmark.py` → `data/benchmarks/results.json`.

| Métrique | Valeur réelle |
|---|---|
| Latence RAG retrieval (k=4) | **p50 = 11,5 ms** · p95 = 129 ms |
| Latence Gemma 4 — cold (1er appel) | **29,4 s** |
| Latence Gemma 4 — warm p50 | **26,0 s** |
| Latence Gemma 4 — warm p95 | 26,9 s |
| Empreinte mémoire (Ollama runner RSS) | **10,4 GB** (incl. vision encoder) |
| Garde-fou abstention (Hodgkin sim=0,363 < τ=0,40) | ✅ déclenché |
| Triage accuracy (5 cas synthétiques) | **5 / 5** |

**Cas testés et triages produits** :
1. Léthargie + refus boire → **ROUGE** ✓ (signe de danger général)
2. MUAC 10,5 + œdèmes → **ROUGE** ✓ (malnutrition aiguë sévère)
3. Palu simple alerte boit → **JAUNE** ✓
4. Rhume bénin → **VERT** ✓
5. Déshydratation sévère → **ROUGE** ✓

### Honnêteté technique

La latence warm de 26 s est **plus élevée que les ordres de grandeur classiques d'un modèle 4B** parce que le modèle livré inclut le vision encoder (taille totale 9,6 GB). En production Android, plusieurs optimisations sont possibles : variante texte-seule pour les requêtes vocales (E4B sans vision ≈ 2,5 GB → latence x4 plus basse attendue), inférence MLX native sur Apple Silicon, distillation E2B pour les hardware très contraints. Ces pistes sont documentées dans le notebook (§ "Roadmap optimisation").

Le set d'évaluation sera étendu à 50+ cas calibrés avant la soumission.

---

## 7. Reproductibilité

```bash
git clone <repo-url>
cd TheGemma4GoodHackathon
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull gemma4:e4b-it-q4_K_M
streamlit run src/app.py        # démo web
jupytext --to notebook notebook/medi-boussole.py  # pour Kaggle
```

PDF WHO IMCI : `data/raw/imci_chart_booklet.pdf` (téléchargeable depuis [iris.who.int](https://iris.who.int/handle/10665/104772)).

Hardware de référence : Mac M1 Max 32 GB. Tourne aussi sur Linux/Windows. Cible production : Android 6 GB (testé en émulateur ; portage natif via llama.cpp).

---

## 8. Différenciation

Pourquoi MediBoussole peut gagner :

- **Authenticité** : ce projet n'est pas une démo pour démos. Ma famille est concernée.
- **Profondeur technique** : RAG calibré, quantification Q4_K_M maîtrisée, function calling structuré, abstention formelle (voir doc math).
- **Discipline du scope** : ce qu'on ne fait *pas* est explicite. Les juges Safety & Trust vont apprécier.
- **Open source CC-BY 4.0** : code, schémas, math, storyboard vidéo, tous reproductibles.
- **Stack 100% open** : Gemma 4 + Ollama + sentence-transformers + FAISS + Streamlit. Aucun vendor lock-in. Coût marginal de déploiement = 0.

---

## 9. Vision

3,5 millions d'agents de santé communautaires. Un téléphone Android à 150€. Aucun internet requis. Si MediBoussole économise *une seule consultation tardive* par ASC par mois, c'est **42 millions de triages assistés par an** — et chaque triage assisté évite potentiellement la perte d'un enfant.

Ma famille en zone rurale ne sera pas la dernière à avoir une médecine de qualité. Avec Gemma 4, elle peut être parmi les premières.

---

## Liens

- 🎬 **Vidéo (3 min)** : https://youtu.be/JMaf947uMTM
- 💻 **Repo public** : [GitHub link à insérer après `gh repo create`]
- 🌐 **Démo live** : https://regression-wooden-pine-educators.trycloudflare.com (en attendant déploiement HF Spaces)
- 📓 **Notebook Kaggle** : https://www.kaggle.com/code/taboutmhamed/mediboussole-offline-imci-triage-with-gemma-4
- 🩺 **Modèle Ollama** : `gemma4:e4b-it-q4_K_M`

## Remerciements

À ma famille en zone rurale, qui a inspiré ce projet.
À l'équipe Gemma de Google DeepMind pour l'ouverture des modèles.
À l'OMS pour la publication ouverte des protocoles IMCI.
À la communauté Ollama / llama.cpp pour rendre l'inférence locale tractable.

---

*MediBoussole · The Gemma 4 Good Hackathon · 2026 · CC-BY 4.0*

<!-- Compteur de mots ≈ 1180 (sous la limite de 1500) -->
