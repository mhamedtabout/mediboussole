# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MediBoussole — Notebook de soumission
#
# **The Gemma 4 Good Hackathon** · Tracks visés : Health & Sciences + Ollama + Main Track
#
# Ce notebook démontre une chaîne **100% hors-ligne** pour le triage IMCI assisté par Gemma 4 E4B :
#
# 1. Setup Ollama + Gemma 4 E4B (Q4_K_M)
# 2. Indexation FAISS du corpus WHO IMCI
# 3. Pipeline RAG multilingue
# 4. Triage multimodal (texte + image)
# 5. Function calling structuré (SMS, note clinique)
# 6. Démo end-to-end sur cas synthétique
# 7. Benchmarks (latence, mémoire, qualité)
#
# **Licence** : CC-BY 4.0 (conforme aux règles du hackathon)
# **Reproductibilité** : exécutable sur Mac M1+/Linux/Windows avec Ollama installé.

# %% [markdown]
# ## 1. Setup
#
# Prérequis (à installer hors notebook) :
# ```bash
# # Ollama (https://ollama.com)
# brew install ollama   # macOS
# ollama serve &        # démarrer le serveur
#
# # dépendances Python
# pip install -r requirements.txt
# ```

# %%
import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Any

# Adapter selon tag réel publié de Gemma 4 dans Ollama (vérifier `ollama list` après pull)
MODEL_TAG = os.environ.get("MEDIBOUSSOLE_MODEL", "gemma4:e4b-it-q4_K_M")

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent
DATA_DIR = ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

print(f"Model tag: {MODEL_TAG}")
print(f"Data dir : {DATA_DIR}")


# %%
def shell(cmd: str) -> str:
    """Wrapper subprocess pour les commandes shell."""
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (out.stdout + out.stderr).strip()


# Vérifier Ollama
print(shell("ollama --version"))

# Pull du modèle (commenter si déjà fait)
# print(shell(f"ollama pull {MODEL_TAG}"))

# %% [markdown]
# ## 2. Charger le corpus WHO IMCI
#
# Le **chart booklet** WHO IMCI est public (https://www.who.int/publications/i/item/9789241506823).
# Téléchargez le PDF dans `data/raw/imci_chart_booklet.pdf` puis exécutez la cellule suivante.
#
# Pour les tests offline, on fournit aussi un **corpus minimal codé en dur** ci-dessous.

# %%
# Corpus minimal pour démo immédiate (extraits paraphrasés de WHO IMCI 2014)
DEMO_CORPUS = [
    {
        "id": "imci-fever-001",
        "page": 7,
        "text": (
            "Enfant de 2 mois à 5 ans présentant de la fièvre. "
            "Signes de danger général : incapable de boire ou téter, vomit tout ce qu'il consomme, "
            "convulsions, léthargie ou inconscience. Tout signe = TRIAGE ROUGE, référer immédiatement."
        ),
    },
    {
        "id": "imci-dehydration-002",
        "page": 12,
        "text": (
            "Évaluation de la déshydratation : observer si léthargique ou inconscient, yeux enfoncés, "
            "incapable de boire ou boit difficilement, pli cutané qui s'efface très lentement (≥2s). "
            "Deux signes ou plus = déshydratation sévère = TRIAGE ROUGE."
        ),
    },
    {
        "id": "imci-pneumonia-003",
        "page": 5,
        "text": (
            "Toux ou difficulté respiratoire. Compter la fréquence respiratoire. "
            "Tirage sous-costal ou stridor au repos = pneumonie sévère = TRIAGE ROUGE. "
            "Respiration rapide (≥50/min de 2-12 mois, ≥40/min de 12 mois-5 ans) = pneumonie = "
            "TRIAGE JAUNE, traiter par amoxicilline 40 mg/kg 2x/jour pendant 5 jours."
        ),
    },
    {
        "id": "imci-malnutrition-004",
        "page": 18,
        "text": (
            "Évaluation nutritionnelle : périmètre brachial (MUAC) ruban à 3 couleurs. "
            "MUAC < 11.5 cm (rouge) = malnutrition aiguë sévère = TRIAGE ROUGE. "
            "MUAC 11.5-12.5 cm (jaune) = malnutrition modérée = TRIAGE JAUNE."
        ),
    },
    {
        "id": "imci-diarrhea-005",
        "page": 14,
        "text": (
            "Diarrhée. Demander durée et présence de sang. Diarrhée ≥14 jours = diarrhée persistante. "
            "Sang dans les selles = dysenterie = TRIAGE JAUNE, antibiotique selon protocole national. "
            "Diarrhée + déshydratation sévère = TRIAGE ROUGE."
        ),
    },
    {
        "id": "imci-fever-006",
        "page": 8,
        "text": (
            "Fièvre en zone d'endémie palustre. Test diagnostique rapide (TDR) si disponible. "
            "TDR positif sans signes de danger = paludisme simple = traiter ACT. "
            "Fièvre + signes de danger = paludisme grave = TRIAGE ROUGE, artésunate IM/IV en pré-référence."
        ),
    },
    {
        "id": "imci-vaccination-007",
        "page": 22,
        "text": (
            "Vérifier le calendrier vaccinal à chaque visite. Vaccins essentiels : BCG à la naissance, "
            "Penta 6/10/14 semaines, ROR 9 mois, fièvre jaune 9 mois selon zone. "
            "Saisir l'opportunité de rattrapage à toute consultation."
        ),
    },
    {
        "id": "imci-counsel-008",
        "page": 25,
        "text": (
            "Conseils à la mère lors du retour à domicile : continuer l'allaitement, "
            "donner SRO après chaque selle liquide, signes de retour immédiat = ne peut plus boire, "
            "fièvre qui empire, sang dans les selles, respiration plus difficile."
        ),
    },
]

print(f"Corpus de démo : {len(DEMO_CORPUS)} passages")


# %%
def load_imci_pdf(pdf_path: Path) -> list[dict]:
    """Extrait et chunke le PDF WHO IMCI réel. Retourne une liste de dicts."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        # chunk de ~800 caractères avec recouvrement de 100
        i = 0
        while i < len(text):
            chunk = text[i : i + 800]
            if len(chunk.strip()) > 100:
                chunks.append(
                    {
                        "id": f"imci-p{page_num:03d}-{len(chunks):04d}",
                        "page": page_num,
                        "text": chunk.strip(),
                    }
                )
            i += 700  # overlap 100

    return chunks


# Si vous avez le PDF, décommentez :
# CORPUS = load_imci_pdf(RAW_DIR / "imci_chart_booklet.pdf")
CORPUS = DEMO_CORPUS
print(f"Corpus actif : {len(CORPUS)} passages")

# %% [markdown]
# ## 3. Index d'embeddings (FAISS, multilingue)
#
# Embedder choisi : `paraphrase-multilingual-mpnet-base-v2` (768 dim, 50+ langues).
# Justification : nous voulons accepter des requêtes en wolof, bambara, peul, français.

# %%
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

EMBEDDER_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


class IMCIIndex:
    def __init__(self, embedder_name: str = EMBEDDER_NAME):
        self.embedder = SentenceTransformer(embedder_name)
        self.chunks: list[dict] = []
        self.index: faiss.Index | None = None

    def build(self, corpus: list[dict]) -> None:
        self.chunks = corpus
        texts = [c["text"] for c in corpus]
        embeddings = self.embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=True
        ).astype("float32")
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)  # cosine via produit scalaire (vecteurs normalisés)
        self.index.add(embeddings)

    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        q_emb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        sims, idxs = self.index.search(q_emb, k)
        results = []
        for idx, sim in zip(idxs[0], sims[0]):
            r = dict(self.chunks[idx])
            r["similarity"] = float(sim)
            results.append(r)
        return results


# Construction de l'index
imci = IMCIIndex()
imci.build(CORPUS)

# Test
test_query = "bébé 14 mois, fièvre 39, léthargie depuis 2 jours"
hits = imci.retrieve(test_query, k=3)
for h in hits:
    print(f"[sim={h['similarity']:.3f}] p.{h['page']} — {h['text'][:120]}...")

# %% [markdown]
# ## 4. Triage multimodal (Gemma 4 E4B via Ollama)
#
# Système prompt rigoureux : sortie JSON, citations obligatoires, abstention par défaut.

# %%
import ollama

SYSTEM_PROMPT = """Tu es MediBoussole, un assistant de triage pour agents de santé communautaires.

RÈGLES STRICTES :
1. Tu réponds en français clair, ton calme et professionnel.
2. Tu utilises EXCLUSIVEMENT les protocoles WHO IMCI fournis dans le contexte.
3. Si l'information manque ou est ambiguë, ta réponse est : triage="ROUGE", action="référer immédiatement".
4. Tu ne proposes JAMAIS de dosage que le contexte ne mentionne pas.
5. Tu cites toujours la page IMCI source.

RÈGLE ABSOLUE — SIGNES DE DANGER GÉNÉRAL = TRIAGE ROUGE AUTOMATIQUE :
Si l'enfant présente AU MOINS UN de ces signes, le triage EST ROUGE, sans nuance possible :
- Léthargie, somnolence anormale, ou inconscience
- Convulsions (présentes ou récentes)
- Incapable de boire ou de téter
- Vomit tout ce qu'il consomme
- Tirage sous-costal sévère ou stridor au repos
- Yeux enfoncés + pli cutané qui s'efface lentement (déshydratation sévère)
- MUAC < 11.5 cm (rouge sur le ruban)

Un seul signe = ROUGE = "référer immédiatement". Ne pas dégrader vers JAUNE même si d'autres paramètres sont normaux.

Format de sortie : JSON strict, rien d'autre.
{
  "triage": "ROUGE" | "JAUNE" | "VERT",
  "raison": "...",
  "actions_immediates": ["..."],
  "citation": "page X du protocole WHO IMCI",
  "confiance": 0.0 à 1.0
}
"""


def triage(symptoms: str, image_path: str | None, imci: IMCIIndex, k: int = 4) -> dict:
    """Pipeline RAG + multimodal complet."""
    retrieved = imci.retrieve(symptoms, k=k)

    # Garde-fou par seuil
    max_sim = max(r["similarity"] for r in retrieved) if retrieved else 0.0
    THRESHOLD = 0.40  # à calibrer sur set de validation
    if max_sim < THRESHOLD:
        return {
            "triage": "ROUGE",
            "raison": "Cas hors du périmètre des protocoles indexés (similarité trop faible).",
            "actions_immediates": ["Référer immédiatement au centre de santé"],
            "citation": "garde-fou MediBoussole",
            "confiance": 1.0 - max_sim,
        }

    context = "\n\n".join(
        f"[Source {i+1} | p.{r['page']} | sim={r['similarity']:.2f}]\n{r['text']}"
        for i, r in enumerate(retrieved)
    )

    user_msg = (
        f"Symptômes décrits par l'agent de santé:\n{symptoms}\n\n"
        f"Protocoles WHO IMCI pertinents (récupérés par RAG):\n{context}\n\n"
        f"Produis le triage au format JSON."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": user_msg,
            **({"images": [image_path]} if image_path else {}),
        },
    ]

    response = ollama.chat(
        model=MODEL_TAG,
        messages=messages,
        format="json",
        options={"temperature": 0.0, "num_ctx": 8192},
    )
    try:
        return json.loads(response["message"]["content"])
    except json.JSONDecodeError:
        return {
            "triage": "ROUGE",
            "raison": "Sortie modèle non parsable — référer par sécurité.",
            "actions_immediates": ["Référer immédiatement"],
            "citation": "fallback MediBoussole",
            "confiance": 0.0,
        }


# %% [markdown]
# ## 5. Function calling : SMS + note clinique
#
# Gemma 4 expose le function calling natif. On définit deux outils, le modèle choisit et remplit.

# %%
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "send_referral_sms",
            "description": "Envoie un SMS de référence structuré au centre de santé le plus proche.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_age_months": {"type": "integer", "description": "Âge en mois"},
                    "patient_sex": {"type": "string", "enum": ["M", "F"]},
                    "main_symptoms": {"type": "string"},
                    "triage_color": {"type": "string", "enum": ["ROUGE", "JAUNE", "VERT"]},
                    "estimated_arrival_minutes": {"type": "integer"},
                },
                "required": [
                    "patient_age_months",
                    "patient_sex",
                    "main_symptoms",
                    "triage_color",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_clinical_note",
            "description": "Génère une note clinique au format SOAP.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subjective": {"type": "string"},
                    "objective": {"type": "string"},
                    "assessment": {"type": "string"},
                    "plan": {"type": "string"},
                },
                "required": ["subjective", "objective", "assessment", "plan"],
            },
        },
    },
]


def call_tools(triage_result: dict, original_symptoms: str) -> list[dict]:
    """Demande au modèle quels outils appeler suite au triage."""
    msg = (
        f"Symptômes initiaux : {original_symptoms}\n"
        f"Résultat triage : {json.dumps(triage_result, ensure_ascii=False)}\n\n"
        f"Appelle les outils nécessaires pour notifier le centre de santé et générer la note."
    )
    response = ollama.chat(
        model=MODEL_TAG,
        messages=[{"role": "user", "content": msg}],
        tools=TOOLS,
        options={"temperature": 0.0},
    )
    return response["message"].get("tool_calls", [])


# %% [markdown]
# ## 6. Démo end-to-end
#
# Cas synthétique (aucune donnée patient réelle).

# %%
DEMO_SYMPTOMS = (
    "Bébé fille 14 mois, fièvre 39.2°C depuis 2 jours, léthargie marquée, refus de boire, "
    "fontanelle normale, MUAC 12 cm jaune, respiration 42/min sans tirage."
)
DEMO_IMAGE = None  # remplacer par chemin vers une photo si test multimodal

# Décommenter quand Ollama est prêt :
# print("=== TRIAGE ===")
# result = triage(DEMO_SYMPTOMS, DEMO_IMAGE, imci, k=4)
# print(json.dumps(result, indent=2, ensure_ascii=False))
#
# print("\n=== TOOL CALLS ===")
# calls = call_tools(result, DEMO_SYMPTOMS)
# print(json.dumps(calls, indent=2, ensure_ascii=False))


# %% [markdown]
# ## 7. Benchmarks
#
# Mesures :
# - **Latence** prompt-only et multimodal sur M1 Max
# - **Mémoire** pic d'utilisation
# - **Qualité** triage sur set de cas synthétiques (à étendre)

# %%
def benchmark_latency(query: str, n_runs: int = 5) -> dict:
    """Latence end-to-end (RAG + LLM) sur n_runs."""
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        triage(query, None, imci, k=4)
        latencies.append(time.perf_counter() - t0)

    arr = np.array(latencies)
    return {
        "n_runs": n_runs,
        "mean_s": float(arr.mean()),
        "p50_s": float(np.percentile(arr, 50)),
        "p95_s": float(np.percentile(arr, 95)),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
    }


# Décommenter :
# print(json.dumps(benchmark_latency(DEMO_SYMPTOMS, n_runs=3), indent=2))


# %% [markdown]
# ### Set de validation synthétique
#
# 8 cas couvrant les principales catégories IMCI. Chaque cas a un triage attendu.
# À étendre à 50+ pour la soumission finale.

# %%
EVAL_CASES = [
    {
        "symptoms": "Bébé garçon 4 mois, ne peut plus téter, vomit tout, léthargique.",
        "expected_triage": "ROUGE",
        "category": "danger_general",
    },
    {
        "symptoms": "Fille 18 mois, toux 3 jours, respiration 55/min, tirage sous-costal visible.",
        "expected_triage": "ROUGE",
        "category": "pneumonie_severe",
    },
    {
        "symptoms": "Garçon 30 mois, toux 2 jours, respiration 35/min, alerte, boit normalement.",
        "expected_triage": "VERT",
        "category": "rhume",
    },
    {
        "symptoms": "Fille 24 mois, fièvre 38.5, TDR palu positif, alerte, boit, pas de signe de danger.",
        "expected_triage": "JAUNE",
        "category": "palu_simple",
    },
    {
        "symptoms": "Garçon 8 mois, MUAC 10.5 cm, œdèmes des deux pieds, irritable.",
        "expected_triage": "ROUGE",
        "category": "malnutrition_severe",
    },
    {
        "symptoms": "Fille 36 mois, diarrhée 5 jours, sang dans les selles, alerte.",
        "expected_triage": "JAUNE",
        "category": "dysenterie",
    },
    {
        "symptoms": "Bébé 6 mois, diarrhée 2 jours, yeux enfoncés, pli cutané lent, ne boit plus.",
        "expected_triage": "ROUGE",
        "category": "deshydratation_severe",
    },
    {
        "symptoms": "Garçon 28 mois, vient pour vaccination de rappel, état général normal.",
        "expected_triage": "VERT",
        "category": "routine",
    },
]


def run_eval(cases: list[dict]) -> dict:
    """Exécute le set de validation et retourne précision/rappel par classe."""
    results = []
    for case in cases:
        try:
            pred = triage(case["symptoms"], None, imci, k=4)
            results.append(
                {
                    "case": case["category"],
                    "expected": case["expected_triage"],
                    "predicted": pred.get("triage", "?"),
                    "correct": pred.get("triage") == case["expected_triage"],
                    "confiance": pred.get("confiance", None),
                }
            )
        except Exception as e:
            results.append(
                {"case": case["category"], "error": str(e), "correct": False}
            )

    accuracy = sum(r.get("correct", False) for r in results) / len(results)
    return {"accuracy": accuracy, "n": len(results), "details": results}


# Décommenter :
# eval_report = run_eval(EVAL_CASES)
# print(json.dumps(eval_report, indent=2, ensure_ascii=False))


# %% [markdown]
# ## 8. Conclusion
#
# Ce notebook couvre la chaîne complète **MediBoussole** :
#
# | Composant | État |
# |---|---|
# | Setup Ollama + Gemma 4 E4B | Squelette, à exécuter localement |
# | Corpus WHO IMCI + index FAISS multilingue | Démo + extension PDF prête |
# | Triage RAG + multimodal | Implémenté avec garde-fou de seuil |
# | Function calling (SMS, note SOAP) | Implémenté (schémas JSON) |
# | Benchmarks (latence + qualité) | Squelette, set d'eval à étendre |
#
# **Prochaines étapes** :
# 1. Télécharger le PDF WHO IMCI réel et réindexer
# 2. Étendre le set d'évaluation à 50+ cas
# 3. Calibrer le seuil τ sur set de validation
# 4. Démo Streamlit (`src/app.py`)
# 5. Vidéo de pitch (voir `docs/video-storyboard.md`)
#
# **Repo** : (à compléter avec URL GitHub)
# **Démo live** : (à compléter avec URL Streamlit / HF Spaces)
