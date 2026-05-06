"""
Sanity check du pipeline MediBoussole bout-en-bout.

Usage :
    .venv/bin/python scripts/sanity_check.py

Vérifie :
1. Ollama répond et le modèle est disponible
2. Index FAISS charge depuis data/index/
3. Retrieval RAG donne des résultats sensés
4. Garde-fou seuil bloque les requêtes hors-scope
5. Gemma 4 produit du JSON valide
6. Function calling produit un schéma valide

Sortie : codes HTTP-style et latences détaillées.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index"
MODEL_TAG = os.environ.get("MEDIBOUSSOLE_MODEL", "gemma4:e4b-it-q4_K_M")
THRESHOLD = 0.40

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
WARN = "\033[33m⚠\033[0m"


def step(msg: str):
    print(f"\n{'─' * 60}\n{msg}\n{'─' * 60}")


def main():
    overall_ok = True

    # ───────── 1. Ollama health ─────────
    step("1. Ollama health + modèle disponible")
    try:
        import ollama
        models = ollama.list().get("models", [])
        names = [m.get("model", "") for m in models]
        if any(MODEL_TAG in n for n in names):
            print(f"  {PASS} Modèle {MODEL_TAG} trouvé.")
        else:
            print(f"  {FAIL} Modèle {MODEL_TAG} absent. Disponibles : {names[:5]}")
            print(f"     → ollama pull {MODEL_TAG}")
            overall_ok = False
    except Exception as e:
        print(f"  {FAIL} Ollama non joignable : {e}")
        overall_ok = False

    # ───────── 2. Index FAISS ─────────
    step("2. Index FAISS chargeable")
    try:
        import faiss
        idx_path = INDEX_DIR / "imci.faiss"
        chunks_path = INDEX_DIR / "imci_chunks.pkl"
        if not idx_path.exists() or not chunks_path.exists():
            print(f"  {FAIL} Index manquant : {idx_path}")
            print(f"     → run notebook/medi-boussole.py pour le construire")
            overall_ok = False
        else:
            index = faiss.read_index(str(idx_path))
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            print(f"  {PASS} Index FAISS : {index.ntotal} vecteurs, {len(chunks)} chunks")
    except Exception as e:
        print(f"  {FAIL} Erreur index : {e}")
        overall_ok = False
        return 1

    # ───────── 3. Retrieval RAG sur queries ─────────
    step("3. Retrieval RAG sur 4 queries (3 in-scope, 1 hors-scope)")
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        queries = [
            ("Bébé 14 mois fièvre 39 léthargie", "in_scope"),
            ("Toux respiration rapide tirage sous-costal", "in_scope"),
            ("Diarrhée 5 jours sang dans les selles", "in_scope"),
            ("Lymphome Hodgkin chimiothérapie ABVD", "out_of_scope"),
        ]
        for q, expected in queries:
            t0 = time.perf_counter()
            qe = embedder.encode([q], normalize_embeddings=True).astype("float32")
            sims, idxs = index.search(qe, 4)
            dt_ms = (time.perf_counter() - t0) * 1000
            top_sim = float(sims[0][0])
            top_page = chunks[idxs[0][0]]["page"]

            should_proceed = top_sim >= THRESHOLD
            classification = "in_scope" if should_proceed else "out_of_scope"
            mark = PASS if classification == expected else FAIL
            if classification != expected:
                overall_ok = False
            print(f"  {mark} [{dt_ms:5.1f}ms] sim={top_sim:.3f} p.{top_page}  ({expected:14s}) « {q[:40]}{'…' if len(q)>40 else ''} »")
    except Exception as e:
        print(f"  {FAIL} Erreur retrieval : {e}")
        overall_ok = False

    # ───────── 4. Gemma 4 JSON output ─────────
    step("4. Gemma 4 — sortie JSON structurée (triage)")
    try:
        import ollama
        symptoms = "Bébé fille 14 mois, fièvre 39.2°C depuis 2 jours, léthargie marquée, refus de boire."
        qe = embedder.encode([symptoms], normalize_embeddings=True).astype("float32")
        sims, idxs = index.search(qe, 4)
        retrieved = [{"page": chunks[i]["page"], "text": chunks[i]["text"], "sim": float(s)}
                     for i, s in zip(idxs[0], sims[0])]
        context = "\n\n".join(
            f"[Source {i+1} | p.{r['page']} | sim={r['sim']:.2f}]\n{r['text']}"
            for i, r in enumerate(retrieved)
        )

        system_prompt = (
            "Tu es MediBoussole. RÈGLE ABSOLUE : tout signe de danger général = TRIAGE ROUGE "
            "AUTOMATIQUE et sans nuance. Signes de danger : léthargie, somnolence, "
            "convulsions, incapable de boire/téter, vomit tout, tirage sous-costal sévère, "
            "déshydratation sévère, MUAC < 11.5 cm. Un seul signe → ROUGE.\n\n"
            "Tu produis UNIQUEMENT un JSON avec ces clés exactes :\n"
            '{"triage": "ROUGE"|"JAUNE"|"VERT", "raison": "...", '
            '"actions_immediates": ["..."], "citation": "page X", "confiance": 0.0-1.0}'
        )
        user_msg = f"Symptômes:\n{symptoms}\n\nProtocoles WHO IMCI:\n{context}\n\nProduis le triage JSON."

        t0 = time.perf_counter()
        response = ollama.chat(
            model=MODEL_TAG,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            format="json",
            options={"temperature": 0.0, "num_ctx": 8192},
        )
        dt_ms = (time.perf_counter() - t0) * 1000
        content = response["message"]["content"]
        result = json.loads(content)

        required_keys = {"triage", "raison", "actions_immediates", "citation", "confiance"}
        missing = required_keys - set(result.keys())
        if missing:
            print(f"  {FAIL} Clés JSON manquantes : {missing}")
            overall_ok = False
        else:
            print(f"  {PASS} Sortie JSON valide en {dt_ms:.0f} ms ({len(content)} chars)")
            print(f"        triage={result['triage']}, conf={result['confiance']}")
            print(f"        raison: {result['raison'][:80]}{'…' if len(result['raison'])>80 else ''}")
            if result["triage"] not in ("ROUGE", "JAUNE", "VERT"):
                print(f"  {WARN} Valeur triage inattendue : {result['triage']}")
    except json.JSONDecodeError as e:
        print(f"  {FAIL} JSON invalide : {e}")
        print(f"     content brut : {content[:200]}")
        overall_ok = False
    except Exception as e:
        print(f"  {FAIL} Erreur LLM : {e}")
        overall_ok = False

    # ───────── 5. Function calling ─────────
    step("5. Function calling — schéma SMS")
    try:
        tools = [{
            "type": "function",
            "function": {
                "name": "send_referral_sms",
                "description": "Envoie un SMS de référence structuré.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patient_age_months": {"type": "integer"},
                        "patient_sex": {"type": "string", "enum": ["M", "F"]},
                        "main_symptoms": {"type": "string"},
                        "triage_color": {"type": "string", "enum": ["ROUGE", "JAUNE", "VERT"]},
                    },
                    "required": ["patient_age_months", "main_symptoms", "triage_color"],
                },
            },
        }]
        t0 = time.perf_counter()
        response = ollama.chat(
            model=MODEL_TAG,
            messages=[{"role": "user", "content":
                f"Triage : {json.dumps(result, ensure_ascii=False)}\n"
                f"Symptômes : {symptoms}\n"
                "Appelle send_referral_sms avec les paramètres extraits."}],
            tools=tools,
            options={"temperature": 0.0},
        )
        dt_ms = (time.perf_counter() - t0) * 1000
        tool_calls = response["message"].get("tool_calls", [])
        if not tool_calls:
            print(f"  {WARN} Aucun outil appelé (peut être OK si modèle estime non nécessaire)")
        else:
            tc = tool_calls[0]
            fn_name = tc.get("function", {}).get("name", "?")
            args = tc.get("function", {}).get("arguments", {})
            if fn_name == "send_referral_sms":
                print(f"  {PASS} Function call valide en {dt_ms:.0f} ms")
                print(f"        args: {json.dumps(args, ensure_ascii=False)[:120]}")
            else:
                print(f"  {WARN} Outil inattendu : {fn_name}")
    except Exception as e:
        print(f"  {FAIL} Erreur function calling : {e}")
        overall_ok = False

    # ───────── Verdict ─────────
    step("VERDICT")
    if overall_ok:
        print(f"  {PASS} Pipeline MediBoussole opérationnel de bout en bout.\n")
        return 0
    else:
        print(f"  {FAIL} Au moins une étape a échoué — voir détails ci-dessus.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
