"""
Benchmark formel du pipeline MediBoussole pour le writeup Kaggle.

Mesure :
- Latence retrieval RAG (n_runs)
- Latence Gemma 4 first-call (cold) vs warm
- Latence end-to-end p50 / p95
- Taille mémoire approchée (via ps)

Sortie JSON dans data/benchmarks/results.json + console résumé.
"""

from __future__ import annotations

import json
import os
import pickle
import statistics
import subprocess
import time
from pathlib import Path

import faiss
import ollama
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index"
OUT_DIR = ROOT / "data" / "benchmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_TAG = os.environ.get("MEDIBOUSSOLE_MODEL", "gemma4:e4b-it-q4_K_M")

EVAL_QUERIES = [
    "Bébé fille 14 mois, fièvre 39.2°C depuis 2 jours, léthargie marquée, refus de boire, MUAC 12 cm jaune, respiration 42/min sans tirage.",
    "Garçon 8 mois, MUAC 10.5 cm, œdèmes des deux pieds, irritable.",
    "Fille 24 mois, fièvre 38.5, TDR palu positif, alerte, boit, pas de signe de danger.",
    "Garçon 30 mois, toux 2 jours, respiration 35/min, alerte, boit normalement.",
    "Bébé 6 mois, diarrhée 2 jours, yeux enfoncés, pli cutané lent, ne boit plus.",
]

SYSTEM_PROMPT = """Tu es MediBoussole. RÈGLE ABSOLUE : tout signe de danger général = TRIAGE ROUGE AUTOMATIQUE.
Signes de danger : léthargie, somnolence, convulsions, incapable de boire/téter, vomit tout,
tirage sous-costal sévère, déshydratation sévère, MUAC < 11.5 cm. Un seul signe → ROUGE.

Tu produis UNIQUEMENT un JSON :
{"triage": "ROUGE"|"JAUNE"|"VERT", "raison": "...", "actions_immediates": ["..."], "citation": "page X", "confiance": 0.0-1.0}"""


def percentile(values: list[float], p: float) -> float:
    """Percentile 0-100 sans numpy."""
    s = sorted(values)
    if not s:
        return 0.0
    idx = (len(s) - 1) * p / 100
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def get_ollama_memory() -> dict:
    """Approximation mémoire via ps."""
    try:
        out = subprocess.check_output(
            ["ps", "-A", "-o", "rss,command"], text=True
        )
        for line in out.splitlines():
            if "ollama" in line.lower() and "runner" in line.lower():
                rss_kb = int(line.strip().split()[0])
                return {"ollama_runner_rss_mb": round(rss_kb / 1024, 1)}
        for line in out.splitlines():
            if "ollama" in line.lower() and "serve" in line.lower():
                rss_kb = int(line.strip().split()[0])
                return {"ollama_serve_rss_mb": round(rss_kb / 1024, 1)}
    except Exception:
        pass
    return {}


def main():
    print(f"\n{'═' * 60}")
    print(f"MediBoussole — Benchmark formel")
    print(f"Modèle : {MODEL_TAG}")
    print(f"{'═' * 60}\n")

    # Charger l'index + embedder
    print("Chargement index FAISS + embedder...")
    index = faiss.read_index(str(INDEX_DIR / "imci.faiss"))
    with open(INDEX_DIR / "imci_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    embedder = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    print(f"  → {index.ntotal} vecteurs indexés\n")

    # 1. Latence retrieval seul
    print("1. Latence RAG retrieval (k=4) sur 5 queries × 5 runs")
    retrieval_lat = []
    for q in EVAL_QUERIES:
        for _ in range(5):
            t0 = time.perf_counter()
            qe = embedder.encode([q], normalize_embeddings=True).astype("float32")
            index.search(qe, 4)
            retrieval_lat.append((time.perf_counter() - t0) * 1000)
    print(f"   p50 = {percentile(retrieval_lat, 50):6.1f} ms")
    print(f"   p95 = {percentile(retrieval_lat, 95):6.1f} ms")
    print(f"   max = {max(retrieval_lat):6.1f} ms\n")

    # 2. Latence Gemma 4 (cold + warm)
    print("2. Latence Gemma 4 (1 cold-call + 4 warm-calls)")
    llm_lat = []
    triage_distrib = {"ROUGE": 0, "JAUNE": 0, "VERT": 0, "?": 0}

    for i, q in enumerate(EVAL_QUERIES):
        # RAG
        qe = embedder.encode([q], normalize_embeddings=True).astype("float32")
        sims, idxs = index.search(qe, 4)
        context = "\n\n".join(
            f"[Source {j+1} | p.{chunks[ix]['page']} | sim={float(sims[0][j]):.2f}]\n{chunks[ix]['text']}"
            for j, ix in enumerate(idxs[0])
        )
        user_msg = f"Symptômes: {q}\n\nProtocoles WHO IMCI:\n{context}\n\nProduis le triage JSON."

        # LLM
        t0 = time.perf_counter()
        try:
            resp = ollama.chat(
                model=MODEL_TAG,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                format="json",
                options={"temperature": 0.0, "num_ctx": 8192},
            )
            dt_ms = (time.perf_counter() - t0) * 1000
            llm_lat.append(dt_ms)

            try:
                result = json.loads(resp["message"]["content"])
                triage_distrib[result.get("triage", "?")] = (
                    triage_distrib.get(result.get("triage", "?"), 0) + 1
                )
            except (json.JSONDecodeError, KeyError):
                triage_distrib["?"] += 1
            tag = "cold" if i == 0 else "warm"
            print(f"   query {i+1} ({tag}) : {dt_ms:7.1f} ms  → {result.get('triage', '?')}")
        except Exception as e:
            print(f"   query {i+1} : ERROR {e}")

    if llm_lat:
        cold_lat = llm_lat[0]
        warm_lat = llm_lat[1:]
        print(f"\n   cold = {cold_lat:7.1f} ms")
        if warm_lat:
            print(f"   warm p50 = {percentile(warm_lat, 50):6.1f} ms")
            print(f"   warm p95 = {percentile(warm_lat, 95):6.1f} ms")
        print(f"   distribution triage : {triage_distrib}")

    # 3. Memory
    print("\n3. Empreinte mémoire (approximée)")
    mem = get_ollama_memory()
    for k, v in mem.items():
        print(f"   {k} : {v} MB")

    # Sauver les résultats
    results = {
        "model": MODEL_TAG,
        "n_queries": len(EVAL_QUERIES),
        "retrieval_ms": {
            "p50": percentile(retrieval_lat, 50),
            "p95": percentile(retrieval_lat, 95),
            "max": max(retrieval_lat),
        },
        "llm_ms": {
            "cold": llm_lat[0] if llm_lat else None,
            "warm_p50": percentile(llm_lat[1:], 50) if len(llm_lat) > 1 else None,
            "warm_p95": percentile(llm_lat[1:], 95) if len(llm_lat) > 1 else None,
        },
        "triage_distribution": triage_distrib,
        "memory_mb": mem,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path = OUT_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n→ Résultats : {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
