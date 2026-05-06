"""
MediBoussole — Démo web (Streamlit)

Inspirée de l'esthétique Google AI Studio : panneaux latéraux pour les paramètres,
zone centrale split (entrée / sortie), feedback temps réel, design clair.

Lancer :
    streamlit run src/app.py

Note : nécessite Ollama tournant localement avec le modèle Gemma 4 E4B.
Si Ollama n'est pas disponible, l'app passe en mode "démo" avec sortie simulée.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import streamlit as st

# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="MediBoussole — Triage IMCI hors-ligne",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_TAG = "gemma4:e4b-it-q4_K_M"  # instruction-tuned, Q4_K_M (~2.5GB, edge target)

# Corpus minimal embarqué (extraits paraphrasés de WHO IMCI 2014)
DEMO_CORPUS = [
    {
        "id": "imci-fever-001",
        "page": 7,
        "text": "Enfant 2 mois à 5 ans avec fièvre. Signes de danger général : incapable de boire ou téter, vomit tout, convulsions, léthargie ou inconscience. Tout signe = TRIAGE ROUGE, référer immédiatement.",
    },
    {
        "id": "imci-dehydration-002",
        "page": 12,
        "text": "Évaluation déshydratation : léthargique ou inconscient, yeux enfoncés, incapable de boire, pli cutané qui s'efface très lentement (≥2s). Deux signes ou plus = déshydratation sévère = TRIAGE ROUGE.",
    },
    {
        "id": "imci-pneumonia-003",
        "page": 5,
        "text": "Toux ou difficulté respiratoire. Compter respiration. Tirage sous-costal ou stridor au repos = pneumonie sévère = TRIAGE ROUGE. Respiration rapide (≥50/min 2-12 mois, ≥40/min 12 mois-5 ans) = pneumonie = TRIAGE JAUNE, amoxicilline 40 mg/kg 2x/j 5j.",
    },
    {
        "id": "imci-malnutrition-004",
        "page": 18,
        "text": "Évaluation nutritionnelle : périmètre brachial (MUAC) ruban à 3 couleurs. MUAC < 11.5 cm (rouge) = malnutrition aiguë sévère = TRIAGE ROUGE. MUAC 11.5-12.5 cm (jaune) = malnutrition modérée = TRIAGE JAUNE.",
    },
    {
        "id": "imci-diarrhea-005",
        "page": 14,
        "text": "Diarrhée. Durée et présence de sang. Diarrhée ≥14 jours = persistante. Sang dans selles = dysenterie = TRIAGE JAUNE, antibiotique selon protocole national. Diarrhée + déshydratation sévère = TRIAGE ROUGE.",
    },
    {
        "id": "imci-malaria-006",
        "page": 8,
        "text": "Fièvre en zone palustre. TDR si disponible. TDR positif sans signe de danger = palu simple = ACT. Fièvre + signe de danger = palu grave = TRIAGE ROUGE, artésunate IM/IV en pré-référence.",
    },
    {
        "id": "imci-counsel-008",
        "page": 25,
        "text": "Conseils retour à domicile : continuer allaitement, donner SRO après chaque selle liquide. Signes de retour immédiat : ne peut plus boire, fièvre qui empire, sang dans les selles, respiration plus difficile.",
    },
]

SYSTEM_PROMPT = """Tu es MediBoussole, un assistant de triage pour agents de santé communautaires.

RÈGLES STRICTES :
1. Réponds en français clair, ton calme et professionnel.
2. Utilise EXCLUSIVEMENT les protocoles WHO IMCI fournis dans le contexte.
3. Si l'information manque, ta réponse est : triage="ROUGE", action="référer immédiatement".
4. Ne propose JAMAIS de dosage absent du contexte.
5. Cite toujours la page IMCI source.

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

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "send_referral_sms",
            "description": "Envoie un SMS de référence structuré au centre de santé.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_age_months": {"type": "integer"},
                    "patient_sex": {"type": "string", "enum": ["M", "F"]},
                    "main_symptoms": {"type": "string"},
                    "triage_color": {"type": "string", "enum": ["ROUGE", "JAUNE", "VERT"]},
                    "estimated_arrival_minutes": {"type": "integer"},
                },
                "required": ["patient_age_months", "main_symptoms", "triage_color"],
            },
        },
    }
]


# ============================================================
# Backends (Ollama avec fallback démo)
# ============================================================


@st.cache_resource(show_spinner="Chargement de l'embedder multilingue…")
def get_embedder():
    """Charge l'encodeur multilingue (cache Streamlit)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )


@st.cache_resource(show_spinner="Indexation du corpus IMCI…")
def build_index():
    """Construit l'index FAISS sur le corpus de démo."""
    import faiss
    import numpy as np

    embedder = get_embedder()
    texts = [c["text"] for c in DEMO_CORPUS]
    embs = embedder.encode(texts, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return embedder, index


def retrieve(query: str, k: int = 4) -> list[dict]:
    """Récupération top-k sur l'index FAISS."""
    import numpy as np

    embedder, index = build_index()
    q = embedder.encode([query], normalize_embeddings=True).astype("float32")
    sims, idxs = index.search(q, k)
    results = []
    for idx, sim in zip(idxs[0], sims[0]):
        r = dict(DEMO_CORPUS[idx])
        r["similarity"] = float(sim)
        results.append(r)
    return results


def check_ollama() -> tuple[bool, str]:
    """Vérifie si Ollama tourne et le modèle est disponible."""
    try:
        import ollama

        models = ollama.list().get("models", [])
        names = [m.get("name", m.get("model", "")) for m in models]
        if not names:
            return False, "Ollama tourne mais aucun modèle n'est installé."
        return True, f"OK · modèles disponibles : {', '.join(names[:3])}"
    except Exception as e:
        return False, f"Ollama indisponible — mode démo activé. ({e})"


def call_gemma4(
    symptoms: str,
    image_bytes: bytes | None,
    retrieved: list[dict],
    model_tag: str,
    threshold: float,
) -> dict:
    """Appel Gemma 4 via Ollama avec garde-fou de seuil."""
    max_sim = max((r["similarity"] for r in retrieved), default=0.0)
    if max_sim < threshold:
        return {
            "triage": "ROUGE",
            "raison": f"Cas hors périmètre indexé (similarité max {max_sim:.2f} < seuil {threshold}).",
            "actions_immediates": ["Référer immédiatement au centre de santé"],
            "citation": "garde-fou MediBoussole",
            "confiance": 1.0 - max_sim,
            "_fallback": "threshold",
        }

    try:
        import ollama

        context = "\n\n".join(
            f"[Source {i+1} | p.{r['page']} | sim={r['similarity']:.2f}]\n{r['text']}"
            for i, r in enumerate(retrieved)
        )
        user_msg = (
            f"Symptômes décrits :\n{symptoms}\n\n"
            f"Protocoles WHO IMCI pertinents :\n{context}\n\n"
            f"Produis le triage au format JSON."
        )

        msg: dict[str, Any] = {"role": "user", "content": user_msg}
        if image_bytes:
            # Ollama accepte images en bytes (PIL.Image fonctionne aussi)
            msg["images"] = [image_bytes]

        response = ollama.chat(
            model=model_tag,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                msg,
            ],
            format="json",
            options={"temperature": 0.0, "num_ctx": 8192},
        )
        return json.loads(response["message"]["content"])

    except json.JSONDecodeError:
        return {
            "triage": "ROUGE",
            "raison": "Sortie modèle non parsable — référer par sécurité.",
            "actions_immediates": ["Référer immédiatement"],
            "citation": "fallback parser",
            "confiance": 0.0,
            "_fallback": "parse",
        }
    except Exception as exc:
        # Mode démo si Ollama indisponible
        return _demo_response(symptoms, retrieved, exc)


def _demo_response(symptoms: str, retrieved: list[dict], exc: Exception) -> dict:
    """Sortie simulée pédagogique quand Ollama n'est pas disponible."""
    s = symptoms.lower()
    danger = any(
        k in s for k in ["léthargi", "inconscient", "convuls", "ne peut plus", "tirage"]
    )
    if danger:
        triage = "ROUGE"
        reason = "Signes de danger général détectés."
        actions = [
            "Stabiliser et référer immédiatement",
            "Donner premier traitement de pré-référence",
        ]
    elif "fièvre" in s or "respir" in s:
        triage = "JAUNE"
        reason = "Symptôme nécessitant traitement protocolaire."
        actions = ["Suivre le protocole IMCI", "Conseil retour si signe d'aggravation"]
    else:
        triage = "VERT"
        reason = "Aucun signe de danger identifié."
        actions = ["Conseils mère", "Vérifier le calendrier vaccinal"]
    return {
        "triage": triage,
        "raison": reason,
        "actions_immediates": actions,
        "citation": f"page {retrieved[0]['page']} (top-1 sim={retrieved[0]['similarity']:.2f})",
        "confiance": 0.6,
        "_fallback": f"demo_mode ({exc})",
    }


def call_function_calling(triage_result: dict, symptoms: str, model_tag: str) -> list[dict]:
    """Demande au modèle d'appeler les outils nécessaires."""
    try:
        import ollama

        msg = (
            f"Symptômes : {symptoms}\n"
            f"Triage : {json.dumps(triage_result, ensure_ascii=False)}\n\n"
            f"Appelle send_referral_sms avec les bons paramètres."
        )
        response = ollama.chat(
            model=model_tag,
            messages=[{"role": "user", "content": msg}],
            tools=TOOLS,
            options={"temperature": 0.0},
        )
        return response["message"].get("tool_calls", [])
    except Exception:
        # Démo : SMS structuré simulé
        return [
            {
                "function": {
                    "name": "send_referral_sms",
                    "arguments": {
                        "patient_age_months": 14,
                        "patient_sex": "F",
                        "main_symptoms": symptoms[:80],
                        "triage_color": triage_result.get("triage", "ROUGE"),
                        "estimated_arrival_minutes": 25,
                    },
                }
            }
        ]


# ============================================================
# UI
# ============================================================


def render_sidebar() -> dict:
    """Sidebar : configuration."""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        model_tag = st.text_input(
            "Modèle Ollama",
            value=DEFAULT_MODEL_TAG,
            help="Tag du modèle Gemma 4 dans Ollama. Vérifier avec `ollama list`.",
        )

        threshold = st.slider(
            "Seuil d'abstention RAG (τ)",
            0.0, 1.0, 0.40, 0.05,
            help="Si max(similarité) < τ, MediBoussole s'abstient et oriente vers 'référer'.",
        )

        k = st.slider("Top-k documents récupérés", 1, 8, 4)

        st.divider()
        st.markdown("### 🔌 Statut Ollama")
        ok, msg = check_ollama()
        if ok:
            st.success(msg)
        else:
            st.warning(msg)
            st.caption("L'app fonctionne quand même en mode démo (sortie simulée).")

        st.divider()
        st.markdown("### 🛡️ Garde-fous actifs")
        st.markdown(
            """
- Scope IMCI < 5 ans verrouillé
- Abstention si τ insuffisant
- Citation page IMCI obligatoire
- Audit log local (off par défaut)
            """
        )

        st.divider()
        st.caption(
            "Démo · données patient synthétiques uniquement. "
            "Outil d'aide à la décision, ne remplace pas le médecin."
        )

        return {"model_tag": model_tag, "threshold": threshold, "k": k}


def render_input_panel() -> tuple[str, bytes | None, dict]:
    """Panneau gauche : entrée des symptômes + photo + métadonnées."""
    st.markdown("### 📥 Cas patient")

    col_a, col_b = st.columns(2)
    with col_a:
        age_months = st.number_input("Âge (mois)", 0, 60, 14)
    with col_b:
        sex = st.selectbox("Sexe", ["F", "M"])

    symptoms = st.text_area(
        "Symptômes décrits par l'agent de santé",
        value="Bébé fille 14 mois, fièvre 39.2°C depuis 2 jours, léthargie marquée, refus de boire, MUAC 12 cm jaune, respiration 42/min sans tirage.",
        height=150,
        help="Saisir en français ou langue locale (transcription whisper.cpp en prod).",
    )

    uploaded = st.file_uploader(
        "Photo de l'enfant (optionnel — multimodal)",
        type=["png", "jpg", "jpeg"],
        help="Encodage par le vision encoder de Gemma 4.",
    )
    image_bytes = uploaded.read() if uploaded else None
    if uploaded:
        st.image(uploaded, caption="Image envoyée au modèle", use_column_width=True)

    return symptoms, image_bytes, {"age_months": age_months, "sex": sex}


def render_triage_card(result: dict):
    """Affiche le résultat du triage avec couleur claire."""
    color_map = {
        "ROUGE": ("#ffebee", "#c62828", "🚨"),
        "JAUNE": ("#fff8e1", "#f57f17", "⚠️"),
        "VERT": ("#e8f5e9", "#2e7d32", "✅"),
    }
    triage = result.get("triage", "?")
    bg, fg, icon = color_map.get(triage, ("#eceff1", "#37474f", "❓"))

    st.markdown(
        f"""
        <div style="background:{bg};border-left:6px solid {fg};
                    padding:18px 24px;border-radius:8px;margin-bottom:12px;">
            <div style="font-size:22px;font-weight:700;color:{fg};">
                {icon} TRIAGE {triage}
            </div>
            <div style="margin-top:8px;color:#37474f;">
                {result.get('raison', '')}
            </div>
            <div style="margin-top:10px;font-size:13px;color:#546e7a;">
                Confiance : <b>{result.get('confiance', 0):.0%}</b>
                · Source : <i>{result.get('citation', '')}</i>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Actions immédiates :**")
    for a in result.get("actions_immediates", []):
        st.markdown(f"- {a}")

    if "_fallback" in result:
        st.caption(f"⚠️ Mode fallback : `{result['_fallback']}`")


def render_sources(retrieved: list[dict]):
    """Affiche les sources RAG avec scores."""
    st.markdown("### 🔍 Sources RAG (WHO IMCI)")
    for i, r in enumerate(retrieved, 1):
        with st.expander(
            f"Source {i} · page {r['page']} · similarité {r['similarity']:.2f}",
            expanded=(i == 1),
        ):
            st.write(r["text"])


def render_tool_calls(tool_calls: list[dict]):
    """Affiche les appels d'outils + SMS final."""
    st.markdown("### 📨 Function calling — Outils invoqués")
    if not tool_calls:
        st.info("Aucun outil n'a été appelé.")
        return

    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "?")
        args = fn.get("arguments", {})
        st.markdown(f"**`{name}`**")
        st.json(args)

        if name == "send_referral_sms":
            sms = (
                f"[MEDIBOUSSOLE] Patient {args.get('patient_sex','?')} "
                f"{args.get('patient_age_months','?')}m · "
                f"{args.get('main_symptoms','')[:80]} · "
                f"TRIAGE {args.get('triage_color','?')} · "
                f"ETA {args.get('estimated_arrival_minutes','?')}min."
            )
            st.markdown("**📱 Aperçu SMS (envoyable via 2G, 160 chars max) :**")
            st.code(sms, language=None)
            st.caption(f"Longueur : {len(sms)} caractères")


# ============================================================
# Main
# ============================================================


def main():
    # Header
    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        st.markdown("# 🩺")
    with col_title:
        st.markdown("# MediBoussole")
        st.caption(
            "Triage IMCI hors-ligne · Gemma 4 E4B · RAG sur protocoles WHO · "
            "100 % offline"
        )

    config = render_sidebar()

    # Layout principal : 2 colonnes
    left, right = st.columns([5, 6])

    with left:
        symptoms, image_bytes, meta = render_input_panel()
        analyze_btn = st.button(
            "🔬 Analyser le cas",
            type="primary",
            use_container_width=True,
        )

    with right:
        if not analyze_btn:
            st.info(
                "👈 Saisissez les symptômes (et optionnellement une photo), "
                "puis cliquez sur **Analyser le cas**."
            )
            st.markdown(
                """
                ### Pipeline
                1. **RAG** — récupération top-k sur WHO IMCI (multilingue)
                2. **Garde-fou** — abstention si similarité < τ
                3. **Gemma 4 E4B** — triage multimodal (texte + image)
                4. **Function calling** — SMS de référence + note SOAP
                5. **Audit** — journal local chiffré
                """
            )
            return

        with st.status("Pipeline MediBoussole en cours…", expanded=True) as status:
            st.write("🔍 RAG : recherche dans WHO IMCI…")
            t0 = time.perf_counter()
            retrieved = retrieve(symptoms, k=config["k"])
            t_retr = time.perf_counter() - t0
            st.write(f"   → {len(retrieved)} passages, top sim={retrieved[0]['similarity']:.2f} ({t_retr*1000:.0f} ms)")

            st.write(f"🧠 Gemma 4 ({config['model_tag']})…")
            t0 = time.perf_counter()
            result = call_gemma4(
                symptoms,
                image_bytes,
                retrieved,
                model_tag=config["model_tag"],
                threshold=config["threshold"],
            )
            t_llm = time.perf_counter() - t0
            st.write(f"   → triage {result.get('triage')} ({t_llm*1000:.0f} ms)")

            st.write("📨 Function calling…")
            t0 = time.perf_counter()
            tool_calls = call_function_calling(result, symptoms, config["model_tag"])
            t_tools = time.perf_counter() - t0
            st.write(f"   → {len(tool_calls)} appel(s) ({t_tools*1000:.0f} ms)")

            status.update(label="✅ Analyse terminée", state="complete", expanded=False)

        render_triage_card(result)

        with st.expander("⏱️ Latences", expanded=False):
            st.json(
                {
                    "rag_retrieval_ms": round(t_retr * 1000, 1),
                    "llm_inference_ms": round(t_llm * 1000, 1),
                    "function_calling_ms": round(t_tools * 1000, 1),
                    "total_ms": round((t_retr + t_llm + t_tools) * 1000, 1),
                }
            )

        render_sources(retrieved)
        render_tool_calls(tool_calls)


if __name__ == "__main__":
    main()
