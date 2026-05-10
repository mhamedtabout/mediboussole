"""
MediBoussole Studio — démo multi-modes inspirée de Google AI Studio Apps.

Modes :
  • Triage          — pipeline RAG + LLM + function calling (cas concret)
  • Conversation    — chat multi-tour avec Gemma 4 + RAG
  • RAG Explorer    — explorer le corpus WHO IMCI, voir similarités
  • Tool Sandbox    — tester n'importe quel schéma de function calling
  • Benchmark       — relancer un benchmark live

Layout : sidebar settings | main inputs | preview panel (JSON + curl + Python).

Lancer :
    streamlit run src/app.py
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import streamlit as st

# ============================================================
# Config
# ============================================================

st.set_page_config(
    page_title="MediBoussole Studio",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index"

DEFAULT_MODEL = "gemma4:e4b-it-q4_K_M"
AVAILABLE_VARIANTS = [
    "gemma4:e4b-it-q4_K_M",
    "gemma4:e4b-it-q8_0",
    "gemma4:e2b-it-q4_K_M",
    "gemma4:e2b-it-q8_0",
    "gemma4:26b-a4b-it-q4_K_M",
    "gemma4:31b-it-q4_K_M",
]

DEMO_CORPUS = [
    {"id": "imci-fever-001", "page": 7,
     "text": "Enfant 2 mois à 5 ans avec fièvre. Signes de danger général : incapable de boire ou téter, vomit tout, convulsions, léthargie ou inconscience. Tout signe = TRIAGE ROUGE, référer immédiatement."},
    {"id": "imci-dehyd-002", "page": 12,
     "text": "Évaluation déshydratation : léthargique ou inconscient, yeux enfoncés, incapable de boire, pli cutané qui s'efface très lentement (≥2s). Deux signes ou plus = déshydratation sévère = TRIAGE ROUGE."},
    {"id": "imci-pneum-003", "page": 5,
     "text": "Toux ou difficulté respiratoire. Compter respiration. Tirage sous-costal ou stridor au repos = pneumonie sévère = TRIAGE ROUGE. Respiration rapide (≥50/min 2-12 mois, ≥40/min 12 mois-5 ans) = pneumonie = TRIAGE JAUNE, amoxicilline 40 mg/kg 2x/j 5j."},
    {"id": "imci-malnut-004", "page": 18,
     "text": "Évaluation nutritionnelle : périmètre brachial (MUAC) ruban à 3 couleurs. MUAC < 11.5 cm (rouge) = malnutrition aiguë sévère = TRIAGE ROUGE. MUAC 11.5-12.5 cm (jaune) = malnutrition modérée = TRIAGE JAUNE."},
    {"id": "imci-diarr-005", "page": 14,
     "text": "Diarrhée. Durée et présence de sang. Diarrhée ≥14 jours = persistante. Sang dans selles = dysenterie = TRIAGE JAUNE, antibiotique selon protocole national. Diarrhée + déshydratation sévère = TRIAGE ROUGE."},
    {"id": "imci-mal-006", "page": 8,
     "text": "Fièvre en zone palustre. TDR si disponible. TDR positif sans signe de danger = palu simple = ACT. Fièvre + signe de danger = palu grave = TRIAGE ROUGE, artésunate IM/IV en pré-référence."},
    {"id": "imci-couns-008", "page": 25,
     "text": "Conseils retour à domicile : continuer allaitement, donner SRO après chaque selle liquide. Signes de retour immédiat : ne peut plus boire, fièvre qui empire, sang dans les selles, respiration plus difficile."},
]

DEFAULT_SYSTEM_PROMPT = """Tu es MediBoussole, un assistant de triage pour agents de santé communautaires.

RÈGLES STRICTES :
1. Réponds en français clair, ton calme et professionnel.
2. Utilise EXCLUSIVEMENT les protocoles WHO IMCI fournis dans le contexte.
3. Si l'information manque, ta réponse est : triage="ROUGE", action="référer immédiatement".
4. Ne propose JAMAIS de dosage absent du contexte.
5. Cite toujours la page IMCI source.

RÈGLE ABSOLUE — SIGNES DE DANGER GÉNÉRAL = TRIAGE ROUGE AUTOMATIQUE :
Si l'enfant présente AU MOINS UN de ces signes, le triage EST ROUGE :
- Léthargie, somnolence anormale, ou inconscience
- Convulsions
- Incapable de boire ou de téter
- Vomit tout ce qu'il consomme
- Tirage sous-costal sévère ou stridor au repos
- Yeux enfoncés + pli cutané qui s'efface lentement
- MUAC < 11.5 cm

Format JSON strict :
{"triage": "ROUGE"|"JAUNE"|"VERT", "raison": "...", "actions_immediates": ["..."], "citation": "page X", "confiance": 0.0-1.0}"""

DEFAULT_TOOLS = [
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


# ============================================================
# Backends
# ============================================================


@st.cache_resource(show_spinner=False)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")


@st.cache_resource(show_spinner=False)
def load_index_and_chunks():
    """Charge l'index FAISS persistent si dispo, sinon construit depuis DEMO_CORPUS."""
    import faiss
    import numpy as np

    persistent_idx = INDEX_DIR / "imci.faiss"
    persistent_chunks = INDEX_DIR / "imci_chunks.pkl"

    if persistent_idx.exists() and persistent_chunks.exists():
        index = faiss.read_index(str(persistent_idx))
        with open(persistent_chunks, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks, "persistent (WHO IMCI complet)"

    embedder = get_embedder()
    embs = embedder.encode([c["text"] for c in DEMO_CORPUS],
                            normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index, DEMO_CORPUS, "embarqué (7 chunks de démo)"


def retrieve(query: str, k: int = 4) -> list[dict]:
    import numpy as np
    embedder = get_embedder()
    index, chunks, _ = load_index_and_chunks()
    q = embedder.encode([query], normalize_embeddings=True).astype("float32")
    sims, idxs = index.search(q, k)
    return [
        {**chunks[i], "similarity": float(s)}
        for i, s in zip(idxs[0], sims[0])
    ]


def check_ollama() -> tuple[bool, str, list[str]]:
    try:
        import ollama
        models = ollama.list().get("models", [])
        names = [m.get("model", m.get("name", "")) for m in models]
        if not names:
            return False, "Ollama tourne mais aucun modèle.", []
        return True, f"{len(names)} modèle(s) installé(s)", names
    except Exception as e:
        return False, f"Ollama indisponible : {e}", []


def call_gemma(
    messages: list[dict],
    model: str,
    temperature: float,
    num_ctx: int,
    json_format: bool,
    tools: list[dict] | None = None,
    images: list[bytes] | None = None,
    top_p: float = 1.0,
    max_tokens: int = 0,
    stop: list[str] | None = None,
    stream: bool = False,
):
    """Appel Gemma 4 via Ollama. Si stream=True, retourne un générateur."""
    import ollama

    msgs = [dict(m) for m in messages]
    if images and msgs:
        msgs[-1]["images"] = images

    options: dict = {
        "temperature": temperature,
        "num_ctx": num_ctx,
        "top_p": top_p,
    }
    if max_tokens > 0:
        options["num_predict"] = max_tokens
    if stop:
        options["stop"] = [s for s in stop if s.strip()]

    kwargs: dict = {
        "model": model,
        "messages": msgs,
        "options": options,
    }
    if json_format and not tools and not stream:
        kwargs["format"] = "json"
    if tools:
        kwargs["tools"] = tools

    if stream:
        kwargs["stream"] = True
        return ollama.chat(**kwargs)  # générateur
    else:
        t0 = time.perf_counter()
        response = ollama.chat(**kwargs)
        elapsed = time.perf_counter() - t0
        return response, elapsed


def transcribe_audio(audio_bytes: bytes, language: str = "fr") -> str:
    """Transcribe audio via Whisper (si disponible)."""
    try:
        import tempfile
        import whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            path = f.name
        model = whisper.load_model("base")
        result = model.transcribe(path, language=language)
        return result["text"].strip()
    except ImportError:
        return "[whisper non installé : pip install openai-whisper]"
    except Exception as e:
        return f"[erreur transcription : {e}]"


def save_case_history(case: dict) -> None:
    """Append un cas dans data/audit/history.jsonl pour audit."""
    audit_dir = ROOT / "data" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    history_file = audit_dir / "history.jsonl"
    with open(history_file, "a") as f:
        f.write(json.dumps(case, ensure_ascii=False) + "\n")


def load_case_history(limit: int = 20) -> list[dict]:
    """Charge les N derniers cas depuis l'audit log."""
    history_file = ROOT / "data" / "audit" / "history.jsonl"
    if not history_file.exists():
        return []
    lines = history_file.read_text().splitlines()
    cases = []
    for line in lines[-limit:]:
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(cases))


# ============================================================
# Sidebar — paramètres globaux
# ============================================================


def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("### 🩺 MediBoussole Studio")
        st.caption("Multi-modes · Gemma 4 · WHO IMCI RAG")

        st.divider()

        # Statut Ollama
        st.markdown("#### 🔌 Statut")
        ok, msg, available = check_ollama()
        if ok:
            st.success(msg)
        else:
            st.warning(msg)
            st.caption("Mode démo activé (sortie simulée).")

        # Index
        idx, chunks, label = load_index_and_chunks()
        st.info(f"📚 Corpus : {len(chunks)} chunks · {label}")

        st.divider()

        st.markdown("#### ⚙️ Modèle")
        model = st.selectbox(
            "Variante Gemma 4",
            options=AVAILABLE_VARIANTS,
            index=0,
            help="Vérifier `ollama list` que la variante est installée.",
        )

        with st.expander("Sampling", expanded=False):
            temperature = st.slider("Temperature", 0.0, 2.0, 0.0, 0.1,
                                     help="0 = déterministe, 1 = créatif, 2 = chaotique")
            top_p = st.slider("Top-p (nucleus)", 0.0, 1.0, 1.0, 0.05,
                              help="Échantillonnage par masse cumulative")
            max_tokens = st.number_input("Max output tokens (0 = illimité)",
                                          0, 8192, 0, 100)
            stop_sequences_raw = st.text_input(
                "Stop sequences (séparées par |)",
                value="",
                help="Ex : </JSON>|END",
            )
            stop_sequences = [s for s in stop_sequences_raw.split("|") if s.strip()]

        with st.expander("Contexte / output", expanded=False):
            num_ctx = st.select_slider(
                "Context window",
                options=[2048, 4096, 8192, 16384, 32768],
                value=8192,
            )
            stream = st.toggle("Streaming (token par token)", value=False,
                               help="Réponse progressive plutôt qu'attendre la fin")
            voice_output = st.toggle("Voice output (TTS macOS)", value=False,
                                     help="Lecture vocale du diagnostic après triage")

        st.divider()

        st.markdown("#### 🔍 RAG")
        top_k = st.slider("Top-k retrieved", 1, 8, 4)
        threshold = st.slider("Seuil τ d'abstention", 0.0, 1.0, 0.40, 0.05)

        st.divider()

        st.markdown("#### 🌍 Localisation")
        ui_lang = st.selectbox(
            "Langue de l'interface",
            ["Français", "English", "Wolof (auto)", "Bambara (auto)"],
            index=0,
            help="Le contenu reste FR ; les boutons s'adaptent (à venir)",
        )

        st.divider()

        st.markdown("#### 🛡️ Garde-fous")
        scope_lock = st.toggle("Scope IMCI < 5 ans verrouillé", value=True)
        require_citation = st.toggle("Citation page IMCI obligatoire", value=True)
        audit_log = st.toggle("Audit log local (persiste les cas)", value=True)
        webhook_red = st.toggle("Webhook si triage ROUGE (fictif)", value=False,
                                help="Mode démo — log simulé dans la console")

        st.divider()

        with st.expander("📂 Historique des cas", expanded=False):
            history = load_case_history(limit=10)
            if history:
                for h in history:
                    triage = h.get("result", {}).get("triage", "?")
                    timestamp = h.get("timestamp", "")[:16]
                    icon = {"ROUGE": "🚨", "JAUNE": "⚠️", "VERT": "✅"}.get(triage, "❓")
                    label = h.get("symptoms", "")[:50]
                    st.markdown(f"{icon} `{timestamp}` · {label}…")
                if st.button("📥 Exporter historique JSONL", use_container_width=True):
                    audit_path = ROOT / "data" / "audit" / "history.jsonl"
                    if audit_path.exists():
                        st.download_button(
                            "Télécharger",
                            audit_path.read_bytes(),
                            file_name="mediboussole_history.jsonl",
                            mime="application/jsonl",
                        )
            else:
                st.caption("Aucun cas archivé pour l'instant.")

        st.divider()
        st.caption(
            "Outil d'aide à la décision · ne remplace pas le médecin.\n\n"
            "The Gemma 4 Good Hackathon · CC-BY 4.0"
        )

        return {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stop_sequences": stop_sequences,
            "num_ctx": num_ctx,
            "stream": stream,
            "voice_output": voice_output,
            "top_k": top_k,
            "threshold": threshold,
            "ui_lang": ui_lang,
            "scope_lock": scope_lock,
            "require_citation": require_citation,
            "audit_log": audit_log,
            "webhook_red": webhook_red,
            "ollama_ok": ok,
        }


# ============================================================
# Mode 1 — Triage
# ============================================================


def render_triage_mode(cfg: dict):
    st.markdown("### 🚨 Mode Triage")
    st.caption("Pipeline complet : RAG → garde-fou τ → Gemma 4 multimodal → JSON structuré.")

    col_left, col_right = st.columns([5, 6], gap="large")

    with col_left:
        st.markdown("#### 📥 Cas patient")
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Âge (mois)", 0, 60, 14, key="t_age")
        with c2:
            sex = st.selectbox("Sexe", ["F", "M"], key="t_sex")

        # Voice input (mic) — option AI Studio-like
        with st.expander("🎤 Saisie vocale (au lieu du texte)", expanded=False):
            audio_input = st.audio_input("Enregistrer description vocale", key="t_audio")
            if audio_input is not None:
                with st.spinner("Transcription Whisper..."):
                    transcribed = transcribe_audio(audio_input.read(), language="fr")
                st.success(f"Transcription : {transcribed}")
                st.session_state["t_symptoms"] = transcribed

        symptoms = st.text_area(
            "Symptômes (texte ou résultat de transcription)",
            value=st.session_state.get(
                "t_symptoms",
                "Bébé fille 14 mois, fièvre 39.2°C depuis 2 jours, léthargie marquée, refus de boire.",
            ),
            height=140,
            key="t_symptoms",
        )

        with st.expander("Prompt système (avancé)"):
            sys_prompt = st.text_area(
                "system_prompt",
                value=DEFAULT_SYSTEM_PROMPT,
                height=300,
                key="t_sys",
                label_visibility="collapsed",
            )

        uploaded = st.file_uploader("Photo (multimodal)", type=["png", "jpg", "jpeg"], key="t_img")
        image_bytes = uploaded.read() if uploaded else None
        if uploaded:
            st.image(uploaded, use_column_width=True)

        run = st.button("🔬 Analyser", type="primary", use_container_width=True, key="t_run")

    with col_right:
        if not run:
            st.info("👈 Saisir le cas et cliquer **Analyser**.")
            return

        with st.status("Pipeline en cours…", expanded=True) as status:
            t0 = time.perf_counter()
            retrieved = retrieve(symptoms, k=cfg["top_k"])
            t_retr = (time.perf_counter() - t0) * 1000
            top_sim = max(r["similarity"] for r in retrieved)
            st.write(f"🔍 RAG : top sim = {top_sim:.3f} ({t_retr:.0f} ms)")

            if top_sim < cfg["threshold"]:
                result = {
                    "triage": "ROUGE",
                    "raison": f"Cas hors périmètre indexé (sim {top_sim:.2f} < τ {cfg['threshold']}).",
                    "actions_immediates": ["Référer immédiatement"],
                    "citation": "garde-fou MediBoussole",
                    "confiance": 1.0 - top_sim,
                }
                t_llm = 0.0
                st.write("🛡️ Garde-fou déclenché — abstention.")
            else:
                context = "\n\n".join(
                    f"[Source {i+1} | p.{r['page']} | sim={r['similarity']:.2f}]\n{r['text']}"
                    for i, r in enumerate(retrieved)
                )
                user_msg = f"Symptômes :\n{symptoms}\n\nProtocoles WHO IMCI :\n{context}\n\nProduis le triage JSON."
                msgs = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                ]
                if cfg["ollama_ok"]:
                    try:
                        response, t_llm_s = call_gemma(
                            msgs, cfg["model"], cfg["temperature"], cfg["num_ctx"],
                            json_format=True,
                            images=[image_bytes] if image_bytes else None,
                            top_p=cfg["top_p"],
                            max_tokens=cfg["max_tokens"],
                            stop=cfg["stop_sequences"],
                        )
                        result = json.loads(response["message"]["content"])
                        t_llm = t_llm_s * 1000
                        st.write(f"🧠 Gemma 4 : triage {result.get('triage')} ({t_llm:.0f} ms)")
                    except Exception as e:
                        result = {
                            "triage": "ROUGE",
                            "raison": f"Erreur LLM : {e} — référer par sécurité.",
                            "actions_immediates": ["Référer immédiatement"],
                            "citation": "fallback",
                            "confiance": 0.0,
                        }
                        t_llm = 0
                else:
                    result = _demo_response(symptoms, retrieved)
                    t_llm = 50

            status.update(label="✅ Terminé", state="complete", expanded=False)

        render_triage_card(result)

        # Voice output (TTS macOS) — option AI Studio-like
        if cfg["voice_output"]:
            try:
                import subprocess as sp
                speech_text = f"Triage {result.get('triage', 'inconnu')}. {result.get('raison', '')}"
                sp.Popen(["say", "-v", "Thomas", "-r", "170", speech_text])
                st.caption("🔊 Lecture vocale en cours…")
            except Exception:
                st.caption("⚠️ Voice output indisponible (macOS uniquement)")

        # Webhook ROUGE (option)
        if cfg["webhook_red"] and result.get("triage") == "ROUGE":
            st.error("📡 Webhook ROUGE déclenché (mode démo) → log simulé envoyé")

        # Audit log (option)
        if cfg["audit_log"]:
            save_case_history({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "age_months": age,
                "sex": sex,
                "symptoms": symptoms,
                "result": result,
                "rag_max_sim": top_sim,
                "model": cfg["model"],
            })

        # Boutons d'export
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.download_button(
                "📥 Exporter JSON",
                data=json.dumps(result, indent=2, ensure_ascii=False),
                file_name=f"triage_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_e2:
            md_export = (
                f"# Triage MediBoussole\n\n"
                f"**Date** : {time.strftime('%Y-%m-%d %H:%M')}\n"
                f"**Patient** : {sex} · {age} mois\n"
                f"**Symptômes** : {symptoms}\n\n"
                f"## Résultat\n\n"
                f"- **Triage** : {result.get('triage','?')}\n"
                f"- **Raison** : {result.get('raison','')}\n"
                f"- **Citation** : {result.get('citation','')}\n"
                f"- **Confiance** : {result.get('confiance', 0):.0%}\n\n"
                f"## Actions immédiates\n\n"
                + "\n".join(f"- {a}" for a in result.get("actions_immediates", []))
                + "\n"
            )
            st.download_button(
                "📝 Exporter Markdown",
                data=md_export,
                file_name=f"triage_{int(time.time())}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        render_preview_panel(
            mode="triage",
            messages=[{"role": "system", "content": sys_prompt[:200] + "..."},
                      {"role": "user", "content": symptoms}],
            cfg=cfg,
            result=result,
            latencies={"rag_ms": t_retr, "llm_ms": t_llm},
            sources=retrieved,
        )


def render_triage_card(result: dict):
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
            <div style="margin-top:8px;color:#37474f;">{result.get('raison','')}</div>
            <div style="margin-top:10px;font-size:13px;color:#546e7a;">
                Confiance : <b>{result.get('confiance', 0):.0%}</b> ·
                <i>{result.get('citation','')}</i>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if result.get("actions_immediates"):
        st.markdown("**Actions immédiates :**")
        for a in result["actions_immediates"]:
            st.markdown(f"- {a}")


def _demo_response(symptoms: str, retrieved: list[dict]) -> dict:
    s = symptoms.lower()
    danger = any(k in s for k in ["léthargi", "inconscient", "convuls", "ne peut plus", "tirage"])
    if danger:
        triage, reason = "ROUGE", "Signes de danger général détectés."
    elif "fièvre" in s or "respir" in s:
        triage, reason = "JAUNE", "Symptôme nécessitant traitement protocolaire."
    else:
        triage, reason = "VERT", "Pas de signe de danger."
    return {
        "triage": triage,
        "raison": reason + " (mode démo, sans Gemma 4)",
        "actions_immediates": ["Suivre protocole IMCI"],
        "citation": f"page {retrieved[0]['page']}",
        "confiance": 0.6,
    }


# ============================================================
# Mode 2 — Conversation
# ============================================================


def render_chat_mode(cfg: dict):
    st.markdown("### 💬 Mode Conversation")
    st.caption("Chat multi-tours avec Gemma 4 + RAG automatique sur chaque message.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]

    col_left, col_right = st.columns([5, 6], gap="large")

    with col_left:
        # Affichage de la conversation
        for msg in st.session_state.chat_history[1:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("Message à MediBoussole…")
        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # RAG enrichment
            retrieved = retrieve(prompt, k=cfg["top_k"])
            ctx = "\n\n".join(
                f"[p.{r['page']} sim={r['similarity']:.2f}] {r['text']}"
                for r in retrieved[:3]
            )
            enriched = (
                st.session_state.chat_history[:-1]
                + [{"role": "user", "content": f"{prompt}\n\nContexte WHO IMCI :\n{ctx}"}]
            )

            with st.chat_message("assistant"):
                if cfg["ollama_ok"]:
                    try:
                        response, t = call_gemma(
                            enriched, cfg["model"], cfg["temperature"], cfg["num_ctx"],
                            json_format=False,
                        )
                        reply = response["message"]["content"]
                        st.markdown(reply)
                        st.caption(f"⏱️ {t*1000:.0f} ms · sources : p.{', p.'.join(str(r['page']) for r in retrieved[:3])}")
                    except Exception as e:
                        reply = f"Erreur Gemma 4 : {e}"
                        st.error(reply)
                else:
                    reply = "(Mode démo) Je vous suggère de référer si vous voyez le moindre signe de danger général."
                    st.info(reply)

            st.session_state.chat_history.append({"role": "assistant", "content": reply})

        if st.button("🗑️ Effacer la conversation"):
            st.session_state.chat_history = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
            st.rerun()

    with col_right:
        st.markdown("#### 📊 État")
        st.metric("Messages", len(st.session_state.chat_history) - 1)
        if len(st.session_state.chat_history) > 1:
            with st.expander("Historique complet (JSON)", expanded=False):
                st.json(st.session_state.chat_history)


# ============================================================
# Mode 3 — RAG Explorer
# ============================================================


def render_rag_explorer(cfg: dict):
    st.markdown("### 🔍 RAG Explorer")
    st.caption("Explorer le corpus WHO IMCI : recherche sémantique, similarités, multilingue.")

    col_left, col_right = st.columns([5, 6], gap="large")

    with col_left:
        query = st.text_input(
            "Requête (français / wolof / bambara / peul)",
            value="bébé fièvre léthargie",
            key="r_q",
        )
        k = st.slider("Top-k", 1, 10, cfg["top_k"], key="r_k")
        run = st.button("🔍 Rechercher", type="primary", key="r_run")

        st.markdown("##### Exemples de requêtes")
        for ex in [
            "déshydratation sévère pli cutané",
            "MUAC malnutrition aiguë sévère",
            "diarrhée sang dans les selles",
            "Lymphome Hodgkin (hors-scope)",
            "lethargic baby fever (anglais)",
        ]:
            if st.button(ex, key=f"ex_{hash(ex)}"):
                st.session_state["r_q"] = ex
                st.rerun()

    with col_right:
        if not run and "r_q" not in st.session_state:
            st.info("👈 Saisir une requête.")
            return

        q = st.session_state.get("r_q", query)
        t0 = time.perf_counter()
        retrieved = retrieve(q, k=k)
        t = (time.perf_counter() - t0) * 1000

        st.metric("Latence retrieval", f"{t:.1f} ms")

        max_sim = max(r["similarity"] for r in retrieved)
        if max_sim < cfg["threshold"]:
            st.warning(f"⚠️ Garde-fou déclencherait : max sim {max_sim:.3f} < τ {cfg['threshold']}")
        else:
            st.success(f"✅ Garde-fou OK : max sim {max_sim:.3f} ≥ τ {cfg['threshold']}")

        for i, r in enumerate(retrieved, 1):
            with st.expander(f"#{i} · p.{r['page']} · sim={r['similarity']:.3f}", expanded=(i == 1)):
                st.write(r["text"])
                st.progress(min(r["similarity"], 1.0))


# ============================================================
# Mode 4 — Tool Sandbox
# ============================================================


def render_tool_sandbox(cfg: dict):
    st.markdown("### 🔧 Tool Sandbox")
    st.caption("Tester n'importe quel schéma JSON de function calling avec Gemma 4.")

    col_left, col_right = st.columns([5, 6], gap="large")

    with col_left:
        st.markdown("##### Schémas (JSON)")
        tools_json = st.text_area(
            "tools",
            value=json.dumps(DEFAULT_TOOLS, indent=2, ensure_ascii=False),
            height=400,
            label_visibility="collapsed",
        )

        prompt = st.text_area(
            "Message utilisateur",
            value="Bébé F 14 mois, fièvre 39 + léthargie. Triage ROUGE. Appelle send_referral_sms et generate_clinical_note.",
            height=120,
            key="ts_prompt",
        )
        run = st.button("🔧 Tester", type="primary", key="ts_run")

    with col_right:
        if not run:
            st.info("👈 Définir tools + message, puis **Tester**.")
            return

        try:
            tools = json.loads(tools_json)
        except json.JSONDecodeError as e:
            st.error(f"JSON invalide : {e}")
            return

        if not cfg["ollama_ok"]:
            st.warning("Ollama indisponible. Mode démo affiche une trace simulée.")
            st.json([{"function": {"name": "send_referral_sms",
                                    "arguments": {"patient_age_months": 14, "main_symptoms": prompt[:80],
                                                  "triage_color": "ROUGE"}}}])
            return

        try:
            response, t = call_gemma(
                [{"role": "user", "content": prompt}],
                cfg["model"], cfg["temperature"], cfg["num_ctx"],
                json_format=False, tools=tools,
            )
            calls = response["message"].get("tool_calls", [])
            st.metric("Latence", f"{t*1000:.0f} ms")
            st.metric("Tool calls produits", len(calls))
            for i, tc in enumerate(calls, 1):
                with st.expander(f"Tool call #{i} : `{tc.get('function', {}).get('name', '?')}`",
                                 expanded=True):
                    st.json(tc.get("function", {}).get("arguments", {}))
            if not calls:
                st.warning("Aucun outil n'a été appelé. Texte brut :")
                st.code(response["message"].get("content", ""))
        except Exception as e:
            st.error(f"Erreur : {e}")


# ============================================================
# Mode 5 — Compare A/B
# ============================================================


def render_compare_ab(cfg: dict):
    st.markdown("### ⚖️ Compare A/B — deux modèles côte-à-côte")
    st.caption("Compare les sorties de deux variantes Gemma 4 sur le même cas.")

    col_setup_a, col_setup_b = st.columns(2)
    with col_setup_a:
        model_a = st.selectbox("Modèle A", AVAILABLE_VARIANTS, index=0, key="cmp_a")
        temp_a = st.slider("Temperature A", 0.0, 2.0, 0.0, 0.1, key="cmp_ta")
    with col_setup_b:
        model_b = st.selectbox("Modèle B", AVAILABLE_VARIANTS,
                               index=min(2, len(AVAILABLE_VARIANTS) - 1), key="cmp_b")
        temp_b = st.slider("Temperature B", 0.0, 2.0, 0.0, 0.1, key="cmp_tb")

    case = st.text_area(
        "Cas à comparer",
        value="Bébé 8 mois, MUAC 10.5 cm, œdèmes des deux pieds, irritable.",
        height=100,
        key="cmp_case",
    )
    if not st.button("⚖️ Comparer", type="primary", key="cmp_run"):
        return

    if not cfg["ollama_ok"]:
        st.warning("Ollama requis pour ce mode.")
        return

    retrieved = retrieve(case, k=cfg["top_k"])
    ctx = "\n\n".join(f"[p.{r['page']}]\n{r['text']}" for r in retrieved)
    msgs = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": f"{case}\n\nWHO IMCI:\n{ctx}\n\nJSON triage."},
    ]

    col_a, col_b = st.columns(2)
    for col, model_name, temp in [(col_a, model_a, temp_a), (col_b, model_b, temp_b)]:
        with col:
            st.markdown(f"#### {model_name}")
            with st.spinner(f"Inférence {model_name}…"):
                try:
                    response, t = call_gemma(
                        msgs, model_name, temp, cfg["num_ctx"], json_format=True,
                    )
                    result = json.loads(response["message"]["content"])
                    render_triage_card(result)
                    st.metric("Latence", f"{t*1000:.0f} ms")
                    with st.expander("JSON brut"):
                        st.json(result)
                except Exception as e:
                    st.error(f"Erreur : {e}")


# ============================================================
# Mode 6 — Benchmark
# ============================================================


def render_benchmark(cfg: dict):
    st.markdown("### 📊 Benchmark Live")
    st.caption("Reproduire les chiffres du writeup : latence retrieval, LLM cold/warm, accuracy.")

    if not cfg["ollama_ok"]:
        st.warning("Ollama requis pour ce mode.")
        return

    n_runs = st.slider("Runs par cas", 1, 5, 2, key="b_n")
    if not st.button("🚀 Lancer benchmark", type="primary"):
        return

    test_cases = [
        ("Bébé garçon 4 mois, ne peut plus téter, vomit tout, léthargique.", "ROUGE"),
        ("Garçon 30 mois, toux 2 jours, respiration 35/min, alerte, boit normalement.", "VERT"),
        ("Bébé 6 mois, diarrhée 2 jours, yeux enfoncés, ne boit plus.", "ROUGE"),
    ]

    rag_lats, llm_lats = [], []
    correct = 0
    progress = st.progress(0.0)
    for ci, (case, expected) in enumerate(test_cases):
        for r in range(n_runs):
            t0 = time.perf_counter()
            retrieved = retrieve(case, k=cfg["top_k"])
            rag_lats.append((time.perf_counter() - t0) * 1000)

            ctx = "\n\n".join(f"[p.{x['page']}]\n{x['text']}" for x in retrieved)
            msgs = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": f"{case}\n\n{ctx}\n\nJSON triage."},
            ]
            try:
                response, t = call_gemma(msgs, cfg["model"], 0.0, cfg["num_ctx"], json_format=True)
                llm_lats.append(t * 1000)
                result = json.loads(response["message"]["content"])
                if result.get("triage") == expected:
                    correct += 1
            except Exception:
                llm_lats.append(0)
            progress.progress(((ci * n_runs) + r + 1) / (len(test_cases) * n_runs))

    progress.empty()
    c1, c2, c3 = st.columns(3)
    c1.metric("RAG p50", f"{sorted(rag_lats)[len(rag_lats)//2]:.1f} ms")
    c2.metric("LLM p50", f"{sorted(llm_lats)[len(llm_lats)//2]:.0f} ms")
    c3.metric("Accuracy", f"{correct}/{len(test_cases) * n_runs}")

    with st.expander("Détail latences"):
        st.json({"rag_lats_ms": rag_lats, "llm_lats_ms": llm_lats})


# ============================================================
# Preview panel — JSON + curl + Python
# ============================================================


def render_preview_panel(mode: str, messages: list[dict], cfg: dict, result: dict,
                         latencies: dict, sources: list[dict]):
    with st.expander("🔬 Preview & code (AI Studio-like)", expanded=False):
        tab_json, tab_curl, tab_py, tab_lat, tab_src = st.tabs(
            ["JSON", "curl", "Python", "Latences", "Sources RAG"]
        )

        with tab_json:
            st.json(result)

        with tab_curl:
            payload = {
                "model": cfg["model"],
                "messages": messages,
                "format": "json",
                "options": {
                    "temperature": cfg["temperature"],
                    "num_ctx": cfg["num_ctx"],
                },
            }
            curl_cmd = (
                f"curl http://localhost:11434/api/chat \\\n"
                f"  -H 'Content-Type: application/json' \\\n"
                f"  -d '{json.dumps(payload, ensure_ascii=False)[:500]}…'"
            )
            st.code(curl_cmd, language="bash")

        with tab_py:
            py_code = f"""import ollama, json

response = ollama.chat(
    model="{cfg['model']}",
    messages=[
        # ... voir messages tab
    ],
    format="json",
    options={{
        "temperature": {cfg['temperature']},
        "num_ctx": {cfg['num_ctx']},
    }},
)
result = json.loads(response["message"]["content"])
print(result)
"""
            st.code(py_code, language="python")

        with tab_lat:
            cols = st.columns(len(latencies))
            for col, (k, v) in zip(cols, latencies.items()):
                col.metric(k, f"{v:.0f} ms")

        with tab_src:
            for i, r in enumerate(sources, 1):
                st.markdown(f"**Source {i}** · p.{r['page']} · sim {r['similarity']:.3f}")
                st.caption(r["text"][:300] + ("…" if len(r["text"]) > 300 else ""))


# ============================================================
# Main
# ============================================================


def main():
    cfg = render_sidebar()

    # Header
    h1, h2 = st.columns([1, 9])
    with h1:
        st.markdown("# 🩺")
    with h2:
        st.markdown("# MediBoussole Studio")
        st.caption(
            "Triage IMCI hors-ligne · Gemma 4 multimodal · RAG WHO · "
            "Function calling natif · 100% offline-ready"
        )

    # Mode selector (top tabs)
    tab_triage, tab_chat, tab_rag, tab_tools, tab_compare, tab_bench = st.tabs([
        "🚨 Triage",
        "💬 Conversation",
        "🔍 RAG Explorer",
        "🔧 Tool Sandbox",
        "⚖️ Compare A/B",
        "📊 Benchmark",
    ])

    with tab_triage:
        render_triage_mode(cfg)

    with tab_chat:
        render_chat_mode(cfg)

    with tab_rag:
        render_rag_explorer(cfg)

    with tab_tools:
        render_tool_sandbox(cfg)

    with tab_compare:
        render_compare_ab(cfg)

    with tab_bench:
        render_benchmark(cfg)


if __name__ == "__main__":
    main()
