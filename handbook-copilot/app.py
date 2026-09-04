"""Streamlit chat UI for Handbook Copilot.

Run: streamlit run app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from core.llm import AnthropicProvider, LLMProvider, Message, OllamaProvider, OpenAIProvider
from core.pipeline import ingest_pdf
from core.prompts import SYSTEM, build_user_prompt
from core.retriever import retrieve
from core.store import Store

load_dotenv()

st.set_page_config(page_title="Handbook Copilot", page_icon="📘", layout="wide")

PDF_DIR = Path(__file__).resolve().parent / "data" / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ providers

def _build_provider(choice: str, anthropic_key: str, openai_key: str) -> LLMProvider:
    if choice == "Ollama (local)":
        return OllamaProvider()
    if choice == "Anthropic (Claude)":
        return AnthropicProvider(api_key=anthropic_key or os.getenv("ANTHROPIC_API_KEY"))
    return OpenAIProvider(api_key=openai_key or os.getenv("OPENAI_API_KEY"))


# ------------------------------------------------------------------ ingest

@st.cache_resource
def get_store() -> Store:
    """One Store instance per Streamlit process — Chroma clients aren't cheap."""
    return Store()


def _handle_uploads(uploaded_files, force: bool) -> None:
    """Save uploaded PDFs to data/pdfs/ and run them through the pipeline."""
    store = get_store()
    status_area = st.status(f"Ingesting {len(uploaded_files)} file(s)…", expanded=True)

    with status_area:
        for uf in uploaded_files:
            # Save to disk so re-runs / CLI can pick them up too.
            target = PDF_DIR / uf.name
            target.write_bytes(uf.getbuffer())

            st.write(f"**{uf.name}**")
            bar = st.progress(0.0, text="parsing…")

            def progress(stage: str, cur: int, total: int) -> None:
                if stage == "parsing":
                    bar.progress(0.05, text="parsing PDF…")
                elif stage == "chunking":
                    bar.progress(0.1, text="chunking…")
                elif stage == "embedding" and total > 0:
                    frac = 0.1 + 0.9 * (cur / total)
                    bar.progress(frac, text=f"embedding {cur}/{total} chunks…")

            result = ingest_pdf(target, store, force=force, progress=progress)
            bar.empty()

            if result.skipped:
                st.info(f"↳ already ingested — enable *Re-ingest existing* to overwrite.")
            elif result.error:
                st.error(f"↳ failed: {result.error}")
            else:
                st.success(
                    f"↳ {result.pages} pages → {result.chunks} chunks indexed."
                )

    status_area.update(label="Ingest complete.", state="complete", expanded=False)


# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.title("📘 Handbook Copilot")
    st.caption("Local-first RAG over your PDFs")

    store = get_store()
    n_chunks = store.count()
    sources = sorted(store.sources())
    st.metric("Indexed chunks", n_chunks)

    # ---- Upload ---------------------------------------------------
    st.subheader("Add documents")
    uploaded = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        help="Files are saved to data/pdfs/ and indexed immediately.",
    )
    force_reingest = st.checkbox(
        "Re-ingest if already present", value=False,
        help="Overwrite the stored chunks for any file with the same name.",
    )
    if uploaded and st.button("Ingest uploaded files", type="primary", use_container_width=True):
        _handle_uploads(uploaded, force=force_reingest)
        st.rerun()

    # ---- Sources list --------------------------------------------
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        if not sources:
            st.info("No documents indexed yet. Drop a PDF above to get started.")
        else:
            for s in sources:
                col1, col2 = st.columns([4, 1])
                col1.write(f"• {s}")
                if col2.button("✕", key=f"del-{s}", help=f"Remove {s} from the index"):
                    store.delete_source(s)
                    st.rerun()

    st.divider()

    # ---- Model ---------------------------------------------------
    st.subheader("Model")
    provider_choice = st.radio(
        "LLM provider",
        ["Ollama (local)", "Anthropic (Claude)", "OpenAI"],
        index=0,
        help="Ollama runs locally with no API cost. Others need an API key.",
    )
    anthropic_key = st.text_input(
        "Anthropic API key",
        type="password",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        help="Only sent to Anthropic when you use the Claude provider.",
    ) if provider_choice == "Anthropic (Claude)" else ""
    openai_key = st.text_input(
        "OpenAI API key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
    ) if provider_choice == "OpenAI" else ""

    st.divider()
    top_k = st.slider("Excerpts per answer", 3, 8, 5)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ------------------------------------------------------------------ chat

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Ask the handbook")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("citations"):
            with st.expander(f"Sources ({len(m['citations'])})"):
                for h in m["citations"]:
                    st.markdown(
                        f"**{h.source}** — page {h.page_number}"
                        + (f" — §{h.section_path}" if h.section_path else "")
                        + f"  \n_score: {h.score:.2f}_"
                    )
                    st.caption(h.text[:400] + ("…" if len(h.text) > 400 else ""))
                    st.divider()

question = st.chat_input("Ask a question about your documents…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if n_chunks == 0:
        with st.chat_message("assistant"):
            msg = ("No documents are indexed yet. Drop a PDF into the sidebar uploader "
                   "and click **Ingest uploaded files**.")
            st.warning(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg, "citations": []})
    else:
        provider = _build_provider(provider_choice, anthropic_key, openai_key)
        if not provider.available():
            hint = {
                "ollama": "Ollama isn't reachable at $OLLAMA_HOST. Install it and run `ollama serve`.",
                "anthropic": "Paste your Anthropic API key in the sidebar.",
                "openai": "Paste your OpenAI API key in the sidebar.",
            }[provider.name]
            with st.chat_message("assistant"):
                st.error(hint)
            st.session_state.messages.append(
                {"role": "assistant", "content": hint, "citations": []}
            )
        else:
            with st.chat_message("assistant"):
                with st.spinner("Searching handbook…"):
                    retrieved = retrieve(store, question, final_k=top_k)
                messages = [
                    Message(role="system", content=SYSTEM),
                    Message(role="user", content=build_user_prompt(question, retrieved.hits)),
                ]
                placeholder = st.empty()
                acc = ""
                try:
                    for tok in provider.generate(messages):
                        acc += tok
                        placeholder.markdown(acc + "▌")
                    placeholder.markdown(acc)
                except Exception as e:
                    acc = f"⚠️ LLM error: {e}"
                    placeholder.error(acc)

                if retrieved.hits:
                    with st.expander(f"Sources ({len(retrieved.hits)})"):
                        for h in retrieved.hits:
                            st.markdown(
                                f"**{h.source}** — page {h.page_number}"
                                + (f" — §{h.section_path}" if h.section_path else "")
                                + f"  \n_score: {h.score:.2f}_"
                            )
                            st.caption(h.text[:400] + ("…" if len(h.text) > 400 else ""))
                            st.divider()

                st.session_state.messages.append(
                    {"role": "assistant", "content": acc, "citations": retrieved.hits}
                )
