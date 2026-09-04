"""Streamlit chat UI for Handbook Copilot.

Run: streamlit run app.py
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from core.llm import AnthropicProvider, LLMProvider, Message, OllamaProvider, OpenAIProvider
from core.parts import (
    DEFAULT_PART,
    create_part,
    delete_part as registry_delete_part,
    list_parts,
    pdf_dir,
    validate_name,
)
from core.pipeline import ingest_pdf
from core.prompts import SYSTEM, build_user_prompt
from core.retriever import retrieve
from core.store import Store

load_dotenv()

st.set_page_config(page_title="Handbook Copilot", page_icon="📘", layout="wide")


# ------------------------------------------------------------------ providers

def _build_provider(choice: str, anthropic_key: str, openai_key: str) -> LLMProvider:
    if choice == "Ollama (local)":
        return OllamaProvider()
    if choice == "Anthropic (Claude)":
        return AnthropicProvider(api_key=anthropic_key or os.getenv("ANTHROPIC_API_KEY"))
    return OpenAIProvider(api_key=openai_key or os.getenv("OPENAI_API_KEY"))


# ------------------------------------------------------------------ resources

@st.cache_resource
def get_store() -> Store:
    """One Store instance per Streamlit process — Chroma clients aren't cheap."""
    return Store()


# ------------------------------------------------------------------ ingest

def _handle_uploads(uploaded_files, part: str, force: bool) -> None:
    """Save uploaded PDFs under data/pdfs/<part>/ and index them into the part."""
    store = get_store()
    target_dir = pdf_dir(part)
    status_area = st.status(
        f"Ingesting {len(uploaded_files)} file(s) into '{part}'…", expanded=True
    )

    with status_area:
        for uf in uploaded_files:
            target = target_dir / uf.name
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

            result = ingest_pdf(target, store, part=part, force=force, progress=progress)
            bar.empty()

            if result.skipped:
                st.info("↳ already indexed in this part — enable *Re-ingest existing* to overwrite.")
            elif result.error:
                st.error(f"↳ failed: {result.error}")
            else:
                st.success(f"↳ {result.pages} pages → {result.chunks} chunks indexed.")

    status_area.update(label="Ingest complete.", state="complete", expanded=False)


# ------------------------------------------------------------------ session

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "active_part" not in st.session_state:
    st.session_state.active_part = DEFAULT_PART


# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.title("📘 Handbook Copilot")
    st.caption("Local-first RAG over your PDFs")

    store = get_store()
    parts = list_parts()

    # ---- Part selector ------------------------------------------
    st.subheader("Part")
    if st.session_state.active_part not in parts:
        st.session_state.active_part = DEFAULT_PART

    active_part = st.selectbox(
        "Active part",
        parts,
        index=parts.index(st.session_state.active_part),
        label_visibility="collapsed",
        help="A Part is a group of documents (e.g. 'Data Structures', 'Mathematics').",
    )
    if active_part != st.session_state.active_part:
        st.session_state.active_part = active_part
        st.session_state.messages = []  # fresh chat when switching parts
        st.rerun()

    part_chunks = store.count(part=active_part)
    part_sources = sorted(store.sources(part=active_part))
    col_a, col_b = st.columns(2)
    col_a.metric("Documents", len(part_sources))
    col_b.metric("Chunks", part_chunks)

    with st.expander("Manage parts", expanded=False):
        new_name = st.text_input("New part name", key="new_part_input",
                                 placeholder="e.g. Machine Learning")
        if st.button("Create part", use_container_width=True):
            try:
                created = create_part(new_name)
                st.session_state.active_part = created
                st.session_state.messages = []
                st.rerun()
            except ValueError as e:
                st.error(str(e))

        st.divider()
        if active_part == DEFAULT_PART:
            st.caption("The Default part cannot be deleted.")
        else:
            st.write(f"Delete part **{active_part}**")
            confirm = st.checkbox(
                f"I understand this removes all {len(part_sources)} document(s) and "
                f"{part_chunks} chunk(s) from '{active_part}'.",
                key=f"confirm-del-{active_part}",
            )
            if st.button("Delete this part", type="secondary",
                         use_container_width=True, disabled=not confirm):
                store.delete_part(active_part)
                registry_delete_part(active_part)
                folder = pdf_dir(active_part)
                if folder.exists():
                    shutil.rmtree(folder, ignore_errors=True)
                st.session_state.active_part = DEFAULT_PART
                st.session_state.messages = []
                st.rerun()

    st.divider()

    # ---- Upload -------------------------------------------------
    st.subheader(f"Add documents to '{active_part}'")
    uploaded = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader-{active_part}-{st.session_state.uploader_key}",
        help=f"Files are saved to data/pdfs/{active_part}/ and indexed into this part.",
    )
    force_reingest = st.checkbox(
        "Re-ingest if already present", value=False,
        help="Overwrite the stored chunks for any file with the same name in this part.",
    )
    if uploaded and st.button("Ingest uploaded files", type="primary", use_container_width=True):
        _handle_uploads(uploaded, part=active_part, force=force_reingest)
        st.session_state.uploader_key += 1  # clears the dropzone on rerun
        st.rerun()

    # ---- Sources list -------------------------------------------
    with st.expander(f"Documents in '{active_part}' ({len(part_sources)})", expanded=False):
        if not part_sources:
            st.info("No documents yet. Drop a PDF above to get started.")
        else:
            for s in part_sources:
                c1, c2 = st.columns([4, 1])
                c1.write(f"• {s}")
                if c2.button("✕", key=f"del-{active_part}-{s}",
                             help=f"Remove {s} from '{active_part}'"):
                    store.delete_source(s, part=active_part)
                    # Best-effort remove of the saved file too.
                    try:
                        (pdf_dir(active_part) / s).unlink(missing_ok=True)
                    except OSError:
                        pass
                    st.rerun()

    st.divider()

    # ---- Retrieval scope ----------------------------------------
    search_all = st.checkbox(
        "Search across all parts",
        value=False,
        help="Off: answers use only the active part. On: retrieval spans every part.",
    )

    # ---- Model --------------------------------------------------
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

scope_label = "all parts" if search_all else f"part: **{active_part}**"
st.title("Ask the handbook")
st.caption(f"Answering from {scope_label}")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("citations"):
            with st.expander(f"Sources ({len(m['citations'])})"):
                for h in m["citations"]:
                    st.markdown(
                        f"**[{h.part}]** {h.source} — page {h.page_number}"
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

    scope_part = None if search_all else active_part
    total_in_scope = store.count(part=scope_part)

    if total_in_scope == 0:
        with st.chat_message("assistant"):
            msg = (f"No documents indexed in '{active_part}' yet. "
                   "Drop a PDF into the sidebar uploader and click **Ingest uploaded files**.")
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
                    retrieved = retrieve(store, question, final_k=top_k, part=scope_part)
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
                                f"**[{h.part}]** {h.source} — page {h.page_number}"
                                + (f" — §{h.section_path}" if h.section_path else "")
                                + f"  \n_score: {h.score:.2f}_"
                            )
                            st.caption(h.text[:400] + ("…" if len(h.text) > 400 else ""))
                            st.divider()

                st.session_state.messages.append(
                    {"role": "assistant", "content": acc, "citations": retrieved.hits}
                )
