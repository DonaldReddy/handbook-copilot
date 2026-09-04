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
from core.pipeline import ingest_pdf
from core.pods import (
    DEFAULT_POD,
    create_pod,
    delete_pod as registry_delete_pod,
    list_pods,
    pdf_dir,
    rename_pod as registry_rename_pod,
    validate_name,
)
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


# ------------------------------------------------------------------ pod ops

def _rename_pod_everywhere(old: str, new: str, store: Store) -> None:
    """Rename in registry + chunk metadata + on-disk pdf folder."""
    new = validate_name(new)
    # Registry does its own validation (duplicates, Default guard).
    registry_rename_pod(old, new)
    store.rename_pod(old, new)
    old_dir = pdf_dir(old)  # creates it if missing, harmless
    new_dir = pdf_dir(new)
    # If the destination is empty, move; otherwise merge files across.
    if old_dir.exists() and old_dir.resolve() != new_dir.resolve():
        for f in old_dir.iterdir():
            target = new_dir / f.name
            if target.exists():
                target.unlink()
            shutil.move(str(f), str(target))
        try:
            old_dir.rmdir()
        except OSError:
            pass


# ------------------------------------------------------------------ ingest

def _handle_uploads(uploaded_files, pod: str, force: bool) -> None:
    """Save uploaded PDFs under data/pdfs/<pod>/ and index them into the pod."""
    store = get_store()
    target_dir = pdf_dir(pod)
    status_area = st.status(
        f"Ingesting {len(uploaded_files)} file(s) into '{pod}'…", expanded=True
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

            result = ingest_pdf(target, store, pod=pod, force=force, progress=progress)
            bar.empty()

            if result.skipped:
                st.info("↳ already indexed in this pod — enable *Re-ingest existing* to overwrite.")
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
if "active_pod" not in st.session_state:
    st.session_state.active_pod = DEFAULT_POD


# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.title("📘 Handbook Copilot")
    st.caption("Local-first RAG over your PDFs")

    store = get_store()
    pods = list_pods()

    # ---- Pod selector ------------------------------------------
    st.subheader("Pod")
    if st.session_state.active_pod not in pods:
        st.session_state.active_pod = DEFAULT_POD

    active_pod = st.selectbox(
        "Active pod",
        pods,
        index=pods.index(st.session_state.active_pod),
        label_visibility="collapsed",
        help="A Pod is a group of documents (e.g. 'Data Structures', 'Mathematics').",
    )
    if active_pod != st.session_state.active_pod:
        st.session_state.active_pod = active_pod
        st.session_state.messages = []  # fresh chat when switching pods
        st.rerun()

    pod_chunks = store.count(pod=active_pod)
    pod_sources = sorted(store.sources(pod=active_pod))
    col_a, col_b = st.columns(2)
    col_a.metric("Documents", len(pod_sources))
    col_b.metric("Chunks", pod_chunks)

    with st.expander("Manage pods", expanded=False):
        # Create
        st.markdown("**Create pod**")
        new_name = st.text_input("New pod name", key="new_pod_input",
                                 placeholder="e.g. Machine Learning",
                                 label_visibility="collapsed")
        if st.button("Create pod", use_container_width=True):
            try:
                created = create_pod(new_name)
                st.session_state.active_pod = created
                st.session_state.messages = []
                st.rerun()
            except ValueError as e:
                st.error(str(e))

        st.divider()

        # Rename
        st.markdown(f"**Rename '{active_pod}'**")
        if active_pod == DEFAULT_POD:
            st.caption("The Default pod cannot be renamed.")
        else:
            rename_to = st.text_input(
                "New name", key=f"rename-input-{active_pod}",
                placeholder="New pod name", label_visibility="collapsed",
            )
            if st.button("Rename pod", use_container_width=True,
                         disabled=not rename_to.strip()):
                try:
                    _rename_pod_everywhere(active_pod, rename_to, store)
                    st.session_state.active_pod = rename_to.strip()
                    st.session_state.messages = []
                    st.success(f"Renamed to '{rename_to.strip()}'.")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

        st.divider()

        # Delete
        st.markdown(f"**Delete '{active_pod}'**")
        if active_pod == DEFAULT_POD:
            st.caption("The Default pod cannot be deleted.")
        else:
            confirm = st.checkbox(
                f"I understand this removes {len(pod_sources)} document(s) and "
                f"{pod_chunks} chunk(s).",
                key=f"confirm-del-{active_pod}",
            )
            if st.button("Delete this pod", type="secondary",
                         use_container_width=True, disabled=not confirm):
                store.delete_pod(active_pod)
                registry_delete_pod(active_pod)
                folder = pdf_dir(active_pod)
                if folder.exists():
                    shutil.rmtree(folder, ignore_errors=True)
                st.session_state.active_pod = DEFAULT_POD
                st.session_state.messages = []
                st.rerun()

    st.divider()

    # ---- Upload -------------------------------------------------
    st.subheader(f"Add documents to '{active_pod}'")
    uploaded = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader-{active_pod}-{st.session_state.uploader_key}",
        help=f"Files are saved to data/pdfs/{active_pod}/ and indexed into this pod.",
    )
    force_reingest = st.checkbox(
        "Re-ingest if already present", value=False,
        help="Overwrite the stored chunks for any file with the same name in this pod.",
    )
    if uploaded and st.button("Ingest uploaded files", type="primary", use_container_width=True):
        _handle_uploads(uploaded, pod=active_pod, force=force_reingest)
        st.session_state.uploader_key += 1  # clears the dropzone on rerun
        st.rerun()

    # ---- Sources list -------------------------------------------
    with st.expander(f"Documents in '{active_pod}' ({len(pod_sources)})", expanded=False):
        if not pod_sources:
            st.info("No documents yet. Drop a PDF above to get started.")
        else:
            for s in pod_sources:
                c1, c2 = st.columns([4, 1])
                c1.write(f"• {s}")
                if c2.button("✕", key=f"del-{active_pod}-{s}",
                             help=f"Remove {s} from '{active_pod}'"):
                    store.delete_source(s, pod=active_pod)
                    try:
                        (pdf_dir(active_pod) / s).unlink(missing_ok=True)
                    except OSError:
                        pass
                    st.rerun()

    st.divider()

    # ---- Retrieval scope ----------------------------------------
    search_all = st.checkbox(
        "Search across all pods",
        value=False,
        help="Off: answers use only the active pod. On: retrieval spans every pod.",
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

scope_label = "all pods" if search_all else f"pod: **{active_pod}**"
st.title("Ask the handbook")
st.caption(f"Answering from {scope_label}")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("citations"):
            with st.expander(f"Sources ({len(m['citations'])})"):
                for h in m["citations"]:
                    st.markdown(
                        f"**[{h.pod}]** {h.source} — page {h.page_number}"
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

    scope_pod = None if search_all else active_pod
    total_in_scope = store.count(pod=scope_pod)

    if total_in_scope == 0:
        with st.chat_message("assistant"):
            msg = (f"No documents indexed in '{active_pod}' yet. "
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
                    retrieved = retrieve(store, question, final_k=top_k, pod=scope_pod)
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
                                f"**[{h.pod}]** {h.source} — page {h.page_number}"
                                + (f" — §{h.section_path}" if h.section_path else "")
                                + f"  \n_score: {h.score:.2f}_"
                            )
                            st.caption(h.text[:400] + ("…" if len(h.text) > 400 else ""))
                            st.divider()

                st.session_state.messages.append(
                    {"role": "assistant", "content": acc, "citations": retrieved.hits}
                )
