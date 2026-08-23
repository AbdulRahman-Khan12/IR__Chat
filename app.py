"""IR__Chat - Streamlit interface.

Run with:  streamlit run app.py

The interface has one idea behind it: never show an answer without showing
where it came from. Every reply carries the sentence it was taken from, the
document that sentence lives in, and the arithmetic that chose it. A QA system
that cannot be checked is a QA system that cannot be trusted, so the evidence
trail is the one thing given real visual weight here.

Everything else is deliberately quiet: two tabs, one sidebar, no decoration.
"""

from __future__ import annotations

import html

import streamlit as st

from ir_chat import __version__
from ir_chat.config import MAX_DOCS
from ir_chat.corpus import (
    CorpusError,
    CorpusLimitError,
    Document,
    load_bundled_corpus,
)
from ir_chat.dialogue import DialogueManager
from ir_chat.evaluate import evaluate, load_pairs
from ir_chat.extractor import entity_backend
from ir_chat.retriever import Retriever

st.set_page_config(
    page_title="IR__Chat",
    page_icon="\u25c8",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- styling -----------------------------------------------------------------
# Kept to one block. Three jobs only: the evidence card, the marked answer, and
# a monospace face for numbers so scores line up and read as data.
st.markdown(
    """
    <style>
      .stApp { font-feature-settings: "liga" 1; }
      .evidence {
        border-left: 3px solid #0F766E;
        background: #EEF3F3;
        padding: 0.7rem 0.9rem;
        margin: 0.35rem 0 0.6rem 0;
        font-size: 0.94rem;
        line-height: 1.55;
      }
      .evidence mark {
        background: #FCD34D;
        padding: 0 0.15em;
        border-radius: 2px;
      }
      .meta {
        font-family: ui-monospace, "SF Mono", Menlo, monospace;
        font-size: 0.76rem;
        letter-spacing: 0.02em;
        color: #55636F;
        text-transform: uppercase;
      }
      .num { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.83rem; }
      .answer { font-size: 1.28rem; line-height: 1.45; margin-bottom: 0.15rem; }
      .slots { font-family: ui-monospace, Menlo, monospace; font-size: 0.8rem; color: #55636F; }
      div[data-testid="stSidebarUserContent"] { padding-top: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- state -------------------------------------------------------------------
# Held in session_state rather than st.cache_resource: the corpus is mutable
# (users add and remove documents) and the index has to follow it. The corpus
# fingerprint is the trigger - rebuild only when the document set really changed.


def boot() -> None:
    if "corpus" not in st.session_state:
        corpus, notes = load_bundled_corpus()
        st.session_state.corpus = corpus
        st.session_state.notes = notes
        st.session_state.messages = []
        st.session_state.built = None
    corpus = st.session_state.corpus
    if st.session_state.built != corpus.fingerprint:
        st.session_state.manager = DialogueManager(corpus, retriever=Retriever(corpus))
        st.session_state.built = corpus.fingerprint


boot()
manager: DialogueManager = st.session_state.manager
corpus = st.session_state.corpus


def mark_answer(sentence: str, answer: str) -> str:
    """Highlight the answer inside its sentence, escaping everything else."""
    safe = html.escape(sentence)
    target = html.escape(answer)
    if target and target in safe:
        safe = safe.replace(target, f"<mark>{target}</mark>", 1)
    return safe


# =============================================================================
# Sidebar - the library
# =============================================================================

with st.sidebar:
    st.markdown("### Library")
    used = len(corpus)
    st.progress(used / MAX_DOCS)
    st.markdown(
        f"<div class='slots'>{used} of {MAX_DOCS} documents \u00b7 "
        f"{corpus.remaining_slots} slots free</div>",
        unsafe_allow_html=True,
    )

    if corpus.is_full:
        st.info("The library is full. Remove a document to add another.")
    else:
        uploaded = st.file_uploader(
            "Add documents",
            type=["txt", "md"],
            accept_multiple_files=True,
            help=f"Plain text or Markdown. Up to {MAX_DOCS} documents in total.",
        )
        if uploaded:
            added, refused = 0, []
            for item in uploaded:
                try:
                    corpus.add(
                        Document.from_text(
                            item.name,
                            item.getvalue().decode("utf-8", errors="replace"),
                            source="upload",
                        )
                    )
                    added += 1
                except CorpusLimitError as exc:
                    refused.append(str(exc))
                    break
                except CorpusError as exc:
                    refused.append(str(exc))
            if added:
                st.success(f"Added {added} document{'s' if added > 1 else ''}.")
                st.rerun()
            for message in refused:
                st.warning(message)

    st.divider()

    for doc in corpus.documents:
        left, get, drop = st.columns([6, 1, 1])
        with left:
            st.markdown(
                f"**{doc.title}**<br><span class='meta'>{doc.word_count} words "
                f"\u00b7 {doc.source}</span>",
                unsafe_allow_html=True,
            )
        with get:
            st.download_button(
                "\u2193",
                data=doc.download_payload(),
                file_name=doc.download_name,
                mime="text/plain",
                key=f"get-{doc.doc_id}",
                help=f"Download {doc.download_name}",
            )
        with drop:
            if st.button("\u00d7", key=f"drop-{doc.doc_id}", help="Remove from the library"):
                corpus.remove(doc.doc_id)
                st.rerun()

    st.divider()
    with st.expander("System"):
        info = {**manager.stats(), "entity recogniser": entity_backend(), "version": __version__}
        for key, value in info.items():
            st.markdown(
                f"<span class='meta'>{key}</span> "
                f"<span class='num'>{value}</span>",
                unsafe_allow_html=True,
            )


# =============================================================================
# Main
# =============================================================================

st.markdown("## IR__Chat")
st.caption(
    "Ask a question about the library. Every answer shows the sentence it came "
    "from and why that sentence won."
)

ask_tab, eval_tab = st.tabs(["Ask", "Evaluate"])

# --- Ask ---------------------------------------------------------------------

with ask_tab:
    if not st.session_state.messages:
        st.markdown("<span class='meta'>Try one of these</span>", unsafe_allow_html=True)
        columns = st.columns(3)
        for index, suggestion in enumerate(manager.suggestions()):
            if columns[index % 3].button(suggestion, key=f"s{index}", use_container_width=True):
                st.session_state.pending = suggestion
                st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["text"])
                continue

            reply = message["reply"]
            st.markdown(f"<div class='answer'>{html.escape(reply.text)}</div>",
                        unsafe_allow_html=True)

            badges = [f"via {reply.route}"]
            if reply.rewritten:
                badges.append(f"read as \u201c{reply.question}\u201d")
            if reply.fact:
                badges.append(reply.fact.doc_id)
            elif reply.answer and reply.answer.found:
                badges.append(f"{reply.answer.confidence:.2f} confidence")
                badges.append(reply.answer.doc_title)
            st.markdown(
                f"<span class='meta'>{' \u00b7 '.join(badges)}</span>",
                unsafe_allow_html=True,
            )

            if not reply.has_evidence:
                continue

            with st.expander("Evidence"):
                if reply.fact:
                    st.markdown("**Answered by lookup, not by reading text.**")
                    for line in reply.fact.explain():
                        st.markdown(f"<div class='num'>{html.escape(line)}</div>",
                                    unsafe_allow_html=True)
                    source = corpus.get(reply.fact.doc_id)
                    if source:
                        st.caption(f"Source document: {source.title}")

                answer = reply.answer
                if answer and answer.found:
                    st.markdown(
                        f"<div class='evidence'>"
                        f"{mark_answer(answer.sentence, answer.text)}</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"{answer.doc_title} \u00b7 {answer.passage_id}")
                    st.markdown("<span class='meta'>How the score was made"
                                "</span>", unsafe_allow_html=True)
                    for line in answer.explain():
                        st.markdown(f"<div class='num'>{html.escape(line)}</div>",
                                    unsafe_allow_html=True)
                    if answer.alternatives:
                        others = ", ".join(f"{t} ({s})" for t, s in answer.alternatives)
                        st.markdown(
                            f"<span class='meta'>runners-up</span> "
                            f"<span class='num'>{html.escape(others)}</span>",
                            unsafe_allow_html=True,
                        )

                if reply.results:
                    st.markdown("<span class='meta'>Passages considered</span>",
                                unsafe_allow_html=True)
                    for hit in reply.results[:3]:
                        st.markdown(
                            f"<div class='num'>{hit.rank}. {hit.score:.2f} "
                            f"\u00b7 {html.escape(hit.doc_title)}</div>"
                            f"<div style='font-size:0.86rem;color:#404C58;"
                            f"margin-bottom:0.4rem'>{html.escape(hit.snippet(150))}</div>",
                            unsafe_allow_html=True,
                        )

    typed = st.chat_input("Ask about the library\u2026")
    question = typed or st.session_state.pop("pending", None)

    if question:
        st.session_state.messages.append({"role": "user", "text": question})
        reply = manager.respond(question)
        st.session_state.messages.append({"role": "assistant", "reply": reply})
        st.rerun()

    if st.session_state.messages:
        if st.button("Clear conversation"):
            st.session_state.messages = []
            manager.reset()
            st.rerun()

# --- Evaluate ----------------------------------------------------------------

with eval_tab:
    pairs = load_pairs()
    st.markdown(
        f"Run all {len(pairs)} gold questions through the system. Retrieval and "
        "extraction are scored separately, because a pipeline can fail in either "
        "place and the fix is different each time."
    )

    if st.button("Run evaluation", type="primary"):
        with st.spinner("Answering every question\u2026"):
            st.session_state.report = evaluate(manager, pairs)

    report = st.session_state.get("report")
    if report:
        summary = report.summary()
        row_one = st.columns(4)
        row_one[0].metric("Recall@5", f"{summary['recall@k']:.0%}",
                          help="Did the right document reach the extractor?")
        row_one[1].metric("MRR", f"{summary['MRR']:.3f}",
                          help="How high did it rank, on average?")
        row_one[2].metric("Exact match", f"{summary['exact_match']:.0%}",
                          help="Character-for-character correct short answers.")
        row_one[3].metric("Token F1", f"{summary['token_f1']:.0%}",
                          help="Partial credit for overlapping words.")

        row_two = st.columns(3)
        row_two[0].metric("Answer found", f"{summary['answer_found']:.0%}",
                          help="The gold answer appears somewhere in the reply.")
        row_two[1].metric("Refusals correct", f"{summary['refusal_accuracy']:.0%}",
                          help="Questions the corpus cannot answer, correctly declined.")
        row_two[2].metric("Overall", f"{summary['overall_correct']:.0%}")

        st.markdown("**By question type**")
        st.dataframe(
            [
                {"type": name, "questions": stats["n"],
                 "correct": f"{stats['correct']:.0%}", "token F1": f"{stats['f1']:.2f}"}
                for name, stats in report.by_type().items()
            ],
            use_container_width=True, hide_index=True,
        )

        st.markdown("**Which route answered**")
        st.markdown(
            " \u00b7 ".join(f"{route}: {count}" for route, count in report.by_route().items())
        )

        failures = report.failures()
        st.markdown(f"**Failures ({len(failures)})**")
        if not failures:
            st.success("Every gold question was answered correctly.")
        else:
            for failure in failures:
                st.markdown(
                    f"<div class='num'>{html.escape(failure.question)}</div>"
                    f"<div style='font-size:0.85rem'>expected "
                    f"<em>{html.escape(str(failure.gold))}</em> \u00b7 got "
                    f"<em>{html.escape(failure.predicted[:120])}</em></div>",
                    unsafe_allow_html=True,
                )
