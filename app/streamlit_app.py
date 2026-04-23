"""Streamlit UI for the Claims RAG Assistant."""

import sys
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from src.rag_pipeline import ask

st.set_page_config(
    page_title="Claims Document Intelligence",
    page_icon="📋",
    layout="wide",
)

# Header
st.title("📋 Claims Document Intelligence")
st.caption(
    "RAG-powered assistant for Australian Workers Compensation queries. "
    "Answers are grounded in source documents and cited."
)

# Disclaimer — important for responsible AI framing
with st.expander("⚠️ About this tool (read before use)"):
    st.markdown("""
    **What this is:** A decision-support tool that retrieves relevant passages from
    workers compensation legislation and policy documents, then uses Claude to
    generate grounded answers with citations.

    **What this is not:** A replacement for qualified legal, medical, or actuarial
    advice. Always verify critical decisions against the source documents directly.

    **Data used:** Publicly available legislation, regulatory guidelines, and
    clearly-marked synthetic claim narratives. No real claimant data.

    **Limitations:** The system can only answer from its indexed document corpus.
    Questions outside that scope will be declined.
    """)

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander(f"📚 {len(message['sources'])} sources used"):
                for s in message["sources"]:
                    st.markdown(
                        f"**[{s['index']}] {s['file']}** · relevance: {s['score']:.3f}"
                    )
                    st.caption(s["preview"] + "...")

# Chat input
if prompt := st.chat_input("Ask a question about workers compensation..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            result = ask(prompt)
        st.markdown(result["answer"])
        with st.expander(f"📚 {len(result['sources'])} sources used"):
            for s in result["sources"]:
                st.markdown(
                    f"**[{s['index']}] {s['file']}** · relevance: {s['score']:.3f}"
                )
                st.caption(s["preview"] + "...")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })