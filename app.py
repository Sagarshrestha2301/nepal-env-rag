import streamlit as st
from query import answer

st.set_page_config(page_title="Nepal Environmental RAG", layout="wide")
st.title("Nepal Environmental RAG — 100 Paper MVP")

query = st.text_input("Ask a question about Nepal environmental studies")

if st.button("Search") and query:
    with st.spinner("Retrieving and generating answer..."):
        ans, contexts = answer(query)

    st.markdown("### Answer")
    st.write(ans)

    st.markdown("### Sources")
    for c in contexts:
        meta = c["metadata"]
        st.markdown(f"**{meta['paper_id']} — page {meta['page']}**")
        st.caption(c["text"][:800] + "...")
        st.divider()