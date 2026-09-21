import streamlit as st
from query import answer, list_categories

st.set_page_config(page_title="Nepal Environment RAG", layout="wide")
st.title("Nepal Environmental Studies — RAG")
st.caption("Offline. Answers cite paper ID and page.")

cats = ["(any)"] + list_categories()

with st.sidebar:
    st.header("Filters")
    category = st.selectbox("Category", cats)
    theme = st.selectbox(
        "Theme",
        ["(any)", "glof", "flood", "landslide", "earthquake", "glacier",
         "climate", "forest", "biodiversity", "forest_fire", "water",
         "hydropower", "agriculture", "livelihood", "air", "urban_heat",
         "land_use", "policy", "geology", "health", "disaster"],
    )
    basin = st.selectbox(
        "River basin",
        ["(any)", "koshi", "gandaki", "karnali", "bagmati", "rapti",
         "kamala", "trishuli", "narayani", "mahakali"],
    )
    year_min = st.slider("Min year", 1970, 2030, 1970, step=5)

cat_v = None if category == "(any)" else category
theme_v = None if theme == "(any)" else theme
basin_v = None if basin == "(any)" else basin
year_v = None if year_min <= 1970 else year_min

query = st.text_input("Ask a question about Nepal's environment")

if st.button("Search") and query:
    with st.spinner("Retrieving and generating answer..."):
        ans, contexts = answer(query, category=cat_v, theme=theme_v,
                               basin=basin_v, year_min=year_v)
    st.markdown("### Answer")
    st.write(ans)

    st.markdown("### Sources")
    for c in contexts:
        m = c["metadata"]
        st.markdown(
            f"**{m['paper_id']}** — p.{m['page']}  \n"
            f"category: `{m.get('category','')}` | year: `{m.get('year','?')}` "
            f"| themes: `{m.get('themes','')}` | basin: `{m.get('basins','')}`"
        )
        st.caption(c["text"][:800] + "...")
        st.divider()