from query import answer

TESTS = [
    "What are PM2.5 trends in Kathmandu?",
    "What does the literature say about EIA in Nepal?",
    "Impacts of hydropower on Koshi biodiversity?",
    "Community forest studies by district?",
    "Climate adaptation in Karnali?",
]

for q in TESTS:
    print("=" * 60)
    print("Q:", q)
    ans, ctx = answer(q)
    print("A:", ans[:400])
    print("Sources:", [f"{c['metadata']['paper_id']} p{c['metadata']['page']}" for c in ctx])