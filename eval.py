from query import answer

TESTS = [
    # GLOF-specific
    "What causes glacial lake outburst floods in Nepal?",
    "Which glacial lakes in Nepal are considered high risk for GLOF?",
    "What are the impacts of GLOFs on downstream communities?",
    "How is GLOF risk assessed and monitored?",
    "What role does climate change play in GLOF frequency?",

    # Flood hydrology
    "What are the major flood events recorded in Nepal?",
    "What is the relationship between catchment area and peak flood discharge?",
    "What rainfall intensities cause extreme floods in Nepal?",
    "Which river basins in Nepal are most flood-prone?",
    "What is the return period of the 1993 floods in Bagmati?",

    # Flood impacts and management
    "What are the socio-economic impacts of floods in Nepal?",
    "How effective are early warning systems for floods in Nepal?",
    "What structural measures are used for flood control in Nepal?",
    "What role do community forests play in reducing flood risk?",
    "How does land use change affect flood risk in Nepal?",

    # Specific basins / regions
    "What does the literature say about Koshi River floods?",
    "What are the flood characteristics of the Karnali basin?",
    "What studies exist on floods in the Terai region?",
    "What is known about flash floods in the middle hills of Nepal?",

    # Policy and adaptation
    "What policies govern flood disaster management in Nepal?",
    "What adaptation strategies are proposed for flood-prone areas?",
]

for i, q in enumerate(TESTS, 1):
    print("=" * 70)
    print(f"Q{i}: {q}")
    ans, ctx = answer(q)
    print(f"\nA: {ans}\n")
    print("Sources:")
    for c in ctx:
        m = c["metadata"]
        print(f"  - {m['paper_id']} p.{m['page']}")
    print()