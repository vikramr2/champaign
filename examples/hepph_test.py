import champaign

if __name__ == "__main__":
    g = champaign.from_tsv("../data/cit_hepph.tsv")
    print(g)

    # Run champaign with verbose output
    champaign_result = champaign.champaign(g, verbose=True)

    # Fetch leiden clusters from the result
    leiden_clusters = champaign_result.leiden(g, gamma=0.5)

    print("Champaign Result:", champaign_result)
    print("Leiden Clusters:", leiden_clusters)
