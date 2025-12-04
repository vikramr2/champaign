import champaign
import pandas as pd

def leiden_to_df(leiden_result):
    clusters = leiden_result['clusters']
    cluster_assignments = []
    
    for cluster_id, nodes in enumerate(clusters):
        cluster_assignments.extend(list(zip(nodes, [cluster_id]*len(nodes))))
    
    df = pd.DataFrame(cluster_assignments, columns=['node', 'cluster'])
    
    return df


if __name__ == "__main__":
    g = champaign.from_tsv("../data/cit_hepph.tsv")
    print(g)

    # Run champaign with verbose output
    champaign_result = champaign.champaign(g, verbose=False)

    # Fetch leiden clusters from the result
    leiden_clusters = champaign_result.leiden(g, gamma=0.5)

    print("Champaign Result:", champaign_result)
    print("Num Leiden Clusters:", leiden_clusters['num_clusters'])

    # Save the dendrogram to a json file
    champaign_result.save_json("champaign_hepph_result.json")

    # Get a DataFrame of leiden cluster assignments
    df_leiden = leiden_to_df(leiden_clusters)

    # Save the DataFrame to a TSV file
    df_leiden.to_csv("hepph_leiden_clusters.tsv", sep="\t", index=False, header=False)
