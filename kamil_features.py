def kamil_feature_11(dataset, author_paper_pairs):
    feature_dict = {}
    pa_data = dataset["paper_author"]

    print("Welcome to Kamil's feature!")
    for ap in author_paper_pairs:
        pa = ap[::-1]
        #print("PA pair",pa)
        paper_aff = pa_data[(pa_data["AuthorId"] == pa[1]) & (pa_data["PaperId"] == pa[0])]["Affiliation"].unique()
        #print("Paper aff--",paper_aff,"--")
        
        ta_aff = " ".join(pa_data[pa_data["AuthorId"] == pa[1]]["Affiliation"].unique())
        #print("Target aff--",ta_aff,"--")
        coa_aff = get_coauthor_aff(pa_data, pa[0])
        value = 0
        for i,aff in enumerate(coa_aff):
            if(aff != ""):
         #       print(i,"Co auth aff::",aff,"::")
                value += lcs(aff, ta_aff)
        feature_dict[pa] = value
        p#rint(pa," has ",value)
    return feature_dict