def kamil_feature_11(dataset, author_paper_pairs):
    feature_dict = {}
    pa_data = dataset["paper_author"].set_index('AuthorId')
    
    print("Welcome to Kamil's feature!")
    print("AP shape",len(author_paper_pairs))
    print("type",type(author_paper_pairs[0][0]))
    
    for ap in author_paper_pairs:
        #print("PA pair",ap)
        #paper_aff = pa_data.loc[(pa_data.loc[ap[0]]["PaperId"] == ap[1])]["Affiliation"].unique()
        #print("Paper aff--",paper_aff,"--")
        
        ta_aff = " ".join(pa_data.loc[ap[0]]["Affiliation"].unique())
        #print("Target aff--",ta_aff,"--")
        coa_aff = get_coauthor_aff(pa_data, ap[1])
        value = 0
        for i,aff in enumerate(coa_aff):
            if(aff != ""):
         #       print(i,"Co auth aff::",aff,"::")
                value += lcs(aff, ta_aff)
        feature_dict[ap] = value
        #print(ap," -> ",value)