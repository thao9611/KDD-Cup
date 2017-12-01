def kamil_new_f1(dataset, author_paper_pairs):
    feature_dict = {}

    #aid_to_count = dataset['paper_author']['AuthorId'].value_counts()
    print(author_paper_pairs[0])
    for ap_pair in author_paper_pairs:
        feature_dict[ap_pair] = int(ap_pair[0]) + int(ap_pair[1])

    return feature_dict