# functions should return the following format (dictionary)
# { (a1, p1): feature_value1, (a2, p2): feature_value2 ... }

def get_paper_has_how_many_author_in_PaperAuthor(dataset, author_paper_pairs):
    feature_dict = {}

    pid_to_count = dataset['paper_author']['PaperId'].value_counts()
    for ap_pair in author_paper_pairs:
        feature_dict[ap_pair] = pid_to_count[ap_pair[1]]

    return feature_dict

def get_author_publishes_how_many_paper_in_PaperAuthor(dataset, author_paper_pairs):
    feature_dict = {}

    aid_to_count = dataset['paper_author']['AuthorId'].value_counts()
    for ap_pair in author_paper_pairs:
        feature_dict[ap_pair] = aid_to_count[ap_pair[0]]

    return feature_dict

def kamil_new_f1(dataset, author_paper_pairs):
    feature_dict = {}

    #aid_to_count = dataset['paper_author']['AuthorId'].value_counts()
    print(author_paper_pairs[0])
    for ap_pair in author_paper_pairs:
        feature_dict[ap_pair] = int(ap_pair[0]) + int(ap_pair[1])

    return feature_dict