
def lcs(X , Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)
 
    # declaring the array for storing the dp values
    L = np.zeros((m+1,n+1))
 
    """Following steps build L[m+1][n+1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0 :
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j] , L[i][j-1])
 
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]

def get_coauthor_aff(pa_data,pid):
    related_authors = pa_data[pa_data['PaperId'] == pid]['Affiliation']
    return related_authors.unique()

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