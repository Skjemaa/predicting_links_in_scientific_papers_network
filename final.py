#importing models
import random
import numpy as np
import igraph
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn import preprocessing
import nltk
import csv
from sklearn.metrics import f1_score
from sklearn.decomposition import TruncatedSVD, NMF
import scipy as sc
from  sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd
from sklearn.svm import SVC

# choosing stemmer and stopwords
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()

#loading the training set
with open("training_set.txt", "r") as f:
    reader = csv.reader(f)
    training_set  = list(reader)


training_set = [element[0].split(" ") for element in training_set]
training_set = np.array([np.array(t) for t in training_set])

#loading the articles informations
with open("node_information.csv", "r") as f:
    reader = csv.reader(f)
    node_info  = list(reader)   

IDs = [element[0] for element in node_info]
corpus = [element[5] for element in node_info]
vectorizer = TfidfVectorizer(stop_words="english")
# each row is a node in the order of node_info

#TFIDF and LSA features
features_TFIDF = vectorizer.fit_transform(corpus)
lsa=TruncatedSVD(n_components=25,n_iter=5)
LSA = lsa.fit_transform(features_TFIDF)
titles = [element[2] for element in node_info]
vectorizer2 = TfidfVectorizer(stop_words="english")
# each row is a node in the order of node_info
titles_TFIDF = vectorizer2.fit_transform(titles)
titles_TFIDF = titles_TFIDF.toarray()

#not used here, but used before to train only on a subset of the training set
to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set)*1)))
training_set_reduced = [training_set[i] for i in to_keep[:len(to_keep)*1]]

#Opening testing set
with open("testing_set.txt", "r") as f:
    reader = csv.reader(f)
    testing_set  = list(reader)

testing_set = [element[0].split(" ") for element in testing_set]
edges = [(element[0],element[1]) for element in training_set_reduced if element[2]=="1"]
nodes = IDs

print("Starting graph")
## create empty directed graph
g = igraph.Graph(directed=True)
## add vertices
g.add_vertices(nodes)
## add edges
g.add_edges(edges)
pageranks=g.personalized_pagerank(damping=0.5)
print("graph constructed")

#Cleaning authors names
print("Starting remodeling authors")
for t in range(len(node_info)):
    authors = node_info[t][3].split(", ")
    aux = ''
    authors_cons = []
    for i in range(len(authors)):
        if (authors[i]==''):
            aux = authors[i]
        if(authors[i].find('(')<0 and authors[i].find(')')<0):
            authors_cons.append(authors[i])
        elif(authors[i].find('(')<0 and authors[i].find(')')>=0):
            aux = authors[i]
        elif(authors[i].find('(')>=0):
            authors_cons.append(authors[i].split(" (")[0])
        elif (authors[i].find('.')>=0 and authors[i].find(' ')):
            l = authors[i].split('. ')
            if (len(l)<=2):
                authors_cons.append(authors[i])
            if(len(l)>2):
                st = '.'.join(l[:-1])
                st = '. '.join([st,l[(len(l)-1)]])
                authors_cons.append(authors[i])
    node_info[t][3] = ', '.join(authors_cons)
    if(t % 500 == True):
        print(t, " articles processed")

# creating the dictionnay for the " number of citations between authors "  feature.
print("Starting quotations dictionnary")
quotations={}
counter = 1
for v in g.vs:
    source = v['name']
    neig = np.array(IDs)[g.neighbors(v)]
    source_info = [element for element in node_info if element[0]==source][0]
    source_auth = np.array(source_info[3].split(","))
    for auth in source_auth:
        for n in neig:
            target_info = [element for element in node_info if element[0]==n][0]
            target_auth = target_info[3].split(",")
            if auth in quotations.keys():
                for t_auth in target_auth:
                    if t_auth in quotations[auth].keys():
                        quotations[auth][t_auth]+=1
                    else:
                        quotations[auth][t_auth]=1
            else:
                for t_auth in target_auth:
                    quotations[auth]={}
                    quotations[auth][t_auth]=1
    counter += 1
    if counter % 200 == True:
        print (counter, "authors processsed")
print("End of the quotations dictionnary")
#initializing empty list for features
tfidf_distance = []
lsa_distance = []
lsa_distance_euc = []
authorsQuotation = []
page_rank=[]
commonNeighbors =[]
overlap_title=[]
overlap_corpus = []
temp_diff=[]
comm_auth=[]
tfidf_titles = []
for i in range(len(training_set)):
    
    source = training_set_reduced[i][0]
    target = training_set_reduced[i][1]
    
    index_source = IDs.index(source) # stores indexes in the graph g
    index_target = IDs.index(target)  # stores indexes in the graph g
    
    source_info = [element for element in node_info if element[0]==source][0] #find info of the node, the final [0] is used to the elemnet( list of node info ) out of another list containing it
    target_info = [element for element in node_info if element[0]==target][0]
    
    source_corpus = source_info[5].lower().split()
    source_corpus = [token for token in source_corpus if token not in stpwds]
    source_corpus = [stemmer.stem(token) if len(token)>3 else token for token in source_corpus]
    target_corpus = target_info[5].lower().split()
    target_corpus = [token for token in target_corpus if token not in stpwds]
    target_corpus = [stemmer.stem(token) if len(token)>3 else token for token in target_corpus]
	# convert to lowercase and tokenize
    source_title = source_info[2].lower().split(" ")
	# remove stopwords
    source_title = [token for token in source_title if token not in stpwds]
    source_title = [stemmer.stem(token) for token in source_title]
    target_title = target_info[2].lower().split(" ")
    target_title = [token for token in target_title if token not in stpwds]
    target_title = [stemmer.stem(token) for token in target_title]
    
    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")
    
    neigh_source = g.neighbors(index_source)
    neigh_target = g.neighbors(index_target)
    
    comm_neigh = len(set(neigh_source).intersection(set(neigh_target)))
    commonNeighbors.append(comm_neigh)
    #number of citations between authors
    authors_quotation = 0
    for s_author in source_auth:
        for t_author in target_auth:
            if s_author in quotations.keys():
                if t_author in quotations[s_author].keys():
                    authors_quotation+= quotations[s_author][t_author]


    source_page_rank=pageranks[index_source]
    target_page_rank=pageranks[index_target]

    tfidf_titles.append(sc.spatial.distance.cosine(titles_TFIDF[index_source], titles_TFIDF[index_target]))
    overlap_corpus.append(len(set(source_corpus).intersection(target_corpus)))
    lsa_distance.append(sc.spatial.distance.cosine(LSA[index_source],LSA[index_target])) #cosine similarity
    lsa_distance_euc.append(np.linalg.norm(LSA[index_source]-LSA[index_target] ))
    overlap_title.append(len(set(source_title).intersection(set(target_title))))
    temp_diff.append(int(source_info[1]) - int(target_info[1]))
    comm_auth.append(len(set(source_auth).intersection(set(target_auth))))
    authorsQuotation.append(authors_quotation)
    page_rank.append(source_page_rank+target_page_rank)
    if (i % 10000 == 0):
        print(i, "examples processed")


tfidf_titles = list(np.nan_to_num(tfidf_titles))
training_features = np.array([overlap_title, temp_diff, comm_auth,tfidf_titles,authorsQuotation,page_rank,commonNeighbors, overlap_corpus,lsa_distance,page_rank]).T
# scale
training_features = preprocessing.scale(training_features)
# convert labels into integers then into column array
labels = [int(element[2]) for element in training_set_reduced]
labels = list(labels)
labels_array = np.array(labels)
#importing xgboost
import os
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-6.2.0-posix-seh-rt_v5-rev1\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb
#initializing the six classifier that we have used
classifier1=RandomForestClassifier(n_estimators=300,max_depth=7,max_features=5)
classifier2=SVC(probability=True)
classifier3=xgb.XGBClassifier(max_depth=3,n_estimators=100,learning_rate=0.1,objective='binary:logitraw',colsample_bytree=0.2)
classifier5=xgb.XGBClassifier(max_depth=10,n_estimators=100,learning_rate=0.1,colsample_bytree=0.8)
classifier6=RandomForestClassifier(n_estimators=500,max_depth=10,max_features=7)
classifier7=RandomForestClassifier(n_estimators=500,max_depth=3,max_features=4)
# training
classifier1.fit(training_features, labels_array)
classifier2.fit(training_features, labels_array)
classifier3.fit(training_features, labels_array)
classifier5.fit(training_features, labels_array)
classifier6.fit(training_features, labels_array)
classifier7.fit(training_features, labels_array)

#initializing empty lists for testing features
tfidf_titles_test = []
tfidf_distance_test = []
lsa_distance_test = []
lsa_distance_euc_test = []
authorsQuotation_test = []
page_rank_test=[]
commonNeighbors_test =[]
overlap_title_test=[]
overlap_corpus_test=[]
temp_diff_test=[]
comm_auth_test=[]
for i in range(len(testing_set)):
    source = testing_set[i][0]
    target = testing_set[i][1]
    index_source = IDs.index(source) # stores indexes in the graph g
    index_target = IDs.index(target)  # stores indexes in the graph g
    source_info = [element for element in node_info if element[0]==source][0] #find info of the node, the final [0] is used to the elemnet( list of node info ) out of another list containing it
    target_info = [element for element in node_info if element[0]==target][0]
	# convert to lowercase and tokenize
    source_title = source_info[2].lower().split(" ")
	# remove stopwords
    source_title = [token for token in source_title if token not in stpwds]
    source_title = [stemmer.stem(token) if len(token)>3 else token for token in source_title]
    target_title = target_info[2].lower().split(" ")
    target_title = [token for token in target_title if token not in stpwds]
    target_title = [stemmer.stem(token) if len(token)>3 else token for token in target_title]
    
    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")
    
    source_corpus = source_info[5].lower().split()
    source_corpus = [token for token in source_corpus if token not in stpwds]
    source_corpus = [stemmer.stem(token) for token in source_corpus]
    
    target_corpus = target_info[5].lower().split()
    target_corpus = [token for token in target_corpus if token not in stpwds]
    target_corpus = [stemmer.stem(token) if len(token)>3 else token for token in target_corpus]
    
    neigh_source = g.neighbors(index_source)
    neigh_target = g.neighbors(index_target)
    
    comm_neigh = len(set(neigh_source).intersection(set(neigh_target)))
    commonNeighbors_test.append(comm_neigh)
    
    #number of citations between authors
    authors_quotation = 0
    for s_author in source_auth:
        for t_author in target_auth:
            if s_author in quotations.keys():
                if t_author in quotations[s_author].keys():
                    authors_quotation+= quotations[s_author][t_author]                
    
    source_page_rank=pageranks[index_source]
    target_page_rank=pageranks[index_target]
    
    tfidf_titles_test.append(sc.spatial.distance.cosine(titles_TFIDF[index_source], titles_TFIDF[index_target]))
    overlap_corpus_test.append(len(set(source_corpus).intersection(target_corpus)))
    lsa_distance_test.append(sc.spatial.distance.cosine(LSA[index_source],LSA[index_target])) #cosine similarity
    lsa_distance_euc_test.append(np.linalg.norm(LSA[index_source]-LSA[index_target] ))
    overlap_title_test.append(len(set(source_title).intersection(set(target_title))))
    temp_diff_test.append(int(source_info[1]) - int(target_info[1]))
    comm_auth_test.append(len(set(source_auth).intersection(set(target_auth))))
    authorsQuotation_test.append(authors_quotation)
    page_rank_test.append(source_page_rank+target_page_rank)
    if (i % 1000 == 0):
        print(i, "examples processed")

tfidf_titles_test = list(np.nan_to_num(tfidf_titles_test))
testing_features = np.array([overlap_title_test, temp_diff_test, comm_auth_test,tfidf_titles_test,authorsQuotation_test,page_rank_test,commonNeighbors_test, overlap_corpus_test,lsa_distance_test,page_rank_test]).T
# scale
testing_features = preprocessing.scale(testing_features)
#predicting probability
proba1 = classifier1.predict_proba(testing_features)
proba2 = classifier2.predict_proba(testing_features)
proba3 = classifier3.predict_proba(testing_features)
proba5 = classifier5.predict_proba(testing_features)
proba6 = classifier6.predict_proba(testing_features)
proba7 = classifier7.predict_proba(testing_features)
#proba3 is not really a proba( due to the binary:logitraw objective function ), putting it to [0,1] range with sum 1
proba3exp=np.exp(proba3)
proba3prim=proba3exp/(proba3exp.sum(axis=1).reshape((-1,1)))
pred=np.argmax(proba3prim+proba1+proba2+proba5+proba6+proba7,axis=1)
predictions_SVM = zip(range(len(testing_set)), pred)

# creating output file
with open("testing_line.csv","wb") as pred1:
    csv_out = csv.writer(pred1,quoting=csv.QUOTE_NONNUMERIC)
    csv_out.writerow(["id","category"])
    csv_out = csv.writer(pred1)
    csv_out.writerow('"id","category"')
    for row in predictions_SVM:
        csv_out.writerow(row)


