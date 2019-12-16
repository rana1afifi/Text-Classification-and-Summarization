import Utilities as ut
import os
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import numpy as np
################################## Global Variables ##################################
directory=os.getcwd()+"/BBC News Summary/News Articles/"
categories={
    "business":0,
    "entertainment":1,
    "politics":2,
    "sport":3,
    "tech":4 }
categories_inverse={
    0:"business",
    1:"entertainment",
    2:"politics",
    3:"sport",
    4:"tech" }
ngram_range = (1,2)
min_df = 10
max_df = 1.
max_features = 300
tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)

################################# Classification ########################################

## preprocessing of training data
def preprocess_train_data(directory):
    data =[]
    labels=[]
    for folder in os.listdir(directory):
        label=categories.get(folder,"-1")
        print("starting category ",folder)
        for file in sorted(os.listdir(directory+folder)): ##Q:remove sorted ltr
            try:
             text = open(directory+folder+"/"+file ).read()
            except(UnicodeDecodeError):
                text = open(directory + folder + "/" + file, encoding="iso-8859-1").read() ## there was a file with diff encoding
            print("parsing ",file ," " ,text[0:10])
            clean=ut.text_cleaning(text)
            data.append(clean)
            labels.append(label)
    return np.array(data),np.array(labels)


## This is the function used for training the data and validating model accuracy
def classification_pipeline():
    ## 1st preprocess data
    data,labels= preprocess_train_data(directory)
    # ut.save_file("data.txt",data)
    # ut.save_file("labels.txt",labels)
    ## 2nd Split data (here we use 20:80)
    xtrain, xtest, ytrain ,ytest= train_test_split(data,labels,test_size=0.2, random_state =8)
    # ut.save_file("xtrain.txt",xtrain)
    # ut.save_file("ytrain.txt",ytrain)
    # ut.save_file("xtest.txt",xtest)
    # ut.save_file("ytest.txt", ytest)
    # xtrain = np.array(ut.read_file("clean_data/xtrain.txt"))
    # xtest = np.array(ut.read_file("clean_data/xtest.txt"))
    # ytrain = np.array(ut.read_file("clean_data/ytrain.txt")).astype(int)
    # ytest = np.array(ut.read_file("clean_data/ytest.txt")).astype(int)

    ## 3rd Use the tfidf representation
    features_train = tfidf.fit_transform(xtrain).toarray()
    features_test = tfidf.transform(xtest).toarray()
    ut.save_obj(features_train,"features_train")
    ut.save_obj(tfidf,"tfidf")
    ut.save_obj(features_test,"features_test")

    ## 4th Use the LR Model
    model= LogisticRegression(penalty='l2',  C=1.0, class_weight='balanced', random_state=8, solver='sag', multi_class='multinomial')
    model.fit(features_train,ytrain)

    ## 5th Test on test data
    predictions= model.predict(features_test)
    print("Accuracy is ",accuracy_score(ytest, predictions))
    print(classification_report(ytest, predictions))
    ut.save_obj(model,"LRmodel")

## This is the function we test for unseen file
def classify(filepath):
    ## Open file
    try:
        text = io.open(filepath).read()
    except(UnicodeDecodeError):
        text = open(filepath,encoding="iso-8859-1").read()  ## there was a file with diff encoding
    clean = ut.text_cleaning(text)
    #tfidf_object= ut.load_obj("tfidf")              ## for some reason the object can not be loaded
    xtrain = np.array(ut.read_file("clean_data/xtrain.txt"))
    tfidf.fit_transform(xtrain)
    test = tfidf.transform([clean]).toarray()

    model_object=ut.load_obj("LRmodel")
    predictions= model_object.predict(test)
    for p in predictions:
        print("Model Predicted ",categories_inverse.get(p))



##This is a utility function that I used to most frequent bigrams and ngrams in each category
def get_frequent_ngrams(features_train,labels_train):
    for category, id in categories.items():
        features_chi2 = chi2(features_train, labels_train==id )
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        print("# '{}' category:".format(category))
        print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
        print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))
        print("")





