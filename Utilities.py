import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import os
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import io
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize,sent_tokenize
import pickle

########### global variables ########
stop_words=set(stopwords.words('english'))
word_lemmatizer = WordNetLemmatizer()

def save_file(name , data):
    with open(name, "w+") as  f:
        for d in data:
            f.write(str(d))
            f.write("\n")
    f.close()

def read_file(filename):
    with open(filename,"r") as f:
        text=f.read()
        lines=text.split("\n")
    f.close()
    return lines


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)



def read_files(foldername):
    files=[]
    for file in os.listdir(foldername):
        files.append(file)
    return files


def map_wordnet_pos(word):
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    tag = nltk.pos_tag([word])[0][1][0].upper()
    return(word, tag_dict.get(tag, wordnet.NOUN))


def text_cleaning(file): ##is 's removed ?
## Lowercase
    file = file.lower()
##Remove endlines, possessives , special characters, stopwords
    file = file.replace("\n"," ")
    file=file.replace("'s"," ")
    # file = file.translate(str.maketrans('', '', string.punctuation))
    # file = file.translate(str.maketrans('', '', "0123456789"))
    file=re.sub('[^a-z]+', ' ',file)

##Q: Shall we remove Named Entities?
    tokens = [word for word in file.split() if word not in stop_words]
## Lemmatization
    pos_tags= [map_wordnet_pos(token) for token in tokens] ##Q: try without
    lemmmatized_tokens= [ word_lemmatizer.lemmatize(token,pos=tag) for token,tag in pos_tags]
 #   print(lemmmatized_tokens)
    return ' '.join(lemmmatized_tokens)

############### Summaization ##############################

def readFile(fileName):
    openedFile=io.open(fileName)
    text=openedFile.read()
    openedFile.close()
    return text
def segment(text):
    sentences= sent_tokenize(text)#re.split(r'[!;:\.\?]',str(text))
    return sentences,len(text)

def stemAndStopWords(text,n,st,stops):
    sentenceWordMap={}
    word_tokens = word_tokenize(text)
    filteredSentence = []
    for w in word_tokens:
        stem=st.stem(w)
        if stem not in stops:
            filteredSentence.append(stem)
            if stem in sentenceWordMap:
                sentenceWordMap[stem]+=1
            else:
                sentenceWordMap[stem] = 1
                if stem in n:
                    n[stem] += 1
                else:
                    n[stem] = 1

    return filteredSentence,sentenceWordMap


reWhitespace = re.compile(r"(\s)+")
reNumeric = re.compile(r"[0-9]+")
reTags = re.compile(r"<([^>]+)>")
rePunct = re.compile('([%s])+' % re.escape(string.punctuation))

def tokenize(txtList):
    st = PorterStemmer()
    n={}
    sentencesMaps=[]
    tokenizedSentences=[]
    avgDL=0
    i=0
    while i< len(txtList):
        sentence=str(txtList[i])
        sentence =reNumeric.sub(" ",sentence)
        sentence = reTags.sub(" ", sentence)
        sentence = rePunct.sub(" ", sentence)
        sentence = reWhitespace .sub(" ", sentence)
        #sentence=' '.join(sentence.split())
        filteredSentence, sentenceWordMap=stemAndStopWords(sentence, n, st, stop_words)
        if len(filteredSentence)<=1:
            txtList.pop(i)
            continue
        avgDL+=len(filteredSentence)
        tokenizedSentences.append(filteredSentence)
        sentencesMaps.append(sentenceWordMap)
        i+=1
    avgDL=avgDL/len(tokenizedSentences)
    return sentencesMaps,avgDL,n,tokenizedSentences

def textPreprocessing(file):
    sentences,textLen=segment(readFile(file))
    sentencesMaps, avgDL, n, tokenizedSentences=tokenize(sentences)
    return sentences,textLen,sentencesMaps, avgDL, n, tokenizedSentences