'''
Exploratory data analysis

'''


# Imports

import nltk
nltk.download('punkt') # for sent_tokenize
nltk.download('stopwords') 
nltk.download('wordnet') # for WordNetLemmatizer

import numpy as np
import pandas as pd


# Data partitioning
from sklearn.model_selection import train_test_split
# Text preprocessing/analysis
import re
from nltk import word_tokenize, sent_tokenize, FreqDist
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns


'''
Functions
'''
def tokenize2list(text):
    '''
    INPUT
    text - a string of text

    OUTPUT
    text - a string of text that has been processed with the steps below
        1) Normalize
        2) Remove punctuation
        3) Lemmatize
    '''

    # lower case and remove punctuation
    text = re.sub(r'[^a-zA-Z0-9]'," ",text.lower())

    # tokenize text to words
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens]

    tokens= [ w for w in tokens if w not in stopwords.words("english")]

    token_list=' '.join(tokens)

    return token_list


def tokenize2words(text):
    '''
    INPUT
    text - a string of text

    OUTPUT
    text - a string of text that has been processed with the steps below
        1) Normalize
        2) Remove punctuation
        3) Lemmatize
    '''

    # lower case and remove punctuation
    text = re.sub(r'[^a-zA-Z0-9]'," ",text.lower())

    # tokenize text to words
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens]

    tokens= [ w for w in tokens if w not in stopwords.words("english")]

    return tokens




# Data load 
df=pd.read_csv('./data/emotion_corpus.txt',sep=';')


# How many instances of each target category are there? 
palette_dict={
    'joy':'#FFF633',
    'anger':'#F13027',
    'sadness':'#279BF1',
    'fear':'#9827F1',
    'love':'#F127A1',
    'surprise':'#27F143'
    }


sns.countplot(x=df['target'],order=df['target'].value_counts().index,
              palette=palette_dict)




'''
How many total and unique words are there?
What are the most common words?
What is the term frequency for all words? 
'''




# Combine all records into one document and make everything lower case
corpus = " ".join(df['document'].values).lower()


# Split by whitespace
word_list = corpus.split() 

print(f"Number of strings: {len(word_list)}")

print(f"Number of unique strings: {len(set(word_list))}")

fd = FreqDist(word_list)

word_counts=pd.DataFrame(fd.most_common(),columns=['word','word_count'])
word_counts['TF']=word_counts['word_count']/word_counts['word_count'].sum()

word_counts.sort_values(by='TF',ascending=False,inplace=True)

word_counts['rank']=word_counts['TF'].rank(method='first',ascending=False)

word_counts['quantile']=pd.qcut(word_counts['rank'],100,labels=np.arange(1,101))

#Some of the most common words are stop words, so will be useful to remove stopwords
#and then perform this again,  though good to keep I and im and me and you 










# Setting up train and test sets 
X=df['document'].copy()
Y=df['target'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=123,
                                                    stratify=Y)

# Append sentiment back using indices
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)


# Inspect the first five records of the datasets
train.head()

test.head()






