'''
Exploratory data analysis

Adhering to CRISP-DM principles, it is important to obtain a better 
understanding of the data before preprocessing and modeling.  

'''


# Imports

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('punkt') # for sent_tokenize
nltk.download('stopwords') 
nltk.download('wordnet') # for WordNetLemmatizer

from nltk import word_tokenize, sent_tokenize, FreqDist
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from textblob import TextBlob

from sklearn.model_selection import train_test_split



'''
Functions
'''



def tokenize(text):
    '''
    INPUT
    text - a string of text

    OUTPUT
    text - a string of text that has been processed with the steps below
        1) Normalize
        2) Remove punctuation
        3) Lemmatize
        4) Stop words removed 
    '''

    # lower case and remove punctuation
    text = re.sub(r'[^a-zA-Z0-9]'," ",text.lower())

    # tokenize text to words
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens]
    
    stop_wds=set(stopwords.words("english"))
    
    emotion_relevant_stop_words={'i', 'yours', 'you', 'me', 'against',
                                 'down','myself','very','my'}
    
    # Remove emotion relevant stop words from overall stop words to keep in data
    stop_wds_modified=stop_wds.difference(emotion_relevant_stop_words)
    
    tokens= [w for w in tokens if w not in stop_wds_modified]

    token_str=' '.join(tokens)

    return token_str

def generateSentiment(text):
    '''
    INPUT
    text - a string of text

    OUTPUT
    sentiment - a float between (negative) -1 and 1 (positive) indicating sentimenet
    subjectivity - a float between (objective) 0 and 1 (subjective) indicating subjectivity 

    '''
    
    tb_phrase = TextBlob(text)
    
    sentiment=tb_phrase.sentiment
    
    polarity=sentiment.polarity
    
    subjectivity=sentiment.subjectivity
    
    return polarity,subjectivity

    
# Data load 
df=pd.read_csv('./data/emotion_corpus.txt',sep=';')

df.info()


'''
Section 1:
    
    - 1) How many instances of each target category are there? 
    - 2) If we group some of the target categories together into positive/negative/neutral
    groups, is there a trend?
    - 3) Using TextBlob, what percent of the data is positive/negative/neutral?
    - 4) How does the estimated sentiment from #2 compare to the score from 3? 
'''

# 1.1; Emotion counts
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
plt.title('Counts of emotions')
plt.xlabel('\nEmotion')
plt.show()


print('Percent of emotions within dataset:\n')
print(round(df['target'].value_counts(normalize=True),2)*100)

# 1.2; Estimate sentiment based on emotion groupings
df['est_sentiment']='neutral'
df.loc[df['target'].isin(['joy','love']),'est_sentiment']='positive'
df.loc[df['target'].isin(['anger','fear','sadness']),'est_sentiment']='negative'

est_sent_dict={
    'neutral':'#ACACAC',
    'negative':'#F13027',
    'positive':'#27F143'
    }

sns.countplot(x=df['est_sentiment'],order=df['est_sentiment'].value_counts().index,
              palette=est_sent_dict)
plt.title('Grouping emotions into sentiment categories')
plt.xlabel('\nCategory')
plt.show()


print('Percent of estimated sentiment within dataset:\n')
print(round(df['est_sentiment'].value_counts(normalize=True),2)*100)

#1.3; Determine the sentiment based on the TextBlob library
#   Convert the list of numbers to separate columns
df[['sentiment',
    'sentiment_subjectivity']]=pd.DataFrame(df['document'].apply(generateSentiment).tolist(),
                                            index=df.index)

print(pd.pivot_table(df,index='target',
                     values=['sentiment','sentiment_subjectivity']))

sentiment_agg=pd.pivot_table(df,index='target',
                     values=['sentiment','sentiment_subjectivity']).reset_index()


ax=sentiment_agg.plot.bar(x='target',y='sentiment',color=['#F13027','#F13027',
                                                       '#27F143','#27F143',
                                                     '#F13027','#D2F50F'])
ax.axhline(linewidth=2,color='black')
ax.axhline(y=0)
plt.title('TextBlob Sentiment Score per Emotion')
plt.xlabel('Emotion')
plt.ylabel('Sentiment')
plt.show()

#1.4; Estimated sentiment to TextBlob sentiment
print(pd.pivot_table(df,index='est_sentiment',
                     values=['sentiment','sentiment_subjectivity']))


'''
Section 1 results:
    
- From 1.1
    - Joy and sadness comprise the majority of the dataset
- From 1.2
    - Assuming joy/love are positive, anger/fear/sadness are
negative, and suprise is neutral the data set is 54% negative, 42% positive,and 4% neutral
- From 1.3
    - The average sentiment for anger, fear, and sadness is less than 0
while the average sentiment for joy and love is greater than 0.  
    - The average sentiment for surprise is positive but much closer to 0 since 
that could be positive or negative 
- The subjectivity for surprise is higher than all the other emotion
- From 1.4
    - The negative estimated sentiment matches the TextBlob sentiment since average
    negative estimated sentiment is less than 0, average neutral sentiment is close
    to 0, and average positive sentiment is greater than 0
'''




'''
Section 2a: Word count and common words questions:
    - 1) How many total and unique words are there?
    - 2) What are the most common words?
    - 3) What is the term frequency for all words? 
'''


# Combine all records into one document and make everything lower case
corpus = " ".join(df['document'].values).lower()

# Split by whitespace
word_list = corpus.split() 

print(f"Number of words: {len(word_list)}")

print(f"Number of unique words: {len(set(word_list))}")

fd = FreqDist(word_list)

word_counts=pd.DataFrame(fd.most_common(),columns=['word','word_count'])
word_counts['TF']=word_counts['word_count']/word_counts['word_count'].sum()

word_counts.sort_values(by='TF',ascending=False,inplace=True)

word_counts['rank']=word_counts['TF'].rank(method='first',ascending=False)

word_counts['quantile']=pd.qcut(word_counts['rank'],100,labels=np.arange(1,101))

print(word_counts.loc[word_counts['quantile']==1,'word'].head(20))

'''
Section 2a results:

- There are over 38,000 and 17,000 total and unique words, respectively.
- Many of the most frequent words are stop words, so let's repeat the process 
after removing stop words and performing lemmatization. 


'''




'''
Section 2b:
Word count and common words questions after stop word removal and lemmatization:
    - 1) Are there more stop words in one category than another? 
    - 2) Are there more words in one category than another? 
    - 3) How many total and unique words are there?
    - 4) What are the most common words?
    - 5) What is the term frequency for all words? 

'''

#1-2)
df['n_words']=df['document'].str.split('\s').apply(len)
df['document']=df['document'].apply(tokenize)
df['n_words_stop_rm']=df['document'].str.split('\s').apply(len)
df['n_stop_wds']=df['n_words']-df['n_words_stop_rm']

df.drop(columns=['n_words_stop_rm'],inplace=True)

# N_words distribution based on target
sns.histplot(x=df['n_words'],hue=df['target'],multiple='stack')
plt.title('Distribution of number of words')
plt.xlabel('Number of words')
plt.show()

# Descriptiptive statistics; total and based on target
df.describe()

print(pd.pivot_table(df,index='target',values=['n_words','n_stop_wds']))


#3-5)
# Combine all records into one document and make everything lower case
corpus = " ".join(df['document'].values).lower()


# Split by whitespace
word_list = corpus.split() 

print(f"Number of words: {len(word_list)}")

print(f"Number of unique words: {len(set(word_list))}")

fd = FreqDist(word_list)

word_counts=pd.DataFrame(fd.most_common(),columns=['word','word_count'])
word_counts['TF']=word_counts['word_count']/word_counts['word_count'].sum()

word_counts.sort_values(by='TF',ascending=False,inplace=True)

word_counts['rank']=word_counts['TF'].rank(method='first',ascending=False)

word_counts['quantile']=pd.qcut(word_counts['rank'],100,labels=np.arange(1,101))

print(word_counts.loc[word_counts['quantile']==1,'word'].head(20))


'''
Section 2b results:

    - The average number of stop words is least in sadness (7) and most in love (8),
    but the spread is low
    - The average number of words between all categories rangse between 18-20
    - Distribution of words in all categories is similar
    - There are over 23,000 and 15,000 total and unique words, respectively.
    - Some of the most common words are feel/feeling, very, myself
    

'''







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






