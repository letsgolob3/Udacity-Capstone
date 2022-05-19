'''
To dos:
    -ensure functions have docstrings
    -ensure preprocessing steps BEFORE pipeline are all within a function/class


    - maybe only use more frequent keywords as features? 
    test=X_train_transformed[X_train_transformed>0].count().reset_index()
    
    #https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
    
    
'''


'''
Imports
'''
import numpy as np
import pandas as pd
import re
import pickle

from sklearn import svm

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer

from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from textblob import TextBlob

import matplotlib.pyplot as plt



'''
Functions
'''

    
def count_tokens(text_col):
    '''
    INPUT
    text_col: string of text

    OUTPUT
    n_tokens: integer of number of tokens
    '''        
    # lower case and remove punctuation
    text_col = re.sub(r'[^a-zA-Z0-9]'," ",text_col.lower())

    # tokenize text to words
    tokens = word_tokenize(text_col)
    
    n_tokens=len(tokens)
    
    return n_tokens


def count_n_char(Series_col):
    '''
    INPUT
    Series_col: Column of text

    OUTPUT
    n_char_col: Column containing length of each row in the input
    '''       
    
    n_char_col=Series_col.str.len()
    
    n_char_col=np.array(n_char_col).reshape(-1,1)
    
    return n_char_col

def get_all_words(df,emotion):

    '''
    INPUT
    df: input data
    emotion: string to subset df
    quantile: integer regarding which percentile to take words from

    OUTPUT
    top_words: set containing words in top 5 percentile of records
    '''   
    
    # Combine all records into one document and make everything lower case by emotion
    corpus_sub = " ".join(df.loc[df['target']==emotion,'document'].values).lower()


    # Split by whitespace
    word_list_sub = corpus_sub.split()    
    
    return word_list_sub



def get_top_pctile_words(df,emotion,quantile):
    '''
    INPUT
    df: input data
    emotion: string to subset df
    quantile: integer regarding which percentile to take words from

    OUTPUT
    top_words: set containing words in top 5 percentile of records
    '''    

    word_list_sub=get_all_words(df,emotion)
    
    print(f'{emotion}')
    print(f"Number of words: {len(word_list_sub)}")
    
    print(f"Number of unique words: {len(set(word_list_sub))}")
    
    fd_sub = FreqDist(word_list_sub)
    
    word_counts_sub=pd.DataFrame(fd_sub.most_common(),columns=['word','word_count'])
    word_counts_sub['TF']=word_counts_sub['word_count']/word_counts_sub['word_count'].sum()
    
    word_counts_sub.sort_values(by='TF',ascending=False,inplace=True)
    
    word_counts_sub['rank']=word_counts_sub['TF'].rank(method='first',ascending=False)
    
    word_counts_sub['quantile']=pd.qcut(word_counts_sub['rank'],100,labels=np.arange(1,101))
    
    top_words=set(word_counts_sub.loc[word_counts_sub['quantile']<=quantile,'word'].to_list())

    return top_words

def get_top_words_per_emotion_idx(df,target_list):
    '''
    INPUT
    df: input data
    target: list of targets

    OUTPUT
    df:input data with indicator columns for top key words in each target
    top_words_per_target: dictionary with top keywords for each target
    '''   
    top_words={}
    top_words={emotion:get_top_pctile_words(df,emotion,5) for emotion in target_list}

    all_words={emotion:set(get_all_words(df,emotion)) for emotion in target_list}


    top_words_per_target={}



    for emotionA,word_setA in top_words.items():
        
        A_not_B=[]
        for emotionB,word_setB in all_words.items():
            
            
            if emotionA!=emotionB:
                A_not_B.extend(list(word_setA.difference(word_setB)))
        
        A_not_B=list(set(A_not_B))
        
        top_words_per_target[emotionA]=A_not_B

    # Adding in the indicators as possible features
    for emotion in target_list:
        df[f'has_top_{emotion}_word']=df['document'].str.contains('|'.join(top_words_per_target[emotion]))
    
    
    [df[f'has_top_{emotion}_word'].replace({True:1,False:0},inplace=True) \
         for emotion in target_list]
        
    return df,top_words_per_target

def load_data(path):
    '''
    INPUT
    path: Relative path to data file

    OUTPUT
    df: Emotion data 
    '''   
    
    # Data load 
    df=pd.read_csv(path,sep=';')
    
    print('Loaded data')
    
    return df

def set_train_test_sets(df):
    '''
    INPUT
    df: data file
    
    OUTPUT
    X_train: Training feature data 
    X_test:  Test feature data
    y_train: Training label data
    y_test: Test label data
    '''       
    
    
    # Setting up train and test sets 
    X=df['document'].copy()
    Y=df['target'].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                        random_state=123,
                                                        stratify=Y)
        
    X_train=pd.DataFrame(X_train).reset_index(drop=True)
    
    y_train=y_train.reset_index(drop=True).to_list()
    
    X_test=pd.DataFrame(X_test).reset_index(drop=True)
    
    y_test=y_test.reset_index(drop=True).to_list()         
    
    return X_train, X_test, y_train, y_test

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

def apply_features(X):
    
    '''
    INPUT
    X: features data
    
    OUTPUT
    X: features data with new features
    '''       
    
    print('Generating features')
    
    X['n_tokens'] = X['document'].apply(count_tokens)
    X['n_i']=X['document'].str.count('(\si\s)|(^i\s)|(\sI\s)|(^I\s)')  
    X[['sentiment','sentiment_subjectivity']]=pd.DataFrame(X['document'].apply(generateSentiment).tolist(),
                                                           index=X.index)
    
    # Only want sentiment, not the subjectivity
    X.drop(columns=['sentiment_subjectivity'],inplace=True)
    
    return X


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

    return tokens

def set_pipelines(X_train,y_train):
    '''
    INPUT
    X_train: Training features 
    y_train: Training labels
    
    OUTPUT
    cv: Cross-validated model object
    '''       
    
    
    numeric_cols=['n_tokens','n_i','sentiment']

    str_cols='document'
    
    features_created=['n_characters']
    
    # Setting up the NLP and ML pipelines
    
    # NLP /text preprocessing
#    vectoriser = TfidfVectorizer(token_pattern=r'[a-z]+', stop_words='english')
    vectoriser = TfidfVectorizer(tokenizer=tokenize)    
    
    character_pipe = Pipeline([
        ('character_counter', FunctionTransformer(count_n_char)),
        ('scaler', MinMaxScaler())
        ])
    
    preprocessor = FeatureUnion([
        ('vectoriser', vectoriser),
        ('character', character_pipe)
    ])
    
    # Numeric feature preprocessing
    num_pipe = Pipeline([
        ('Imputer', SimpleImputer()),
        ('scaler', MinMaxScaler())
        ])
    
    
    # Text and numeric feature preprocessing combined
    preprocessor_str_num=ColumnTransformer([
        ('text',preprocessor,str_cols),
        ('num',num_pipe,numeric_cols)
        ])
    
    # Machine learning pipeline
    pipeML=Pipeline([
        ('preprocessor',preprocessor_str_num),
        ('clf',svm.SVC(kernel='linear',C=1,gamma=0.1))
        ])
    
    
    # Parameters for chosen classifier
    # parameters = {
    #     'clf__gamma': [0.1, 1.0,10],
    #     'clf__kernel': ['linear','poly','rbf'],
    #     'clf__C': [.1,1,10,100],
    #     }

    print('Fitting model')
    pipeML.fit(X_train,y_train)
    
    
    
    # cv=RandomizedSearchCV(pipeML,param_distributions=parameters,cv=2,verbose=2)


    # cv.fit(X_train,y_train)

    # print(cv.best_params_)
    
    # print(cv.best_score_)
    
    try:
        terms = preprocessor_str_num.named_transformers_['text'].transformer_list[0][1].get_feature_names()
        columns = terms + features_created + numeric_cols
        X_train_transformed = pd.DataFrame(preprocessor_str_num.transform(X_train).toarray(), 
                                            columns=columns)
        print('transform worked')
    except:
        print('x_train transform did not work')
        pass

    return pipeML
  
def get_performance(model,X_test,y_test):
    '''
    INPUT
    model: trained model
    X_test: Test features
    y_test: Test labels

    OUTPUT
    cv: Cross-validated model object
    '''   
    # Prediction on test set and performance metrics 
    predictions=model.predict(X_test)
    
    # Recall, precision, accuracy, and f1 score for all categories
    print(classification_report(y_test, predictions))
    
    # accuracy_score(y_test, predictions)
    
    cm = confusion_matrix(y_test, predictions)
    
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    #The diagonal entries are the accuracies of each class
    
    cmd=[round(val,3) for val in cm.diagonal()]
    
    print(cmd)

def save_model(model):
    '''
    INPUT
    model: trained model

    OUTPUT
    cv: Cross-validated model object
    '''  
    # Pickling and unpickling model object
    with open('model.pkl','wb') as f:
        pickle.dump(model,f)
    
    pass


def main():
    '''
    INPUT
    None
    
    OUTPUT
    Serialized Model Object
    '''  
    
    # Load data
    df=load_data('./data/emotion_corpus.txt')
    
    # Indicator features for top keywords associated with each emotion
    df,_=get_top_words_per_emotion_idx(df,['surprise','love','anger','sadness',
                                          'fear','joy'])
    
    # Set train and test sets
    X_train, X_test, y_train, y_test=set_train_test_sets(df)
    
    # Add additional features
    X_train = apply_features(X_train)

    X_test = apply_features(X_test) 
    
    
    fitted_model=set_pipelines(X_train,y_train)
    
    get_performance(fitted_model,X_test,y_test)

    save_model(fitted_model)

if __name__ == '__main__':
    main()


