'''
To dos:
    -ensure functions have docstrings
    -ensure preprocessing steps BEFORE pipeline are all within a function/class


    - maybe only use more frequent keywords as features? 
    test=X_train_transformed[X_train_transformed>0].count().reset_index()
'''


'''
Imports
'''
import numpy as np
import pandas as pd
import re
import pickle


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn import svm
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import FunctionTransformer


from sklearn.model_selection import train_test_split

'''
Functions
'''

class features():
    
    
    def __init__(self):
        pass
    
    
    def count_tokens(self,text_col):
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
    


ftGen=features()





# Data load 
df=pd.read_csv('./data/emotion_corpus.txt',sep=';')



# Setting up train and test sets 
X=df['document'].copy()
Y=df['target'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=123,
                                                    stratify=Y)

X_train=pd.DataFrame(X_train.head(10)).reset_index(drop=True)

y_train=y_train.head(10).reset_index(drop=True).to_list()

X_test=pd.DataFrame(X_test.head(10)).reset_index(drop=True)

y_test=y_test.head(10).reset_index(drop=True).to_list()





numeric_cols=['n_tokens','n_i']

str_cols='document'

features_created=['n_characters']


X_train['n_tokens']=X_train['document'].apply(ftGen.count_tokens)
X_train['n_i']=X_train['document'].str.count('(\si\s)|(^i\s)')


# Setting up the NLP and ML pipelines

# NLP /text preprocessing
vectoriser = TfidfVectorizer(token_pattern=r'[a-z]+', stop_words='english')

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
    ('clf',svm.SVC(kernel='linear'))
    ])


pipeML.fit(X_train,y_train)


#Xtest now has to have the right amount of columns
X_test['n_tokens']=X_test['document'].apply(ftGen.count_tokens)
X_test['n_i']=X_test['document'].str.count('(\si\s)|(^i\s)')


pipeML.predict(X_test)



terms = preprocessor_str_num.named_transformers_['text'].transformer_list[0][1].get_feature_names()
columns = terms + features_created + numeric_cols
X_train_transformed = pd.DataFrame(preprocessor_str_num.transform(X_train).toarray(), 
                                   columns=columns)

   
    
    

# Pickling and unpickling model object
with open('pipeML.pkl','wb') as f:
    pickle.dump(pipeML,f)

handler = open('pipeML.pkl', "rb")

pipeML_pkl = pickle.load(handler)

pipeML_pkl.predict(X_test)

