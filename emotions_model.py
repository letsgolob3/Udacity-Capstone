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


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


from sklearn import svm
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import FunctionTransformer


from sklearn.model_selection import train_test_split

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
    

def load_data(path):
    '''
    INPUT
    path: Relative path to data file

    OUTPUT
    df: Emotion data 
    '''   
    
    # Data load 
    df=pd.read_csv(path,sep=';')
    
    print('loaded')
    
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
    
    # X_train=pd.DataFrame(X_train.head(10)).reset_index(drop=True)
    
    # y_train=y_train.head(10).reset_index(drop=True).to_list()
    
    # X_test=pd.DataFrame(X_test.head(10)).reset_index(drop=True)
    
    # y_test=y_test.head(10).reset_index(drop=True).to_list()
        
    
    return X_train, X_test, y_train, y_test


def apply_features(X):
    
    '''
    INPUT
    X: features data
    
    OUTPUT
    X: features data with new features
    '''       
    
    X['n_tokens'] = X['document'].apply(count_tokens)
    X['n_i']=X['document'].str.count('(\si\s)|(^i\s)|(\sI\s)|(^I\s)')    
    
    return X

def set_pipelines(X_train,y_train):
    '''
    INPUT
    X_train: Training features 
    y_train: Training labels
    
    OUTPUT
    cv: Cross-validated model object
    '''       
    
    
    numeric_cols=['n_tokens','n_i']

    str_cols='document'
    
    features_created=['n_characters']
    
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
    
    
    # Parameters for chosen classifier
    parameters = {
        'clf__gamma': [0.1, 1.0,10],
        'clf__kernel': ['linear','poly','rbf'],
        'clf__C': [.1,1,10,100],
        }

    pipeML.fit(X_train,y_train)
    
    cv=RandomizedSearchCV(pipeML,param_distributions=parameters,cv=2,verbose=2)


    cv.fit(X_train,y_train)

    print(cv.best_params_)
    
    print(cv.best_score_)
    
    try:
        terms = preprocessor_str_num.named_transformers_['text'].transformer_list[0][1].get_feature_names()
        columns = terms + features_created + numeric_cols
        X_train_transformed = pd.DataFrame(preprocessor_str_num.transform(X_train).toarray(), 
                                            columns=columns)
        print('transform worked')
    except:
        print('x_train transform did not work')
        pass

    return cv
  
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
    
    # predictions=pipeML.predict(X_test)
    
    # Recall, precision, accuracy, and f1 score for all categories
    print(classification_report(y_test, predictions))
    
    # accuracy_score(y_test, predictions)
    
    cm = confusion_matrix(y_test, predictions)
    
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    #The diagonal entries are the accuracies of each class
    print(cm.diagonal())

def save_model(model):
    '''
    INPUT
    model: trained model

    OUTPUT
    cv: Cross-validated model object
    '''  
    # Pickling and unpickling model object
    with open('{model}.pkl','wb') as f:
        pickle.dump(model,f)
    
    # handler = open('hcv.pkl', "rb")
    
    # pipeML_pkl = pickle.load(handler)
    
    # pipeML_pkl.predict(X_test)
    
    
    pass


def main():
    
    df=load_data('./data/emotion_corpus.txt')
    
    X_train, X_test, y_train, y_test=set_train_test_sets(df)
    
    X_train = apply_features(X_train)

    X_test = apply_features(X_test) 
    
    fitted_model=set_pipelines(X_train,y_train)
    
    get_performance(fitted_model,X_test,y_test)

    save_model(fitted_model)

if __name__ == '__main__':
    main()


