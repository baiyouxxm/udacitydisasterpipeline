import sys
# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])
# import libraries
import joblib
from sqlalchemy import create_engine
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection  import GridSearchCV
from sklearn.metrics import classification_report


def load_data(database_filepath):
       """
       Loads X and y and gets category names
    Args:
        database_filepath (str): string filepath of the sqlite database
    Return:
        X (pandas dataframe): Feature 
        y (pandas dataframe): Classification labels
        category_names (list): List of the category names for classification
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    y = df.drop(['id','message','original','genre'],axis=1)
    category_names=y.columns.values
    return X,y, category_names


def tokenize(text):
    """
    Tokenize the text
    Args:  source string
    Return:
    clean_tokens(str list): clean string list
    """
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens



def build_model():
    """
    Build a CV pipeline to choose the best model
    Args:
    N.a.
    Return:
    A grid search model
    """
    pipeline =  Pipeline([    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf' , TfidfTransformer()),
    ('clf' , MultiOutputClassifier(RandomForestClassifier()))
     ])
    
    parameters = {
        'clf__estimator__min_samples_leaf': [1,10],
        'clf__estimator__max_features': ['auto','log2']
    }

    cv = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1)
    
    return cv
    

def evaluate_model(model, X_test, Y_test, category_names):
    """
    To get the model performance
    Args:
    Model, X test set, Y test set, the category models
    Return:
    Performance measures
    """
    y_pred = model.predict(X_test)
    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], y_pred[:, i]))


def save_model(model, model_filepath):
        """dumps the model to the given filepath
    Args:
        model (scikit-learn model): The fitted model
        model_filepath (string): the filepath to save the model
    Returns:
        None
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()