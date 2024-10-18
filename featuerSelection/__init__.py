# This function is not intended to be invoked directly. Instead it will be
# triggered by an orchestrator function.
# Before running this sample, please:
# - create a Durable orchestration function
# - create a Durable HTTP starter function
# - add azure-functions-durable to requirements.txt
# - run pip install -r requirements.txt

import re
import string
import nltk
# nltk.data.path.append('./nltk_data')
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import joblib

import logging

import azure.functions as func
from azure.storage.blob import BlobServiceClient
# from azure.storage.blob import BlockBlobService

# from azure.storage.blob import BlobServiceClient

from io import StringIO,BytesIO

from dotenv import load_dotenv
import os

load_dotenv()  # Loads environment variables from .env file
connection_string = os.getenv('AZURE_BLOB_CONNECTION_STRING')

import json
import pandas as pd
import time
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB



def main(param) -> str:
    result = cleaning(param['fileName'])
    return result
def cleaning(fileName):
    
    # nltk.download('punkt')      # For sentence tokenization
    # nltk.download('stopwords')  # For stopwords
    # nltk.download('wordnet')
    df=read_file(fileName)
    sentiment_score = {1: 0,
                   2: 0,
                   3: 0,
                   4: 1,
                   5: 1}

    sentiment = {0: 'NEGATIVE',
             1: 'POSITIVE'}


# mapping
    df['sentiment_score'] = df['reviews.rating'].map(sentiment_score)
    df['sentiment'] = df['sentiment_score'].map(sentiment)
    logging.info(45)
    df.dropna(how='all',inplace=True)
    df.dropna(how='all',inplace=True,axis=1)


    try:
        # X_train,y_train,X_test,y_test = selection(df)
        vectorizer = TfidfVectorizer(max_features=700)
        vectorizer.fit(df['text'])

        features = vectorizer.transform(df['text'])
        col = vectorizer.get_feature_names_out()
        arr=features.toarray()

        tf_idf = pd.DataFrame(arr, columns=col)


        # out = modeling(MultinomialNB(), Xtrain = X_train, Xtest = X_test,Y_train=y_train,Y_test=y_test)
        # loaded_model = joblib.load(out)
        logging.info(78)
        # new_predictions = loaded_model.predict(X_test)
        file = readPickel('cutomerRevirewSentimentalByNBM.pkl')
        loaded_model = joblib.load(file)
        logging.info(82)
        f=int(df.shape[1])
        logging.info(84)
        logging.info(f)

        df['predictedValue']= loaded_model.predict(tf_idf)
        logging.info(85)
        logging.info(df.shape[1])
        df=df.drop(columns=['Unnamed: 21','Unnamed: 23'])
        name = saveFile(df)
        x = name

    except:
        x='error'
    return x

def read_file(file,header=0, separator=","):
    # def read_from_data_lake_csv_pandas(file, header=0, separator=","):


    # Download from blob
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    # blob_client = blob_service_client.get_blob_client(container=storage_account_name, blob="Raw")
    blob_client = blob_service_client.get_blob_client(container="filesys/Raw", blob=file)

    blob_data = blob_client.download_blob().content_as_text()
    data = StringIO(blob_data)
    xls = pd.read_csv(data)

    try:   
        return xls
    except:
        # print("Can't read the file " + file_name)
        return 'error'
    


def readPickel(fileName):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    blob_client = blob_service_client.get_blob_client(container="filesys/Raw", blob=fileName)

    blob_data = blob_client.download_blob().readall()

    # Convert the bytes data into a file-like object
    pickle_data = BytesIO(blob_data)

    # Load the model using joblib
    return pickle_data


def selection(data):


    vectorizer = TfidfVectorizer(max_features=700)
    logging.info(80)
    vectorizer.fit(data['text'])
    logging.info(81)

    features = vectorizer.transform(data['text'])
    logging.info(85)
    col = vectorizer.get_feature_names_out()
    logging.info(87)
    arr=features.toarray()
    logging.info(89)

    tf_idf = pd.DataFrame(arr, columns=col)
    # tf_idf = pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names())
    logging.info(88)

    X_train, X_test, y_train, y_test = train_test_split(tf_idf, data['sentiment_score'], test_size=1, random_state=42)
    print(type(X_test))
    print(X_test.shape)
    logging.info(91)

    yy = pd.DataFrame(y_train)
    logging.info(94)

    train_data = pd.concat([X_train, yy],axis=1)
    logging.info(97)

    negative_class = train_data[train_data['sentiment_score'] == 0]
    positive_class = train_data[train_data['sentiment_score'] == 1]
    target_count = train_data['sentiment_score'].value_counts()
    negative_over = negative_class.sample(target_count[1], replace=True)
    logging.info(103)

    df_train_over = pd.concat([positive_class, negative_over], axis=0)
    df_train_over = shuffle(df_train_over)
    df_train_over.dropna(how='all',inplace=True)

    X_train = df_train_over.iloc[:,:-1]
    y_train = df_train_over['sentiment_score']

    return X_train,y_train,X_test,y_test


def saveFile(df):
    csv_data = BytesIO()
    df.to_csv(csv_data, index=False)

    csv_data.seek(0)

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Initialize BlobServiceClient

    # Get the blob client (replace 'mycontainer' and 'myblob.csv' with your container and blob names)
    blob_client = blob_service_client.get_blob_client(container="filesys/Operations", blob="modeledData.csv")
    logging.info('201')

    # Upload the in-memory CSV to Azure Blob Storage
    blob_client.upload_blob(csv_data, overwrite=True)
    logging.info('205')

    return 'modeledData.csv'


# def modeling(Model, Xtrain, Xtest ,Y_train,Y_test,model_name='cutomerRevirewSentimentalByNBM.pkl'):
#     """
#     This function apply countVectorizer with machine learning algorithms. 
#     """
    
#     # Instantiate the classifier: model
#     model = Model
    
#     # Fitting classifier to the Training set (all features)
#     model.fit(Xtrain, Y_train)
    
#     global y_pred
#     # Predicting the Test set results
#     y_pred = model.predict(Xtest)
#     confusion_matrix = pd.crosstab(index=Y_test, columns=np.round(y_pred), rownames=['Actual'], colnames=['Predictions']).astype(int)
#     joblib.dump(model, model_name)

#     return confusion_matrix



# updare modeling function to be
# def modeling(Model, Xtrain, Xtest, model_name):
#     """
#     This function applies CountVectorizer with machine learning algorithms 
#     and saves the trained model.
#     """
    
#     # Instantiate the classifier: model
#     model = Model
    
#     # Fitting classifier to the Training set (all features)
#     model.fit(Xtrain, y_train)
    
#     global y_pred
#     # Predicting the Test set results
#     y_pred = model.predict(Xtest)
    
#     # Assign f1 score to a variable
 
#     # Confusion matrix
#     confusion_matrix = pd.crosstab(index=y_test, columns=np.round(y_pred), rownames=['Actual'], colnames=['Predictions']).astype(int)
    
#     # Save the trained model
#     joblib.dump(model, model_name)
#     print(f'Model saved to {model_name}')

# # Calling modeling function
# modeling(MultinomialNB(), X_train, X_test, )


# Now you can use the loaded model for predictions
# new_predictions = loaded_model.predict(X_test)