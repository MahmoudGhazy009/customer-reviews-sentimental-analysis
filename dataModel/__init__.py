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
from dotenv import load_dotenv
import os

load_dotenv()  # Loads environment variables from .env file
connection_string = os.getenv('AZURE_BLOB_CONNECTION_STRING')

from io import StringIO,BytesIO
import pyarrow as pa

import pyarrow.parquet as pq

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
    

    df=read_file(fileName)
    logging.info(57)
    
    prodCol=["name","asins","brand","categories","keys","manufacturer"]
    dfProduct = df[prodCol].drop_duplicates().copy()
    dfProduct['product_id']=range(1,dfProduct.shape[0]+1)
    prodName = saveFile(dfProduct,'Product.parquet')
    userCol = ['reviews.userCity','reviews.userProvince','reviews.username']
    dfUser = df[userCol].drop_duplicates().copy()
    dfUser['user_id']=range(1,dfUser.shape[0]+1)
    userName = saveFile(dfUser,'User.parquet')

    logging.info('user')
    timeDim = df[['reviews.date']].drop_duplicates()
    timeDim['date_id'] = range(1,timeDim.shape[0]+1)
    timeDim['Date'] = pd.to_datetime(timeDim['reviews.date'])

    timeDim['day'] = timeDim['Date'].dt.day
    timeDim['month'] = timeDim['Date'].dt.month
    timeDim['year'] = timeDim['Date'].dt.year
    timeName = saveFile(timeDim,'Date.parquet')

    logging.info(df.shape[0])
    df = df.merge(dfProduct)
    df = df.merge(timeDim)
    df = df.merge(dfUser)


    factCol=['product_id','user_id','date_id',"reviews.date","reviews.dateAdded","reviews.dateSeen","reviews.doRecommend","reviews.numHelpful","reviews.rating","reviews.sourceURLs"]
    factName = saveFile(df[factCol],'reviewsFact.parquet')
    
    logging.info('date')

    logging.info(59)


    return {'factName':factName,'prodName':prodName,'userName':userName,'timeName':timeName}

def read_file(file,header=0, separator=","):
    # def read_from_data_lake_csv_pandas(file, header=0, separator=","):


    # Download from blob
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    # blob_client = blob_service_client.get_blob_client(container=storage_account_name, blob="Raw")
    blob_client = blob_service_client.get_blob_client(container="filesys/Operations", blob=file)

    blob_data = blob_client.download_blob().content_as_text()
    data = StringIO(blob_data)
    xls = pd.read_csv(data)

    try:   
        return xls
    except:
        # print("Can't read the file " + file_name)
        return 'error'
    



def saveFile(df,fileName):
   
    buffer = BytesIO()

    table = pa.Table.from_pandas(df)
    pq.write_table(table, buffer)

    # Reset the buffer's cursor to the beginning
    buffer.seek(0)


    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Initialize BlobServiceClient

    # Get the blob client (replace 'mycontainer' and 'myblob.csv' with your container and blob names)
    blob_client = blob_service_client.get_blob_client(container="filesys/Operations", blob=fileName)
    logging.info('201')

    # Upload the in-memory CSV to Azure Blob Storage
    blob_client.upload_blob(buffer, overwrite=True)
    logging.info('205')

    return fileName

