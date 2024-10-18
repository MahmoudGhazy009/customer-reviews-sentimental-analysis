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

import logging

import azure.functions as func
from azure.storage.blob import BlobServiceClient
# from azure.storage.blob import BlockBlobService

# from azure.storage.blob import BlobServiceClient

from io import StringIO,BytesIO

import json
import pandas as pd
import time
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()  # Loads environment variables from .env file
connection_string = os.getenv('AZURE_BLOB_CONNECTION_STRING')

def main(param) -> str:
    result = cleaning(param['fileName'])
    return result
def cleaning(fileName):
    
    nltk.download('punkt')      # For sentence tokenization
    nltk.download('stopwords')  # For stopwords
    nltk.download('wordnet')
    df=read_file(fileName)
    try:
        df['text'] = df['reviews.text'].apply(clean_text)
        logging.info('45')

        df['text'] = df['text'].apply(remove_stopwords)
        logging.info('48')


        df.dropna(how='all',inplace=True)
        df.dropna(how='all',inplace=True,axis=1)
        status = saveFile(df)
        # x=df.to_dict(orient='list')
        x=status


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
    
def saveFile(df):
    csv_data = BytesIO()
    df.to_csv(csv_data, index=False)

    csv_data.seek(0)

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Initialize BlobServiceClient

    # Get the blob client (replace 'mycontainer' and 'myblob.csv' with your container and blob names)
    blob_client = blob_service_client.get_blob_client(container="filesys/Raw", blob="cleanedData.csv")
    logging.info('92')

    # Upload the in-memory CSV to Azure Blob Storage
    blob_client.upload_blob(csv_data, overwrite=True)
    logging.info('96')

    return 'cleanedData.csv'



# 1. Clean Text
def clean_text(text: str) -> str:
    """Clean the input text by:
        - Converting to lowercase
        - Removing whitespaces
        - Removing HTML tags
        - Replacing digits with spaces
        - Replacing punctuations with spaces
        - Removing extra spaces and tabs
    ------
    Input: text (str)    
    Output: cleaned text (str)
    """
    text = str(text).lower().strip()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'\d+', ' ', text)   # Replace digits with spaces
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)  # Replace punctuations with spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces and tabs
    return text

# 2. Remove Stopwords
def remove_stopwords(text: str) -> str:
    """Remove common stopwords from the text.
    ------
    Input: text (str)    
    Output: cleaned text without stopwords (str)
    """
    # logging.info('99')

    stop_words = set(stopwords.words('english'))  # Use NLTK's stopwords list
    # logging.info('100')

    # words = word_tokenize(text)
    words = text.split(' ')
    # logging.info('103')

    filtered_sentence = [w for w in words if w not in stop_words]
    return ' '.join(filtered_sentence)

# 3. Stemming
def stemm_text(text: str) -> str:
    """Apply stemming to reduce words to their root form.
    ------
    Input: text (str)    
    Output: stemmed text (str)
    """
    snow = SnowballStemmer('english')
    words = word_tokenize(text)
    stemmed_sentence = [snow.stem(w) for w in words]
    return ' '.join(stemmed_sentence)


