import pandas as pd
import numpy as np
import re
import nltk
import spacy
from string import punctuation
import extract_msg
nltk.download('punkt')
from nltk.tokenize import word_tokenize
# NLTK stopwords modules
nltk.download('stopwords')
from nltk.corpus import stopwords
# NLTK lemmatization modules
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
import io
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import json
import ast

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny , IsAuthenticated
from rest_framework.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_200_OK
)
from rest_framework.response import Response

from django.views.decorators.csrf import csrf_exempt

###########    GLOBALS   ##################################

tfidf = None
category_filename = "category.pkl"
test_filelist_filename = "test_filenames.pkl"
tfidf_file = 'tfidf_file.pkl'


###############################  FUNCTIONS FOR PREPROCESSING DATA #############################################


def remove_emails_urls(dataframe):
    no_emails = re.sub(r'\S*@\S*\s?','',str(dataframe))
    no_url = re.sub(r"http\S+",'',no_emails)
    return no_url

def remove_dates(dataframe):
    # DD/MM/YYYY or MM/DD/YYYY or DD|MM.MM|DD.YYYY format
    dataframe = re.sub(r'(\b(0?[1-9]|[12]\d|30|31)[^\w\d\r\n:](0?[1-9]|1[0-2])[^\w\d\r\n:](\d{4}|\d{2})\b)|(\b(0?[1-9]|1[0-2])[^\w\d\r\n:](0?[1-9]|[12]\d|30|31)[^\w\d\r\n:](\d{4}|\d{2})\b)','',dataframe)
    # October 21, 2014 format
    dataframe = re.sub(r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|(nov|dec)(?:ember)?)(?=\D|$)','',dataframe)
    # mon|monday format
    dataframe = re.sub(r'\b((mon|tues|wed(nes)?|thur(s)?|fri|sat(ur)?|sun)(day)?)\b','',dataframe)
    return dataframe

def remove_useless(dataframe):
    #for body removal words
    dataframe = re.sub('from:','',dataframe)
    dataframe = re.sub('sent:','',dataframe)
    dataframe = re.sub('to:','',dataframe)
    dataframe = re.sub('cc:','',dataframe)
    dataframe = re.sub('bcc:','',dataframe)
    dataframe = re.sub('subject:','',dataframe)
    dataframe = re.sub('message encrypted','',dataframe)
    dataframe = re.sub('warning:','',dataframe)
    #for subject removal words
    dataframe = re.sub('fw:','',dataframe)
    dataframe = re.sub('re:','',dataframe)
    return dataframe

def remove_punctuation(text):
    #function to remove the punctuation
    return re.sub('[^\w\s]','',text)

def remove_no(text):
    return re.sub(r"\d+",'',text)

def remove_of_words(text):
    text = re.sub(r"\b_([a-zA-z]+)_\b",r"\1",text) #replace _word_ to word
    text = re.sub(r"\b_([a-zA-z]+)\b",r"\1",text) #replace _word to word
    text = re.sub(r"\b([a-zA-z]+)_\b",r"\1",text) #replace word_ to word
    text = re.sub(r"\b([a-zA-Z]+)_([a-zA-Z]+)\b",r"\1 \2", text) #replace word1_word2 to word1 word2
    return text

def remove_less_two(text):
    return re.sub(r'\b\w{1,3}\b',"",text) #remove words <3

def remove_char(dataframe):
    result = re.sub(r"\s+",' ',dataframe)
    result = re.sub(r"^\s+|\s+$","",result)
    result = re.sub(r"\b____________________________\b",'',result)
    return result

def remove_stopwords(text):
    all_stop_words = stopwords.words('english')
    greet_sw = ['hello', 'good', 'morning', 'evening', 'afternoon', 'respected', 'dear', 'madam', 'sincerely',
                'regards', 'truly']
    all_stop_words.extend(greet_sw)
    """custom function to remove the stopwords"""
    tokens = word_tokenize(text)
    token_wsw = [w for w in tokens if w not in all_stop_words]
    filter_str = ' '.join(token_wsw)
    return filter_str

def lemmatized(text):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemma = [lemmatizer.lemmatize(word) for word in tokens]
    filter_str = ' '.join(lemma)
    return filter_str


@csrf_exempt
@api_view(["POST"])
@permission_classes((AllowAny,))
def single(request):
    subject = str(request.data.get("subject"))
    body = str(request.data.get("body"))
    if subject is None or body is None:
        return Response({'error': 'Please Provide both Subject and Body.'},
                        status=HTTP_400_BAD_REQUEST)
    try:
        model = pickle.load(open('Train_AI_Model', 'rb'))
    except:
        return Response({'error': 'Model Not Yet Trained , Train It First'},
                        status=HTTP_400_BAD_REQUEST)
    else:
        data = [[subject,body]]
        df = pd.DataFrame(data=data, columns=['Subject','Body'])
        """**Lower Text Case**"""
        df[['Subject', 'Body']] = df[['Subject', 'Body']].apply(lambda x: x.str.lower())

        """**Removing Emails and URLs -** Patterns: ```regexp(r'[\w\.*]+@[\w\.*]+\b');  regexp(r'\S*@\S*\s?')```"""
        df['Subject'] = df['Subject'].apply(remove_emails_urls)
        df['Body'] = df['Body'].apply(remove_emails_urls)

        """**Removing Dates**"""
        df['Subject'] = df['Subject'].apply(remove_dates)
        df['Body'] = df['Body'].apply(remove_dates)

        """**Removing Useless Words -** ```['from:','sent:','to:','message encrypted','warning:','subject:','fw:','re:','cc:','bcc:']```"""
        df['Body'] = df['Body'].apply(remove_useless)
        df['Subject'] = df['Subject'].apply(remove_useless)

        """**Removing of Punctuations -** `!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~` """
        df['Subject'] = df['Subject'].apply(remove_punctuation)
        df['Body'] = df['Body'].apply(remove_punctuation)

        """**Removing Numbers**"""
        df['Subject'] = df['Subject'].apply(remove_no)
        df['Body'] = df['Body'].apply(remove_no)

        """**Replacing “_word_” , “_word” , “word_” kinds to word**"""
        df['Subject'] = df['Subject'].apply(remove_of_words)
        df['Body'] = df['Body'].apply(remove_of_words)

        """**Removing the Short Characters (<3 words)**"""
        df['Subject'] = df['Subject'].apply(remove_less_two)
        df['Body'] = df['Body'].apply(remove_less_two)

        """**Removing Special Characters (\n,\r...)**"""
        df['Subject'] = df['Subject'].apply(remove_char)
        df['Body'] = df['Body'].apply(remove_char)

        """### **NLP Based Preprocessing**
    
        **Removing Stopwords**
        """
        df['Subject'] = df['Subject'].apply(remove_stopwords)
        df['Body'] = df['Body'].apply(remove_stopwords)

        """**Lemmatization** """

        df['Lemma Subject'] = df['Subject'].apply(lemmatized)
        df['Lemma Body'] = df['Body'].apply(lemmatized)

        data = df

        data['Lemma Message'] = data['Lemma Subject'].astype(str) + " " + data['Lemma Body'].astype(str)

        df = data[['Lemma Message']]

        # retreve the tfidf model from disk
        open_file = open(tfidf_file, 'rb')
        tfidf = pickle.load(open_file)
        open_file.close()

        # generate X_test
        X_test = tfidf.transform(df['Lemma Message']).toarray()

        # y_pred_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)

        # Now y_pred contains numbers in ranges 0..number of categories
        # retrieve the  categories names
        open_file = open(category_filename, 'rb')
        cats = pickle.load(open_file)
        open_file.close()

        pred_cat = cats[y_pred[0]]

        return Response(
        {
            'Category': pred_cat
        },
        status=HTTP_200_OK)



@csrf_exempt
@api_view(["POST"])
@permission_classes((AllowAny,))
def batch(request):
    subject_list = request.data.get("subject_list")
    body_list = request.data.get("body_list")
    bodys = body_list.replace('\n',' ').replace('\r',' ')
    subjects = json.loads(subject_list)
    bodies = ast.literal_eval(bodys)

    data = []
    for idx in range(len(subjects)):
        temp = [subjects[idx],bodies[idx]]
        data.append(temp)

    try:
        model = pickle.load(open('Train_AI_Model', 'rb'))
    except:
        return Response({'error': 'Model Not Yet Trained , Train It First'},
                        status=HTTP_400_BAD_REQUEST)
    else:
        df = pd.DataFrame(data=data, columns=['Subject', 'Body'])
        """**Lower Text Case**"""
        df[['Subject', 'Body']] = df[['Subject', 'Body']].apply(lambda x: x.str.lower())

        """**Removing Emails and URLs -** Patterns: ```regexp(r'[\w\.*]+@[\w\.*]+\b');  regexp(r'\S*@\S*\s?')```"""
        df['Subject'] = df['Subject'].apply(remove_emails_urls)
        df['Body'] = df['Body'].apply(remove_emails_urls)

        """**Removing Dates**"""
        df['Subject'] = df['Subject'].apply(remove_dates)
        df['Body'] = df['Body'].apply(remove_dates)

        """**Removing Useless Words -** ```['from:','sent:','to:','message encrypted','warning:','subject:','fw:','re:','cc:','bcc:']```"""
        df['Body'] = df['Body'].apply(remove_useless)
        df['Subject'] = df['Subject'].apply(remove_useless)

        """**Removing of Punctuations -** `!"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~` """
        df['Subject'] = df['Subject'].apply(remove_punctuation)
        df['Body'] = df['Body'].apply(remove_punctuation)

        """**Removing Numbers**"""
        df['Subject'] = df['Subject'].apply(remove_no)
        df['Body'] = df['Body'].apply(remove_no)

        """**Replacing “_word_” , “_word” , “word_” kinds to word**"""
        df['Subject'] = df['Subject'].apply(remove_of_words)
        df['Body'] = df['Body'].apply(remove_of_words)

        """**Removing the Short Characters (<3 words)**"""
        df['Subject'] = df['Subject'].apply(remove_less_two)
        df['Body'] = df['Body'].apply(remove_less_two)

        """**Removing Special Characters (\n,\r...)**"""
        df['Subject'] = df['Subject'].apply(remove_char)
        df['Body'] = df['Body'].apply(remove_char)

        """### **NLP Based Preprocessing**

        **Removing Stopwords**
        """
        df['Subject'] = df['Subject'].apply(remove_stopwords)
        df['Body'] = df['Body'].apply(remove_stopwords)

        """**Lemmatization** """

        df['Lemma Subject'] = df['Subject'].apply(lemmatized)
        df['Lemma Body'] = df['Body'].apply(lemmatized)

        data = df

        data['Lemma Message'] = data['Lemma Subject'].astype(str) + " " + data['Lemma Body'].astype(str)

        df = data[['Lemma Message']]

        # retreve the tfidf model from disk
        open_file = open(tfidf_file, 'rb')
        tfidf = pickle.load(open_file)
        open_file.close()

        # generate X_test
        X_test = tfidf.transform(df['Lemma Message']).toarray()

        # y_pred_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)

        # Now y_pred contains numbers in ranges 0..number of categories
        # retrieve the  categories names
        open_file = open(category_filename, 'rb')
        cats = pickle.load(open_file)
        open_file.close()

        pred_cat = [cats[y_pred[x]] for x in range(len(y_pred))]

        return Response(
            {
                'Categories': pred_cat
            },
            status=HTTP_200_OK)
