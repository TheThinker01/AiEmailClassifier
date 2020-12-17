from django.shortcuts import render
from django.http import HttpResponse
from zipfile import ZipFile, is_zipfile, Path
import os
from outlook_msg import  Message
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


import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers.experimental.preprocessing import TextVectorization
from keras.preprocessing.text import one_hot,Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Reshape,RepeatVector,LSTM,Dense,Flatten,Bidirectional,Embedding,Input,Layer,GRU,Multiply,Activation,Lambda,Dot,TimeDistributed,Dropout,Embedding
from keras.models import Model
from keras.activations import softmax,selu,sigmoid
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
from keras.regularizers import l2
from keras.constraints import min_max_norm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
import tensorflow as tf
import os
import gensim
from tqdm import tqdm

import keras
from .attention_with_context import *
from .sentence_encoder import *
from .word_encoder import *

###########    GLOBALS   ##################################

tfidf = None
category_filename = "category.pkl"
test_filelist_filename = "test_filenames.pkl"
tfidf_file = 'tfidf_file.pkl'

##########     VIEWS     ##################################
# Create your views here.

def index1(request):
    # trained = True
    # new_model = keras.models.load_model('model_and_weights.h5',custom_objects={'word_encoder': word_encoder})
    #
    # try:
    #     pass
    # except:
    #     trained = False
    # finally:
        if request.method == 'GET':
            return render(request, 'home.html', {'train_success': False})
        if request.method == 'POST':
            return render(request, 'home.html', {'train_success': False})


def submit_data(request):
    if request.method == 'GET':
        return HttpResponse("<h1>FORBIDDEN!</h1> <h2>This Page Cannot Be Accessed Directly.</h2>")
    elif request.method == 'POST':
        print("For debug")
        print(request.FILES)
        file = request.FILES['train']
        if is_zipfile(file) is False:
            return HttpResponse("<h1>FORBIDDEN!<h1><h2>You Need To Upload A Zip File Containing The DataSet</h2>")

        # Stores the categories obtained
        cats = []

        # Extract the files to a new directory named by the input folder name
        file_path = ""
        with ZipFile(file,'r') as myzip:
            path = Path(myzip)
            # print(path.name)
            for dir in path.iterdir():
                cats.append(dir.name)
            file_path = os.getcwd() + '\\' + myzip.filename.split('.')[0]
            print(file_path)
            myzip.extractall(path=file_path)

        # save the category file to disk, so that can be retrieved while testing
        open_file = open(category_filename,'wb')
        pickle.dump(cats,open_file)
        open_file.close()

        # Now the Zip file has been extracted to the working directory and  file_path is the absolute path of the folder
        data = []
        for cat in cats:
            sub_path = file_path + '\\' + cat
            for root,directories,files in os.walk(sub_path):
                for file in files:
                    abs_path = os.path.join(root,file)
                    # with extract_msg.Message(abs_path) as msg:
                    with open(abs_path) as msg_file:
                        msg = Message(msg_file)
                        sub = "\""+msg.subject+"\""
                        body = "\""+msg.body+"\""
                        temp = [cat, sub, body]
                    data.append(temp)
        df = pd.DataFrame(data,columns=['Category', 'Subject', 'Body'])

        csv_path = file_path+'.csv'
        df.to_csv(csv_path, index=False, header=True)
        preprocess(csv_path=csv_path)
        with open(csv_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="text/csv")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(csv_path)
            return response
        return HttpResponse("DONE")


def index2(request):
    trained = True
    try:
        f = open('Train_AI_Model','rb')
    except:
        trained = False
    finally:
        if request.method == 'GET':
            return render(request,'home2.html', {'train_success': trained})
        if request.method == 'POST':
            return render(request,'home2.html', {'train_success': trained})

#####  FUNCTION TO PREPROCESS DATA   #############################

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

def preprocess(csv_path,test=False):
    """**Upload the Data**"""
    df = pd.read_csv(csv_path)

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

    """**Saving of Preprocessed Data**"""
    if test:
        df.to_csv('Pre_Test.csv', index=False)
    else:
        df.to_csv('Pre_Train.csv', index=False)

############  MACHINE LEARNING TRAINING FUNCTION  #######################################


def trainml(request):
    try:
        data = pd.read_csv("Pre_Train.csv", encoding='utf-8')
    except:
        return HttpResponse("<h1>FORBIDDEN!<h1><h2>You Need To Upload the Train Dataset first</h2>")
    else:
        data['Category_Id'] = data['Category'].factorize()[0]

        data['Lemma Message'] = data['Lemma Subject'].astype(str) + " " + data['Lemma Body'].astype(str)
        df = data[['Category_Id', 'Lemma Message']]

        category_id_df = data[['Category', 'Category_Id']].drop_duplicates().sort_values('Category_Id')

        """**Text Vectorization**"""
        global tfidf
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                                stop_words='english')

        features = tfidf.fit_transform(df['Lemma Message']).toarray()

        # save the tfidf model
        open_file = open(tfidf_file,'wb')
        pickle.dump(tfidf,open_file)
        open_file.close()

        # continue with everything
        labels = df.Category_Id

        """**Train the Model**"""
        model = RandomForestClassifier(random_state=0, n_estimators=1600, min_samples_split=20, min_samples_leaf=2,
                                       max_features='sqrt', max_depth=15, bootstrap='True')

        # Split the Data
        X_train = features
        y_train = labels
        # Train the Algorithm
        train_model = model.fit(X_train, y_train)

        """**Save the Model**"""
        pickle.dump(train_model, open('Train_AI_Model', 'wb'))

        return render(request, 'home2.html', {'train_success' : True})


####################################################################  DEEP LEARNING TRAIN FUNCTION  #######################################


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

def emb_loss(model, X, Y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  Y_tilde = model(X, training=training)
  a = tf.keras.losses.CategoricalCrossentropy()
  E_loss_T0 = a(Y, Y_tilde)
  return E_loss_T0

def grad_emb(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = emb_loss(model, inputs,targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)


def traindl(request):
    try:
        df = pd.read_csv('Pre_Train.csv')
    except:
        return HttpResponse("<h1>FORBIDDEN!<h1><h2>You Need To First Upload The Train DataSet</h2>")
    else:
        df = df.sample(frac=1)
        """**Applying the Categories to Numbers**"""

        df2 = pd.get_dummies(df['Category'])

        # Voabulary and number of words
        vocab = 10000
        sequence_length = 40

        df1 = df[['Lemma Body']]

        """**Converting to Numpy Array**"""

        # Prepare Tokenizer
        t = Tokenizer()

        words = list(df1['Lemma Body'])
        t.fit_on_texts(words)
        vocab_size = len(t.word_index) + 1

        # integer encode the documents
        encoded_docs = t.texts_to_sequences(words)

        # pad documents to a max length of 40 words
        padded_docs = pad_sequences(encoded_docs, maxlen=sequence_length, padding='post')

        # Preparing the labels into arrays
        labels = df2.to_numpy()

        """**Reshape to (Documents X Sentences X Words)**"""

        a = tf.reshape(padded_docs, (297, 4, 10))

        x_train = a[:]

        y_train = labels[:]

        index_dloc = 'word_embeddings/glove_6B_300d.txt'

        """Here we create a dictionary named embedding vector, which will have keys, defined as words, present in the glove embedding file and the value of that key will be the embedding present in the file. This dictionary will contain all the words available in the glove embedding file."""

        embedding_index = dict()
        f = open(index_dloc)
        for line in tqdm(f):
            value = line.split(' ')
            word = value[0]
            coef = np.array(value[1:], dtype='float32')
            embedding_index[word] = coef
        f.close()

        # create a weight matrix for words in training docs
        embedding_matrix = np.zeros((vocab_size, 300))
        for word, i in t.word_index.items():
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        words = 10
        sentences = 4
        document = 16
        units = 64

        # Shape of Input = (No of document, No of Sentences, No of Words)
        input = Input(batch_input_shape=(document, sentences, words))

        # Word Encoder
        # Reshape into (No of Documents * No of Sentences, No of Words)
        # Embedding layer Output Shape = (No of Documents * No of Sentences, No of Words, Embedding Dimension)
        a1 = word_encoder(lstm_units=128, dense_units=64, emb=300, document=document, sentences=sentences, words=words,
                          embeddings=embedding_matrix)(input)
        a2 = AttentionWithContext()(a1)

        # Sentence Encoder
        a3 = sentence_encoder(lstm_units=128, dense_units=64, document=document, sentences=sentences, units=units)(a2)
        a4 = AttentionWithContext()(a3)
        a5 = Dropout(0.2)(a4)

        # Document Classification
        output = Dense(3, activation='softmax')(a5)
        model = Model(input, output)

        # print('Start Network Training')

        # Instantiate an optimizer
        adam = Adam(learning_rate=0.000099, beta_1=0.9, beta_2=0.999, amsgrad=False)

        # keep results for plotting
        train_loss_results = []

        for epoch in range(15):
            epoch_loss_avg = tf.keras.metrics.CategoricalAccuracy()

            # Training Loop, using the batches of 16
            for i in range(0, 13):
                x = x_train[i * 16:(i + 1) * 16]
                y = y_train[i * 16:(i + 1) * 16]

                # Optimize the model
                loss_value, grads = grad_emb(model, x, y)
                adam.apply_gradients(zip(grads, model.trainable_variables))

                # Track progress
                epoch_loss_avg.update_state(y, model(x))  # Add current batch loss
                # Compare predicted label to actual label

            # End epoch
            train_loss_results.append(epoch_loss_avg.result())

            # print("Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))

        # print('Finish Network Training')
        model.save('model_and_weights')
        return render(request, 'home.html',{'train_succes':True})


#############################################################  SUBMIT TEST DATASET FUNCTION  ###################################


def submit_test(request):
    if request.method == 'GET':
        return HttpResponse("<h1>FORBIDDEN!</h1> <h2>This Page Cannot Be Accessed Directly.</h2>")
    elif request.method == 'POST':
        # print("For debug")
        # print(request.FILES)
        file = request.FILES['test']
        if is_zipfile(file) is False:
            return HttpResponse("<h1>FORBIDDEN!<h1><h2>You Need To Upload A Zip File Containing The DataSet</h2>")

        # Stores the categories obtained
        cats = []

        # Extract the files to a new directory named by the input folder name
        file_path = ""
        with ZipFile(file,'r') as myzip:
            path = Path(myzip)
            # print(path.name)
            for dir in path.iterdir():
                cats.append(dir.name)
            file_path = os.getcwd() + '\\' + myzip.filename.split('.')[0]
            print(file_path)
            myzip.extractall(path=file_path)

        # Now the Zip file has been extracted to the working directory and  file_path is the absolute path of the folder
        data = []
        file_list = []
        for cat in cats:
            sub_path = file_path + '\\' + cat
            for root,directories,files in os.walk(sub_path):
                for file in files:
                    abs_path = os.path.join(root,file)
                    # with extract_msg.Message(abs_path) as msg:
                    with open(abs_path) as msg_file:
                        msg = Message(msg_file)
                        sub = "\""+msg.subject+"\""
                        body = "\""+msg.body+"\""
                        temp = [sub, body]
                    data.append(temp)
                    file_list.append(file)

        # save the names of files for later use in classifying
        open_file = open(test_filelist_filename, 'wb')
        pickle.dump(file_list,open_file)
        open_file.close()


        # Create the dataframe
        df = pd.DataFrame(data,columns=['Subject', 'Body'])

        csv_path = file_path+'.csv'
        df.to_csv(csv_path, index=False, header=True)
        preprocess(csv_path=csv_path,test=True)
        with open(csv_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="text/csv")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(csv_path)
            return response
        return HttpResponse("DONE")


#############################################################  ML TEST DATASET FUNCTION  ###################################


def testml(request):
    try:
        data = pd.read_csv("Pre_Test.csv", encoding='utf-8')
        data['Lemma Message'] = data['Lemma Subject'].astype(str) + " " + data['Lemma Body'].astype(str)

        df = data[['Lemma Message']]

        # retreve the tfidf model from disk
        open_file = open(tfidf_file,'rb')
        tfidf = pickle.load(open_file)
        open_file.close()

        # generate X_test
        X_test = tfidf.transform(df['Lemma Message']).toarray()
        model = pickle.load(open('Train_AI_Model', 'rb'))
        # y_pred_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)

        # Now y_pred contains numbers in ranges 0..number of categories
        # retrieve the  categories names
        open_file = open(category_filename, 'rb')
        cats = pickle.load(open_file)
        open_file.close()

        # Next retrieve the filenames from disk
        open_file = open(test_filelist_filename, 'rb')
        file_list = pickle.load(open_file)
        open_file.close()

        df_dat = []
        for idx, f in enumerate(file_list):
            temp = [f, cats[y_pred[idx]]]
            df_dat.append(temp)

        df = pd.DataFrame(df_dat, columns=['Filename', 'Category'])
        df.to_csv('Test_Output.csv', index=False, header=True)

        return render(request, 'home2.html', {'train_success': True ,'test_done': True, 'output': df_dat})

    except:
        return HttpResponse("<h1>FORBIDDEN!</h1> <h2>First upload the Test Dataset.</h2>")

#############################################################  DL TEST DATASET FUNCTION  ###################################

model = None


def testdl(request):
    try:
        df = pd.read_csv("Pre_Test.csv", encoding='utf-8')
    except:
        return HttpResponse("<h1>FORBIDDEN!<h1><h2>You Need To Upload A Zip File Containing The Test DataSet</h2>")
    else:
        df = df.sample(frac=1)
        # Vocabulary and number of words
        vocab = 10000
        sequence_length = 40
        word = 10
        sentences = int(sequence_length / word)
        document = 8    #####
        units = 64

        df1 = df[['Lemma Body']]
        # Prepare Tokenizer
        t = Tokenizer()

        wors = list(df1['Lemma Body'])
        t.fit_on_texts(wors)  #
        vocab_size = len(t.word_index) + 1

        # integer encode the documents
        encoded_docs = t.texts_to_sequences(wors)

        # pad documents to a max length of 40 words
        padded_docs = pad_sequences(encoded_docs, maxlen=sequence_length, padding='post')

        """**Reshape to (Documents X Sentences X Words)**"""

        a = tf.reshape(padded_docs, (df.shape[0], int(sequence_length/10), 10))
        x_test = a[:]

        batch = document
        global model
        # try:
        if model is None:
            model = keras.models.load_model('model_name')
        # except:
        # return HttpResponse("<h1>FORBIDDEN!<h1><h2>You Need To Train The Model First</h2>")
        # else:
        result = np.zeros((1, 3))
        # result = model.predict(a[:])
        for i in range(0, int(x_test.shape[0] / batch)):
            predictions = model.predict(x_test[batch * i:batch * (i + 1)])
            result = np.vstack((result, predictions))
        result = np.delete(result, (0), axis=0)
        b = pd.DataFrame(result, columns=['MDU', 'Retirements', 'Transfers'])

        b = pd.DataFrame(b.idxmax(axis=1), columns=['Predicted'])

        open_file = open(category_filename, 'rb')
        cats = pickle.load(open_file)
        open_file.close()

        # Next retrieve the filenames from disk
        open_file = open(test_filelist_filename, 'rb')
        file_list = pickle.load(open_file)
        open_file.close()

        df_dat = []
        for idx, f in enumerate(file_list):
            if idx >= int(b.shape[0]/batch)*batch:
                break
            temp = [f, b.iloc[idx][0]]
            df_dat.append(temp)

        df = pd.DataFrame(df_dat, columns=['Filename', 'Category'])
        df.to_csv('Test_Output.csv', index=False, header=True)

        return render(request, 'home.html', {'test_success': True, 'test_done': True, 'output': df_dat})

#############################################################  DOWNLOAD TEST OUTPUT FUNCTION  ##########################

def download(request):
     try:
        fh = open('Test_Output.csv', 'rb')
        response = HttpResponse(fh.read(), content_type="text/csv")
        response['Content-Disposition'] = 'inline; filename=Test_Output.csv'
        return response
     except:
         return HttpResponse("<h1>FORBIDDEN!</h1> <h2>Train the model first.</h2>")






