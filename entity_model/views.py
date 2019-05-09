from django.shortcuts import render
import numpy as np
import array
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import tensorflow as tf
import re
import spacy
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
#from nltk.stem import PorterStemmer
from entity_model.models import entities

stop_words = set(stopwords.words('english'))
nlp = spacy.load('en')
#filename = 'entitymodel.h5'


def load_dataset():
    sentences=[]
    entity=[]
    for row in entities.objects.all():
        sentences.append(row.Question)
        entity.append(row.Entity)
    global unique_entity
    unique_entity=list(set(entity))
    return (entity, sentences)


def get_unique_entity():
    load_dataset()
    return unique_entity

def clean(sentences):
    words = []
    # nlp = spacy.load('en')
    for sent in sentences:
        doc = nlp(sent)
        s = ""
        for w in doc:
            s = s + " " + w.lemma_
        w = word_tokenize(s)
        word = [wrd for wrd in w if not wrd in stop_words and wrd != 'i' and wrd != 'I']
        words.append([i.lower() for i in word])
    return words


def create_tokenizer(words):
    token = Tokenizer()
    token.fit_on_texts(words)
    return token


def length(words):
    return(len(max(words, key = len)))


def encoding_doc(token, words):
    return(token.texts_to_sequences(words))


def padding_doc(encoded_doc):
    return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))


def one_hot(encode):
    o = OneHotEncoder(sparse=False,categories='auto')
    return o.fit_transform(encode)


def create_model(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 128, input_length=max_length, trainable=False))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(90, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(len(unique_entity), activation="softmax"))
    return model


def train_entity_model():
    entity, sentences = load_dataset()
    cleaned_words = clean(sentences)
    global word_tokenizer
    word_tokenizer = create_tokenizer(cleaned_words)
    vocab_size = len(word_tokenizer.word_index) + 1
    global max_length
    max_length = length(cleaned_words)
    encoded_doc = encoding_doc(word_tokenizer, cleaned_words)
    padded_doc = padding_doc(encoded_doc)
    output_tokenizer = create_tokenizer(unique_entity)
    encoded_output = encoding_doc(output_tokenizer, entity)
    encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)
    output_one_hot = one_hot(encoded_output)
    x_train, x_test, y_train, y_test = train_test_split(padded_doc, output_one_hot, shuffle=True, test_size=0.1)
    model = create_model(vocab_size)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    #checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    hist = model.fit(x_train, y_train, epochs=1, batch_size=32, validation_data=(x_test, y_test))
    #callbacks=[checkpoint])
    global entity_model
    entity_model=model #=load_model("entitymodel.h5")
    global graph
    graph = tf.get_default_graph()


def predict_entity(text,classes):
    clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
    doc = nlp(clean)
    s = ""
    for w in doc:
        s = s + " " + w.lemma_
    test_word = word_tokenize(s)
    test_word = [w.lower() for w in test_word if not w in stop_words and w != 'i' and w != 'I']
    test_ls = word_tokenizer.texts_to_sequences(test_word)
    if [] in test_ls:
        test_ls = list(filter(None, test_ls))
    test_ls = np.array(test_ls).reshape(1, len(test_ls))
    x = padding_doc(test_ls)
    with graph.as_default():
        pred = entity_model.predict_proba(x)
        classes = np.array(classes)
        ids = np.argsort(-pred[0])
        classes = classes[ids]
        predictions = -np.sort(-pred[0])
        return classes[0]
