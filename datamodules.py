import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import string
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

def loadXML(path, polarities):
    tree = ET.parse(path)
    sentences = tree.getroot()
    texts = []
    aspectTerms_ = []
    head = []
    head_rel = []
    polarity_ = []
    aspectTerms_rel = []
    embed = []
    span_ = []
    wl = []
    wr = []
    single_term = 0
    for sentence in sentences.findall('sentence'):
        aspectTerms = sentence.find('aspectTerms')
        aspectTerm = []
        if aspectTerms != None:
            text = sentence.find('text').text.lower()
            text_withoutPun = text.translate(
                str.maketrans('', '', string.punctuation))
            for aT in aspectTerms.iter('aspectTerm'):
                polarity = aT.get('polarity').lower()
                if polarity in polarities:
                    span = (aT.get('from'), aT.get('to'))
                    wl.append(text[:int(span[0])].translate(
                        str.maketrans('', '', string.punctuation)))
                    wr.append(text[int(span[1]):].translate(
                        str.maketrans('', '', string.punctuation)))
                    apectTerm = aT.get('term').lower()
                    texts.append(text_withoutPun)
                    aspectTerms_.append(apectTerm)
                    polarity_.append(polarity)
    sentence = pd.DataFrame(columns=['text', 'aspectTerm'])
    sentence['text'] = texts
    sentence['wl'] = wl
    sentence['wr'] = wr
    sentence['aspectTerm'] = aspectTerms_
    sentence['polarity'] = polarity_
    sentence['label'] = sentence['polarity'].astype('category').cat.codes
    sentence = sentence.sample(frac=1, random_state=1221)
    X = [text for text in sentence['text']]
    Y = np.eye(3)[sentence['label'].values.reshape(-1)]

    return sentence, X, Y

def loadGloVec(glove_vec):
    with open(glove_vec, 'r', encoding='UTF-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            w_line = line.split()
            curr_word = w_line[0]
            word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)
    return word_to_vec_map

def word2index(X_train, X_test, data_train, data_test):
    X_train_wl = [text.strip() for text in data_train['wl']]
    X_train_wr = [text.strip() for text in data_train['wr']]
    X_test_wl = [text.strip() for text in data_test['wl']]
    X_test_wr = [text.strip() for text in data_test['wr']]
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    tokenizer.fit_on_texts(X_test)
    words_to_index = tokenizer.word_index
    i = len(words_to_index) + 1
    for ap in data_train["aspectTerm"]:
        if ap not in words_to_index.keys():
            words_to_index[ap] = i
            i += 1
    
    for ap in data_test["aspectTerm"]:
        if ap not in words_to_index.keys():
            words_to_index[ap] = i
            i += 1
    
    X_train_indices = tokenizer.texts_to_sequences(X_train)
    X_test_indices = tokenizer.texts_to_sequences(X_test)
    
    X_train_wl = tokenizer.texts_to_sequences(X_train_wl)
    X_train_wr = tokenizer.texts_to_sequences(X_train_wr)
    X_train_wr = [x[::-1] for x in X_train_wr]
    X_test_wl = tokenizer.texts_to_sequences(X_test_wl)
    X_test_wr = tokenizer.texts_to_sequences(X_test_wr)
    X_test_wr = [x[::-1] for x in X_test_wr]
    
    
    X_train_indices = pad_sequences(X_train_indices)
    maxLen = max([len(i) for i in X_train_indices])
    X_test_indices = pad_sequences(X_test_indices,maxlen=maxLen)
    
    
    X_train_wl = pad_sequences(X_train_wl)
    X_train_wr = pad_sequences(X_train_wr)
    
    maxLen = max([len(i) for i in X_train_wl])
    X_test_wl = pad_sequences(X_test_wl, maxlen= maxLen)
    
    maxLen = max([len(i) for i in X_train_wr])
    X_test_wr = pad_sequences(X_test_wr, maxlen= maxLen)
    
    return words_to_index, X_train_indices, X_test_indices, X_train_wl, X_train_wr, X_test_wl, X_test_wr