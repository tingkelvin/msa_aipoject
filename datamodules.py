import xml.etree.ElementTree as ET
import numpy as np
from numpy.core.numeric import indices
import pandas as pd
import string
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

def loadXML(path, polarities):
    tree = ET.parse(path)
    sentences = tree.getroot()
    texts = []
    aspectTerms_ = []
    polarity_ = []
    wl = []
    wr = []
    for sentence in sentences.findall('sentence'):
        aspectTerms = sentence.find('aspectTerms')
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
                    texts.append(text_withoutPun)
                    polarity_.append(polarity)

    sentence = pd.DataFrame(columns=['text'])
    sentence['text'] = texts
    sentence['left2right'] = wl
    sentence['right2left'] = wr
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
    X_train_wl = [text.strip() for text in data_train['left2right']]
    X_train_wr = [text.strip() for text in data_train['right2left']]
    X_test_wl = [text.strip() for text in data_test['left2right']]
    X_test_wr = [text.strip() for text in data_test['right2left']]
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    tokenizer.fit_on_texts(X_test)
    words_to_index = tokenizer.word_index
    i = len(words_to_index) + 1

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
    X_test_indices = pad_sequences(X_test_indices, maxlen=maxLen)

    X_train_wl = pad_sequences(X_train_wl)
    X_train_wr = pad_sequences(X_train_wr)

    maxLen = max([len(i) for i in X_train_wl])
    X_test_wl = pad_sequences(X_test_wl, maxlen=maxLen)

    maxLen = max([len(i) for i in X_train_wr])
    X_test_wr = pad_sequences(X_test_wr, maxlen=maxLen)

    return words_to_index, X_train_indices, X_test_indices, X_train_wl, X_train_wr, X_test_wl, X_test_wr



def sentence2index(sentence, target, word2index):
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    sentence = sentence.split(' ')
    print(sentence)
    left2rightTemp = sentence[:target]
    right2leftTemp = sentence[target + 1:]
    print(right2leftTemp)
    right2leftTemp = right2leftTemp[::-1]
    left2rightIndecies = []
    right2leftIndecies = []
    print(left2rightTemp)
    print(right2leftTemp)
    for word in left2rightTemp:
        word = word.strip().lower()
        if word in word2index.keys():
            left2rightIndecies.append(word2index[word])
        else:
            left2rightIndecies.append(len(word2index) + 1)

    for word in right2leftTemp:
        word = word.strip().lower()
        if word in word2index.keys():
            right2leftIndecies.append(word2index[word])
        else:
            right2leftIndecies.append(len(word2index) + 1)

    return pad_sequences([left2rightIndecies],
                         maxlen=61), pad_sequences([right2leftIndecies],
                                                   maxlen=61)