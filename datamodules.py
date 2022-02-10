import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import spacy

pos_cat = pd.DataFrame({'pos':['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM','PART', 'PRON', 'PROPN', 'SCONJ', 'SPACE', 'VERB', 'X']},dtype="category")
tag_cat = pd.DataFrame({'tag':['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD','NN', 'NNP','NNPS', 'NNS', 'PDT', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP','TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WRB','XX', '_SP']},dtype="category")
dep_cat = pd.DataFrame({'dep':['ROOT', 'acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod', 'appos',
       'attr', 'aux', 'auxpass', 'cc', 'ccomp', 'compound', 'conj', 'csubj',
       'csubjpass', 'dative', 'dep', 'det', 'dobj', 'expl', 'intj', 'mark',
       'meta', 'neg', 'nmod', 'npadvmod', 'nsubj', 'nsubjpass', 'nummod',
       'oprd', 'parataxis', 'pcomp', 'pobj', 'poss', 'preconj', 'predet',
       'prep', 'prt', 'punct', 'quantmod', 'relcl', 'xcomp']},
      dtype='object')

def loadXML(path, polarities):
    tree = ET.parse(path)
    sentences = tree.getroot()
    texts = []
    polarity_ = []
    lef2right = []
    right2left = []
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
                    lef2right.append(text[:int(span[0])].translate(
                        str.maketrans('', '', string.punctuation)))
                    right2left.append(text[int(span[1]):].translate(
                        str.maketrans('', '', string.punctuation)))
                    texts.append(text_withoutPun)
                    polarity_.append(polarity)

    sentence = pd.DataFrame(columns=['text'])
    sentence['text'] = texts
    sentence['left2right'] = lef2right
    sentence['right2left'] = right2left
    sentence['polarity'] = polarity_
    sentence['label'] = sentence['polarity'].astype('category').cat.codes
    sentence = sentence.sample(frac=1, random_state=1221)
    X = [text for text in sentence['text']]
    Y = np.eye(3)[sentence['label'].values.reshape(-1)]

    return sentence, X, Y

def loadGloVec(glove_vec):
    with open(glove_vec, 'r', encoding='UTF-8') as f:
        word_to_vec_map = {}
        for line in f:
            w_line = line.split()
            curr_word = w_line[0]
            word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)
    return word_to_vec_map

def word2index(X_train, X_test, data_train, data_test):
    X_train_lef2right = [text.strip() for text in data_train['left2right']]
    X_train_right2left = [text.strip() for text in data_train['right2left']]
    X_test_lef2right = [text.strip() for text in data_test['left2right']]
    X_test_right2left = [text.strip() for text in data_test['right2left']]
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    tokenizer.fit_on_texts(X_test)
    words_to_index = tokenizer.word_index

    X_train_indices = tokenizer.texts_to_sequences(X_train)
    X_test_indices = tokenizer.texts_to_sequences(X_test)

    X_train_lef2right = tokenizer.texts_to_sequences(X_train_lef2right)
    X_train_right2left = tokenizer.texts_to_sequences(X_train_right2left)
    X_train_right2left = [x[::-1] for x in X_train_right2left]
    X_test_lef2right = tokenizer.texts_to_sequences(X_test_lef2right)
    X_test_right2left = tokenizer.texts_to_sequences(X_test_right2left)
    X_test_right2left = [x[::-1] for x in X_test_right2left]

    X_train_indices = pad_sequences(X_train_indices)
    maxLen = max([len(i) for i in X_train_indices])
    X_test_indices = pad_sequences(X_test_indices, maxlen=maxLen)

    X_train_lef2right = pad_sequences(X_train_lef2right)
    X_train_right2left = pad_sequences(X_train_right2left)

    maxLen = max([len(i) for i in X_train_lef2right])
    X_test_lef2right = pad_sequences(X_test_lef2right, maxlen=maxLen)

    maxLen = max([len(i) for i in X_train_right2left])
    X_test_right2left = pad_sequences(X_test_right2left, maxlen=maxLen)

    return words_to_index, X_train_indices, X_test_indices, X_train_lef2right, X_train_right2left, X_test_lef2right, X_test_right2left

def depedency2seq(data):
    nlp = spacy.load("en_core_web_sm")
    pos = []
    tag = []
    dep = []
    for i in range(len(data)):
        t = data[i]
        doc = nlp(t)
        posa = []
        taga = []
        depa = []
        for token in doc:
            if token.tag_ != "_SP":
                try:
                    posa.append(pos_cat.index[pos_cat['pos'] == token.pos_].values[0]+1)
                    taga.append(tag_cat.index[tag_cat['tag'] == token.tag_].values[0]+1)
                    depa.append(dep_cat.index[dep_cat['dep'] == token.dep_].values[0]+1)
                except Exception as e:
                    print(t)
                    print(e.__class__)
                    print("Next")
                    print()
               
        pos.append(posa)
        tag.append(taga)
        dep.append(depa)
    return pos,tag,dep
def sliceIndex(lef2right, right2left, pos, tag, dep):
    lef2right_pos_sliced = []
    lef2right_tag_sliced = []
    lef2right_dep_sliced = []
    right2left_pos_sliced = []
    right2left_tag_sliced = []
    right2left_dep_sliced = []
    
    for i in range(len(lef2right)):
        s_len = len(lef2right[i].split())
        lef2right_pos_sliced.append(pos[i][:s_len])
        lef2right_tag_sliced.append(tag[i][:s_len])
        lef2right_dep_sliced.append(dep[i][:s_len])
    for j in range(len(right2left)):
        s_len = len(right2left[j].split())
        if s_len != 0:
            right2left_pos_sliced.append(pos[j][-s_len:][::-1])
            right2left_tag_sliced.append(tag[j][-s_len:][::-1])
            right2left_dep_sliced.append(dep[j][-s_len:][::-1])
        else:
            right2left_pos_sliced.append([])
            right2left_tag_sliced.append([])
            right2left_dep_sliced.append([])
    return pad_sequences(lef2right_pos_sliced,maxlen=61),pad_sequences(lef2right_tag_sliced,maxlen=61),pad_sequences(lef2right_dep_sliced,maxlen=61),pad_sequences(right2left_pos_sliced,maxlen=61),pad_sequences(right2left_tag_sliced,maxlen=61),pad_sequences(right2left_dep_sliced,maxlen=61)
        
    #pad_sequences(pos,value=-20), pad_sequences(tag,value=-20), pad_sequences(dep,value=-20)
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
