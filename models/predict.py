from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import string
import spacy
import pandas as pd

targetLSTM = load_model('spacyLSTM-POSTAGDEP')
file_word2index = open("word2index", 'rb')
word2index = pickle.load(file_word2index)
pos_cat = pd.DataFrame({'pos':['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM','PART', 'PRON', 'PROPN', 'SCONJ', 'SPACE', 'VERB', 'X']},dtype="category")
tag_cat = pd.DataFrame({'tag':['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD','NN', 'NNP','NNPS', 'NNS', 'PDT', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP','TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WRB','XX', '_SP']},dtype="category")
dep_cat = pd.DataFrame({'dep':['ROOT', 'acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod', 'appos',
       'attr', 'aux', 'auxpass', 'cc', 'ccomp', 'compound', 'conj', 'csubj',
       'csubjpass', 'dative', 'dep', 'det', 'dobj', 'expl', 'intj', 'mark',
       'meta', 'neg', 'nmod', 'npadvmod', 'nsubj', 'nsubjpass', 'nummod',
       'oprd', 'parataxis', 'pcomp', 'pobj', 'poss', 'preconj', 'predet',
       'prep', 'prt', 'punct', 'quantmod', 'relcl', 'xcomp']},
      dtype='object')

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
        s_len = len(lef2right[i])
        if s_len!=0:
            lef2right_pos_sliced.append(pos[i][:s_len])
            lef2right_tag_sliced.append(tag[i][:s_len])
            lef2right_dep_sliced.append(dep[i][:s_len])
        else:
            lef2right_pos_sliced.append([])
            lef2right_tag_sliced.append([])
            lef2right_dep_sliced.append([])
    for j in range(len(right2left)):
        s_len = len(right2left[i])
        if s_len != 0:
            right2left_pos_sliced.append(pos[j][-s_len:][::-1])
            right2left_tag_sliced.append(tag[j][-s_len:][::-1])
            right2left_dep_sliced.append(dep[j][-s_len:][::-1])
        else:
            right2left_pos_sliced.append([])
            right2left_tag_sliced.append([])
            right2left_dep_sliced.append([])
    return pad_sequences(lef2right_pos_sliced,maxlen=61),pad_sequences(lef2right_tag_sliced,maxlen=61),pad_sequences(lef2right_dep_sliced,maxlen=61),pad_sequences(right2left_pos_sliced,maxlen=61),pad_sequences(right2left_tag_sliced,maxlen=61),pad_sequences(right2left_dep_sliced,maxlen=61)
sentence = "The food is not good"

left2rightIndecies, right2leftIndecies = sentence2index(
    sentence=sentence,  
    target=1,
    word2index=word2index,
)


test_pos, test_tag, test_dep = depedency2seq([sentence])

left2right_pos_te, left2right_tag_te, left2right_dep_te, right2left_pos_te, right2left_tag_te, right2left_dep_te = sliceIndex([left2rightIndecies],[right2leftIndecies], test_pos, test_tag, test_dep)
loadedTagetModel = load_model('TargetLSTM')
loadedSpacyModel = load_model('spacyLSTM-POSTAGDEP')

predict_class = loadedTagetModel.predict((left2rightIndecies, right2leftIndecies))
print("Prediction in target lstm", predict_class)
predict_class = loadedSpacyModel.predict((left2rightIndecies, right2leftIndecies,left2right_pos_te, left2right_tag_te, left2right_dep_te, right2left_pos_te, right2left_tag_te, right2left_dep_te ))
print("Prediction in Spacy lstm", predict_class)

