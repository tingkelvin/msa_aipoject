
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import string
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
scriptpath = os.path.abspath(__file__)
scriptdir = os.path.dirname(scriptpath)
modeldir = os.path.join(scriptdir, 'targetLSTM')
indexdir = os.path.join(scriptdir, 'word2index')
file_word2index = open(indexdir , 'rb')
labels = []
model= None
word2index = None
def _initialize():
    global labels, model, word2index
    if not labels:
        model = load_model(modeldir)
        word2index = pickle.load(file_word2index)
        labels = ["negative", "neutral", "postive"]

def sentence2index(sentence, target, word2index):
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    sentence = sentence.split(' ')
    left2rightTemp = sentence[:target]
    right2leftTemp = sentence[target + 1:]
    right2leftTemp = right2leftTemp[::-1]
    left2rightIndecies = []
    right2leftIndecies = []
    print(sentence, left2rightTemp,right2leftTemp)

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

def predict_sentence_from_url(sentence, target):
    _initialize()
    left2rightIndecies, right2leftIndecies = sentence2index(
    sentence=sentence,
    target=int(target),
    word2index=word2index,
    )
    prediction = model.predict((left2rightIndecies, right2leftIndecies))[0]
    result = {'sentiment': labels[np.argmax(prediction)], 'confidence':  str(round(prediction[np.argmax(prediction)]*100,2))}
    print(result)
    return result


