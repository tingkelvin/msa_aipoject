from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import string

targetLSTM = load_model('targetLSTM')
file_word2index = open("word2index", 'rb')
word2index = pickle.load(file_word2index)

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

sentence = "The food is good"
left2rightIndecies, right2leftIndecies = sentence2index(
    sentence=sentence,
    target=1,
    word2index=word2index,
)

predict_class = targetLSTM.predict((left2rightIndecies, right2leftIndecies))
print(predict_class)
