import modelAPI
import pickle
import datamodules
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
scriptpath = os.path.abspath(__file__)
scriptdir = os.path.dirname(scriptpath)
modeldir = os.path.join(scriptdir, 'targetLSTM')
indexdir = os.path.join(scriptdir, 'word2index')

#load a trained modelAPI
targetLSTM = modelAPI.loadModel(modeldir)
file_word2index = open(indexdir, 'rb')
word2index = pickle.load(file_word2index)


#prediction
sentence = "this food is so bad"
left2rightIndecies, right2leftIndecies = datamodules.sentence2index(
    sentence=sentence,
    target=1,
    word2index=word2index,
)
print(left2rightIndecies, right2leftIndecies)
predict_class = targetLSTM.predict((left2rightIndecies, right2leftIndecies))
print(predict_class)

#plot the trained modelAPI accuracy
#modelAPI.plot(targetLSTM, "Target LSTM", "batch_size=64, epochs=10")