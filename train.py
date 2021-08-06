from operator import index
import datamodules
import modelAPI
import pickle
import os

scriptpath = os.path.abspath(__file__)
scriptdir = os.path.dirname(scriptpath)

trainDataDir = os.path.join(scriptdir, 'data/Restaurants_Train.xml')
testDataDir = os.path.join(scriptdir, 'data/Restaurants_Test_Gold.xml')
gloVecDir = os.path.join(scriptdir, 'data/glove.twitter.27B.200d.txt')

labels = ['negative', 'neutral', 'positive']
#load data
print("Loading training data.")
data_train, X_train, Y_train = datamodules.loadXML(
    trainDataDir ,
    polarities=labels)

data_test, X_test, Y_test = datamodules.loadXML(
    testDataDir,
    polarities=labels)
print("Training data is loaded.")

# #convert every word to index
print("Indexing training data.")
word2index, X_train_indices, X_test_indices, X_train_left2right, X_train_right2left, X_test_left2right, X_test_right2left = datamodules.word2index(
    X_train, X_test, data_train, data_test)
print("Indexing is done.")

#load global vectors
print("Loading gloVec.")
gloVec = datamodules.loadGloVec(gloVecDir)
embed_vector_len = gloVec['the'].shape[0]
vocab_len = len(word2index)
print("gloVec is loaded.")

#building an embedding layer for every word from our data
print("Creating embedding layer.")
emb_matrix = modelAPI.getEmbeddingMatrix(word2index, vocab_len,
                                         embed_vector_len, gloVec)
embedding = modelAPI.getEmbedding(vocab_len, embed_vector_len, emb_matrix)
print("Embedding layer is created.")
#use pickle to store data for developmemnt process
# word2index_file = open('word2index', 'wb')
# pickle.dump(word2index, word2index_file)

print("Training the model")
# # #building a vanilla LSTM 
# # vanillaLSTM = modelAPI.vanillaLSTM(embedding=embedding, input_shape=((69, )))
# # vanillaLSTM.fit(X_train_indices,
# #                 Y_train,
# #                 batch_size=64,
# #                 epochs=10,
# #                 validation_data=(X_test_indices, Y_test))
# #vanillaLSTM.save('vanillaLSTM')

#building a target LSTM
targetLSTM = modelAPI.targetLSTM(embedding=embedding, input_shape=((61, )))
targetLSTM.fit([X_train_left2right, X_train_right2left],
               Y_train,
               batch_size=64,
               epochs=10,
               validation_data=([X_test_left2right,
                                 X_test_right2left], Y_test))
targetLSTM.save('targetLSTM')
print("Training success and model is saved")

#embedding layer for targetwords
# targetwordEmbedding_train = modelAPI.getTargetWordEmbedding(word2index, X_train_indices, data_train['aspectTerm'].tolist())
# targetwordEmbedding_test = modelAPI.getTargetWordEmbedding(word2index, X_test_indices, data_test['aspectTerm'].tolist())