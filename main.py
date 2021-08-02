from operator import index
import datamodules
import modelAPI
import pickle
#load data
# data_train, X_train, Y_train = datamodules.loadXML(
#     "./data/Restaurants_Train.xml",
#     polarities=['negative', 'neutral', 'positive'])

# data_test, X_test, Y_test = datamodules.loadXML(
#     "./data/Restaurants_Test_Gold.xml",
#     polarities=['negative', 'neutral', 'positive'])

# #convert every word to index
# word2index, X_train_indices, X_test_indices, X_train_left2right, X_train_right2left, X_test_left2right, X_test_right2left = datamodules.word2index(
#     X_train, X_test, data_train, data_test)
file_word2index = open("word2index", 'rb')
word2index = pickle.load(file_word2index)
# print(len(word2index))

#load global vectors
# gloVec = datamodules.loadGloVec("./data/glove.twitter.27B.200d.txt")
# embed_vector_len = gloVec['the'].shape[0]
# vocab_len = len(word2index)

# #building an embedding layer for every word from our data
# emb_matrix = modelAPI.getEmbeddingMatrix(word2index, vocab_len,
#                                          embed_vector_len, gloVec)
# embedding = modelAPI.getEmbedding(vocab_len, embed_vector_len, emb_matrix)

#use pickle to store data for developmemnt process
# word2index_file = open('word2index', 'wb')
# pickle.dump(word2index, word2index_file)

# # #build a vanilla LSTM modelAPI
# # vanillaLSTM = modelAPI.vanillaLSTM(embedding=embedding, input_shape=((69, )))
# # vanillaLSTM.fit(X_train_indices,
# #                 Y_train,
# #                 batch_size=64,
# #                 epochs=10,
# #                 validation_data=(X_test_indices, Y_test))
# #vanillaLSTM.save('vanillaLSTM')

#building a target LSTM
# targetLSTM = modelAPI.targetLSTM(embedding=embedding, input_shape=((61, )))
# targetLSTM.fit([X_train_left2right, X_train_right2left],
#                Y_train,
#                batch_size=64,
#                epochs=10,
#                validation_data=([X_test_left2right,
#                                  X_test_right2left], Y_test))
# targetLSTM.save('targetLSTM')

#load a trained modelAPI
from keras.optimizers import Adam
targetLSTM = modelAPI.loadModel('targetLSTM')

#prediction
sentence = "The food sucks"
left2rightIndecies, right2leftIndecies = datamodules.sentence2index(
    sentence=sentence,
    target=1,
    word2index=word2index,
)
print(left2rightIndecies, right2leftIndecies)
predict_class = targetLSTM.predict((left2rightIndecies, right2leftIndecies))
print(predict_class)

#plot the trained modelAPI accuracy

#modelAPI.plot(vanillaLSTM, "Vanilla LSTM", "batch_size=64, epochs=10")
# modelAPI.plot(targetLSTM, "Target LSTM", "batch_size=64, epochs=10")
#embedding layer for targetwords
# targetwordEmbedding_train = modelAPI.getTargetWordEmbedding(word2index, X_train_indices, data_train['aspectTerm'].tolist())
# targetwordEmbedding_test = modelAPI.getTargetWordEmbedding(word2index, X_test_indices, data_test['aspectTerm'].tolist())