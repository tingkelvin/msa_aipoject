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

# load data
print("Loading training data.")
data_train, X_train, Y_train = datamodules.loadXML(
    trainDataDir,
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
input_length = len(X_train_indices[0])
left2right_length = len(X_train_left2right[0])
right2left_length = len(X_train_right2left[0])

#load global vectors
print("Loading gloVec.")
gloVec = datamodules.loadGloVec(gloVecDir)
embed_vector_len = gloVec['the'].shape[0]
vocab_len = len(word2index)
# glovec_file = open('gloVec', 'wb')
# pickle.dump(gloVec, glovec_file)
# print("gloVec is loaded.")

emb_matrix = modelAPI.getEmbeddingMatrix(word2index, vocab_len,
                                         embed_vector_len, gloVec)

# use pickle to store data for developmemnt process
# word2index_file = open('word2index', 'wb')
# pickle.dump(word2index, word2index_file)

print("Training the model")
# building a vanilla LSTM
# building an embedding layer for every word from our data
# print("Creating embedding layer.")
# emb_matrix = modelAPI.getEmbeddingMatrix(word2index, vocab_len,
#                                          embed_vector_len, gloVec)
# embedding = modelAPI.getEmbedding(vocab_len, embed_vector_len, input_length, emb_matrix)
# print("Embedding layer is created.")
# vanillaLSTM = modelAPI.vanillaLSTM(embedding=embedding, input_shape=((69, )))
# vanillaLSTM.fit(X_train_indices,
#                 Y_train,
#                 batch_size=64,
#                 epochs=10,
#                 validation_data=(X_test_indices, Y_test))
# vanillaLSTM.save('vanillaLSTM')
# modelAPI.plot(vanillaLSTM, "Vanilla LSTM","batch_size=64,epochs=10" )

# building a target LSTM
# print("Creating embedding layer.")
# embedding = modelAPI.getEmbedding(vocab_len, embed_vector_len, right2left_length, emb_matrix)
# print("Embedding layer is created.")
# targetLSTM = modelAPI.targetLSTM(embedding=embedding, input_shape=((61, )))
# targetLSTM.fit([X_train_left2right, X_train_right2left],
#                Y_train,
#                batch_size=64,
#                epochs=10,
#                validation_data=([X_test_left2right,
#                                  X_test_right2left], Y_test))
# targetLSTM.save('targetLSTM')
# print("Training success and model is saved")

#spacyLSTM
train_pos, train_tag, train_dep = datamodules.depedency2seq(
    [text.strip() for text in data_train['text']])
test_pos, test_tag, test_dep = datamodules.depedency2seq(
    [text.strip() for text in data_test['text']])
left2right_pos_tr, left2right_tag_tr, left2right_dep_tr, right2left_pos_tr, right2left_tag_tr, right2left_dep_tr = datamodules.sliceIndex(
    [text.strip() for text in data_train['left2right']], [text.strip() for text in data_train['right2left']], train_pos, train_tag, train_dep)
left2right_pos_te, left2right_tag_te, left2right_dep_te, right2left_pos_te, right2left_tag_te, right2left_dep_te = datamodules.sliceIndex(
    [text.strip() for text in data_test['left2right']], [text.strip() for text in data_test['right2left']], test_pos, test_tag, test_dep)

print("Creating embedding layer.")
embedding = modelAPI.getEmbedding(vocab_len, embed_vector_len, right2left_length, emb_matrix)
print("Embedding layer is created.")

spacyLSTM = modelAPI.spacyLSTM(embedding,(61,), True, False, False)
spacyLSTM.fit([X_train_left2right, X_train_right2left, left2right_pos_tr, right2left_pos_tr, left2right_tag_tr, right2left_tag_tr, left2right_dep_tr, right2left_dep_tr],Y_train, batch_size=64,
               epochs=10,
               validation_data=([X_test_left2right,
                                 X_test_right2left,left2right_pos_te,right2left_pos_te,left2right_tag_te,right2left_tag_te,left2right_dep_te,right2left_dep_te], Y_test))
modelAPI.plot(spacyLSTM, "spacyLSTM with POS", "batch_size=64,epochs=10")
spacyLSTM = modelAPI.spacyLSTM(embedding,(61,), False, True, False)
spacyLSTM.fit([X_train_left2right, X_train_right2left, left2right_pos_tr, right2left_pos_tr, left2right_tag_tr, right2left_tag_tr, left2right_dep_tr, right2left_dep_tr],Y_train, batch_size=64,
               epochs=10,
               validation_data=([X_test_left2right,
                                 X_test_right2left,left2right_pos_te,right2left_pos_te,left2right_tag_te,right2left_tag_te,left2right_dep_te,right2left_dep_te], Y_test))
modelAPI.plot(spacyLSTM, "spacyLSTM with TAG", "batch_size=64,epochs=10")

spacyLSTM = modelAPI.spacyLSTM(embedding,(61,), False, False, True)
spacyLSTM.fit([X_train_left2right, X_train_right2left, left2right_pos_tr, right2left_pos_tr, left2right_tag_tr, right2left_tag_tr, left2right_dep_tr, right2left_dep_tr],Y_train, batch_size=64,
               epochs=10,
               validation_data=([X_test_left2right,
                                 X_test_right2left,left2right_pos_te,right2left_pos_te,left2right_tag_te,right2left_tag_te,left2right_dep_te,right2left_dep_te], Y_test))
modelAPI.plot(spacyLSTM, "spacyLSTM with DEP", "batch_size=64,epochs=10")

spacyLSTM = modelAPI.spacyLSTM(embedding,(61,), True, True, False)
spacyLSTM.fit([X_train_left2right, X_train_right2left, left2right_pos_tr, right2left_pos_tr, left2right_tag_tr, right2left_tag_tr, left2right_dep_tr, right2left_dep_tr],Y_train, batch_size=64,
               epochs=10,
               validation_data=([X_test_left2right,
                                 X_test_right2left,left2right_pos_te,right2left_pos_te,left2right_tag_te,right2left_tag_te,left2right_dep_te,right2left_dep_te], Y_test))
modelAPI.plot(spacyLSTM, "spacyLSTM with POS+Tag", "batch_size=64,epochs=10")
spacyLSTM = modelAPI.spacyLSTM(embedding,(61,), True, False, True)
spacyLSTM.fit([X_train_left2right, X_train_right2left, left2right_pos_tr, right2left_pos_tr, left2right_tag_tr, right2left_tag_tr, left2right_dep_tr, right2left_dep_tr],Y_train, batch_size=64,
               epochs=10,
               validation_data=([X_test_left2right,
                                 X_test_right2left,left2right_pos_te,right2left_pos_te,left2right_tag_te,right2left_tag_te,left2right_dep_te,right2left_dep_te], Y_test))
modelAPI.plot(spacyLSTM, "spacyLSTM with POS+DEP", "batch_size=64,epochs=10")

spacyLSTM = modelAPI.spacyLSTM(embedding,(61,), False, True, True)
spacyLSTM.fit([X_train_left2right, X_train_right2left, left2right_pos_tr, right2left_pos_tr, left2right_tag_tr, right2left_tag_tr, left2right_dep_tr, right2left_dep_tr],Y_train, batch_size=64,
               epochs=10,
               validation_data=([X_test_left2right,
                                 X_test_right2left,left2right_pos_te,right2left_pos_te,left2right_tag_te,right2left_tag_te,left2right_dep_te,right2left_dep_te], Y_test))
modelAPI.plot(spacyLSTM, "spacyLSTM with TAG+DEP", "batch_size=64,epochs=10")

spacyLSTM = modelAPI.spacyLSTM(embedding,(61,), True, True, True)
spacyLSTM.fit([X_train_left2right, X_train_right2left, left2right_pos_tr, right2left_pos_tr, left2right_tag_tr, right2left_tag_tr, left2right_dep_tr, right2left_dep_tr],Y_train, batch_size=64,
               epochs=10,
               validation_data=([X_test_left2right,
                                 X_test_right2left,left2right_pos_te,right2left_pos_te,left2right_tag_te,right2left_tag_te,left2right_dep_te,right2left_dep_te], Y_test))
modelAPI.plot(spacyLSTM, "spacyLSTM with POS+TAG+DEP", "batch_size=64,epochs=10")

# embedding layer for targetwords
# targetwordEmbedding_train = modelAPI.getTargetWordEmbedding(word2index, X_train_indices, data_train['aspectTerm'].tolist())
# targetwordEmbedding_test = modelAPI.getTargetWordEmbedding(word2index, X_test_indices, data_test['aspectTerm'].tolist())
