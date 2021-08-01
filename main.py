import datamodules
import modelAPI

#load data
data_train, X_train, Y_train = datamodules.loadXML(
    "./data/Restaurants_Train.xml",
    polarities=['negative', 'neutral', 'positive'])

data_test, X_test, Y_test = datamodules.loadXML(
    "./data/Restaurants_Test_Gold.xml",
    polarities=['negative', 'neutral', 'positive'])

# #convert every word to index
word2index, X_train_indices, X_test_indices, X_train_wl, X_train_wr, X_test_wl, X_test_wr = datamodules.word2index(
    X_train, X_test, data_train, data_test)

# #load global vectors
gloVec = datamodules.loadGloVec("./data/glove.twitter.27B.200d.txt")
embed_vector_len = gloVec['the'].shape[0]
vocab_len = len(word2index)

# #building an embedding layer for every word from our data
emb_matrix = modelAPI.getEmbeddingMatrix(word2index, vocab_len,
                                         embed_vector_len, gloVec)
embedding = modelAPI.getEmbedding(vocab_len, embed_vector_len, emb_matrix)

# #build a vanilla LSTM modelAPI
vanillaLSTM = modelAPI.vanillaLSTM(embedding=embedding, input_shape=((69, )))
vanillaLSTM.fit(X_train_indices,
                Y_train,
                batch_size=64,
                epochs=10,
                validation_data=(X_test_indices, Y_test))
#vanillaLSTM.save('vanillaLSTM')

#load a trained modelAPI
#reconstructed_model = modelAPI.loadModel('vanillaLSTM')

#plot the trained modelAPI accuracy
modelAPI.plot(vanillaLSTM, "Vanilla LSTM", "batch_size=64, epochs=10")

#embedding layer for targetwords
# targetwordEmbedding_train = modelAPI.getTargetWordEmbedding(word2index, X_train_indices, data_train['aspectTerm'].tolist())
# targetwordEmbedding_test = modelAPI.getTargetWordEmbedding(word2index, X_test_indices, data_test['aspectTerm'].tolist())