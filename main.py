import datamodules

#load data
data_train, X_train, Y_train = datamodules.loadXML(
    "./data/Restaurants_Train.xml",
    polarities=['negative', 'neutral', 'positive'])

data_test, X_test, Y_test = datamodules.loadXML(
    "./data/Restaurants_Test_Gold.xml",
    polarities=['negative', 'neutral', 'positive'])

#convert sentence to index
words_to_index, X_train_indices, X_test_indices, X_train_wl, X_train_wr, X_test_wl, X_test_wr = datamodules.word2index(
    X_train, X_test, data_train, data_test)

#load global vectors
gloVec = datamodules.loadGloVec("./data/glove.twitter.27B.200d.txt")
embed_vector_len = gloVec['the'].shape[0]
