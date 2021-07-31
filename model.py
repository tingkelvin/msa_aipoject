import numpy as np
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, Dropout, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt

tf.random.set_seed(1234)
adam = Adam(learning_rate = 0.001)

def getEmbeddingMatrix(word2index,vocab_len,embed_vector_len,gloVec):
    emb_matrix = np.zeros((vocab_len+1, embed_vector_len))
    for word, index in word2index.items():
        split_word = word.split()
        if len(split_word) == 1:
            embedding_vector = gloVec.get(word)
            if embedding_vector is not None:
                emb_matrix[index, :] = embedding_vector
        else:
            average_embedding = np.zeros((1, embed_vector_len))
            total = 0
            for word in split_word:
                embedding_vector = gloVec.get(word)
                if embedding_vector is not None:
                    average_embedding += embedding_vector
                    total += 1
            if total != 0:
                emb_matrix[index, :] = average_embedding/total
            else:
                emb_matrix[index, :] = average_embedding
    return emb_matrix

def getEmbedding(vocab_len, embed_vector_len, emb_matrix):
    return Embedding(input_dim=vocab_len+1, output_dim=embed_vector_len,input_length=69, weights = [emb_matrix], trainable=False)

def vanillaLSTM(embedding, input_shape):
    X_indices = Input(input_shape)
    embeddingLayer = embedding(X_indices)
    X = LSTM(128)(embeddingLayer)
    X = Dropout(0.2)(X)
    X = Dense(3, activation='softmax')(X)
    model = Model(inputs=X_indices, outputs=X)
    return model

def train(model, X_train_indices, Y_train, X_test_indices, Y_test):
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_indices, Y_train, batch_size=64, epochs=10, validation_data=(X_test_indices, Y_test))

def plot(model, title):
    plt.plot(model.history.history['accuracy'])
    plt.plot(model.history.history['val_accuracy'])
    plt.title(title+' accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    plt.title(title+' loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    ##m_epoch = 0
    ##max_accuracy = 0
    ##max_val_accuracy = 0
    ##for i in range(len(model.history.history['val_accuracy'])):
       ## if model.history.history['val_accuracy'][i] > max_val_accuracy:
       ##     m_epoch = i
       ##     max_val_accuracy = model.history.history['val_accuracy'][i]
       ##     max_accuaracy = model.history.history['accuracy'][i]
    ##max_accuracy = round(max_accuracy* 100,2)
    ##max_val_accuracy = round(max_val_accuracy* 100,2)
    ##print(f"Model accuracy maximized at epoch", m_epoch,",", "Test data acccuracy:", max_val_accuracy, "%")