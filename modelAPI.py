from datamodules import word2index
import numpy as np
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, Dropout, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages, FigureCanvasPdf

tf.random.set_seed(1234)
adam = Adam(learning_rate=0.001)


def getEmbeddingMatrix(word2index, vocab_len, embed_vector_len, gloVec):
    emb_matrix = np.zeros((vocab_len + 1, embed_vector_len))
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
                emb_matrix[index, :] = average_embedding / total
            else:
                emb_matrix[index, :] = average_embedding
    return emb_matrix


def getEmbedding(vocab_len, embed_vector_len, emb_matrix):
    return Embedding(input_dim=vocab_len + 1,
                     output_dim=embed_vector_len,
                     input_length=69,
                     weights=[emb_matrix],
                     trainable=False)


def getTargetWordEmbedding(word2index, indices, targetwords):
    embedding = []
    for i in range(len(targetwords)):
        row = []
        ap = targetwords[i]
        indice = indices[i]
        for index in indice:
            if index > 0:
                if word2index.get(ap):
                    row.append(word2index.get(ap))
                else:
                    print("not found")
            else:
                row.append(0)
        embedding.append(np.array(row))
    return np.array(embedding)


def vanillaLSTM(embedding, input_shape):
    X_indices = Input(input_shape)
    embeddingLayer = embedding(X_indices)
    X = LSTM(128)(embeddingLayer)
    X = Dropout(0.2)(X)
    X = Dense(3, activation='softmax')(X)
    model = Model(inputs=X_indices, outputs=X)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def loadModel(model_dir):
    return keras.models.load_model(model_dir)


def plot(model, title, para):
    saveDir = "result/" + title + ".pdf"
    m_epoch = 0
    max_accuracy = 0
    max_val_accuracy = 0
    for i in range(len(model.history.history['val_accuracy'])):
        if model.history.history['val_accuracy'][i] > max_val_accuracy:
            m_epoch = i
            max_val_accuracy = model.history.history['val_accuracy'][i]

    max_accuracy = str(round(max_accuracy * 100, 2))
    max_val_accuracy = str(round(max_val_accuracy * 100, 2))

    summary = "Accuracy is " + max_val_accuracy + "% with the following set up: " + para + "."
    fig, (accuracyPlot, lossPlot) = plt.subplots(2)
    fig.suptitle('The accuracy and loss of training data and test data in ' +
                 title,
                 fontsize=8,
                 fontweight='bold')
    accuracyPlot.set_title("Accuracy")
    accuracyPlot.plot(model.history.history['accuracy'],
                      label='Training data accuracy')
    accuracyPlot.plot(model.history.history['val_accuracy'],
                      label='Test data accuracy')
    accuracyPlot.set_xlabel("Epoch \n")
    accuracyPlot.set_ylabel("Accuracy")

    lossPlot.set_title("Loss")
    lossPlot.plot(model.history.history['loss'],
                  label='Training data accuracy')
    lossPlot.plot(model.history.history['val_loss'],
                  label='test data accuracy')
    lossPlot.set_xlabel("Epoch \n\n" + summary)
    lossPlot.set_ylabel("Loss")
    fig.tight_layout()
    fig.savefig(saveDir)
