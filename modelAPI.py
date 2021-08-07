import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages, FigureCanvasPdf

tf.random.set_seed(1234)
adam = Adam(learning_rate=0.001)


def getEmbeddingMatrix(word2index, vocab_len, embed_vector_len, gloVec):
    emb_matrix = np.zeros((vocab_len + 2, embed_vector_len))
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
    emb_matrix[-1] = emb_matrix.mean(axis=0)
    return emb_matrix


def getEmbedding(vocab_len, embed_vector_len, input_length, emb_matrix):
    return Embedding(input_dim=vocab_len + 2,
                     output_dim=embed_vector_len,
                     input_length=input_length,
                     weights=[emb_matrix],
                     trainable=False)

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


def targetLSTM(embedding, input_shape):
    left2Right_indices = Input(input_shape)
    right2Left_indices = Input(input_shape)
    left2RightWordEmb = embedding(left2Right_indices)
    right2LeftWordEmb = embedding(right2Left_indices)

    h = LSTM(128)(left2RightWordEmb)
    h = Dropout(0.2)(h)

    h2 = LSTM(128)(right2LeftWordEmb)
    h2 = Dropout(0.2)(h2)

    h2 = tf.concat([h, h2], 1)
    o = Dense(3, activation='softmax')(h2)

    model = Model(inputs=[
        left2Right_indices, right2Left_indices
    ],
                  outputs=o)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def spacyLSTM(embedding, input_shape, pos, tag, dep):
    embedding_pos = Embedding(input_dim = 17, output_dim=9, input_length=61,trainable=False)
    embedding_tag = Embedding(input_dim = 36, output_dim=18, input_length=61,trainable=False)
    embedding_dep = Embedding(input_dim = 45, output_dim=23, input_length=61,trainable=False)

    #word embedding
    left2Right_indices = Input(input_shape)
    right2Left_indices = Input(input_shape)
    left2RightWordEmb = embedding(left2Right_indices)
    right2LeftWordEmb = embedding(right2Left_indices)

    #POS embedding
    left2RightPos_indices = Input(input_shape)
    right2LeftPos_indices = Input(input_shape)
    left2RightPosEmb = embedding_pos(left2RightPos_indices)
    right2LeftPosEmb = embedding_pos(right2LeftPos_indices)

    #TAG embedding
    left2RightTag_indices = Input(input_shape)
    right2LeftTag_indices = Input(input_shape)
    left2RightTagEmb = embedding_tag(left2RightTag_indices)
    right2LeftTagEmb = embedding_tag(right2LeftTag_indices)

    #DEP embedding
    left2RightDep_indices = Input(input_shape)
    right2LeftDep_indices = Input(input_shape)
    left2RightDepEmb = embedding_dep(left2RightDep_indices)
    right2LeftDepEmb = embedding_dep(right2LeftDep_indices)

    if pos:
        left2RightWordEmb = tf.concat([left2RightWordEmb, left2RightPosEmb],2)
        right2LeftWordEmb = tf.concat([right2LeftWordEmb, right2LeftPosEmb],2)
    
    if tag:
        left2RightWordEmb = tf.concat([left2RightWordEmb, left2RightTagEmb],2)
        right2LeftWordEmb = tf.concat([right2LeftWordEmb, right2LeftTagEmb],2)
    
    if dep:
        left2RightWordEmb = tf.concat([left2RightWordEmb, left2RightDepEmb],2)
        right2LeftWordEmb = tf.concat([right2LeftWordEmb, right2LeftDepEmb],2)

    

    h = LSTM(128)(left2RightWordEmb)
    h = Dropout(0.2)(h)
    
    h2 = LSTM(128)(right2LeftWordEmb)
    h2 = Dropout(0.2)(h2)
    
    h2 = tf.concat([h,h2],1)
    o = Dense(3, activation='softmax')(h2)

    model = Model(inputs=[
        left2Right_indices, right2Left_indices, left2RightPos_indices, right2LeftPos_indices, left2RightTag_indices, right2LeftTag_indices, left2RightDep_indices, right2LeftDep_indices
    ],
                  outputs=o)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
    
def loadModel(model_dir):
    return load_model(model_dir)


def plot(model, title, para):
    saveDir = "result/" + title + ".pdf"
    max_accuracy = 0
    max_val_accuracy = 0
    for i in range(len(model.history.history['val_accuracy'])):
        if model.history.history['val_accuracy'][i] > max_val_accuracy:
            max_val_accuracy = model.history.history['val_accuracy'][i]

    max_accuracy = str(round(max_accuracy * 100, 2))
    max_val_accuracy = str(round(max_val_accuracy * 100, 2))

    summary = "Accuracy is " + max_val_accuracy + "% with the following set up: " + para + "."
    fig, (lossPlot, accuracyPlot) = plt.subplots(2)
    fig.suptitle('The accuracy and loss of training data and test data in ' +
                 title,
                 fontsize=10,
                 fontweight='bold')
    lossPlot.set_title("Loss")
    lossPlot.plot(model.history.history['loss'],
                      )
    lossPlot.plot(model.history.history['val_loss'],
                      )
    lossPlot.set_xlabel("Epoch \n" )
    lossPlot.set_ylabel("Loss")
    lossPlot.set_yticks(np.arange(0, 1, 0.1))
    lossPlot.grid()
 
    accuracyPlot.set_title("Accuracy")
    accuracyPlot.plot(model.history.history['accuracy'],
                      label='Training data')
    accuracyPlot.plot(model.history.history['val_accuracy'],
                      label='Test data')
    accuracyPlot.set_xlabel("Epoch \n\n"+ summary)
    accuracyPlot.set_ylabel("Accuracy")
    accuracyPlot.legend()
    accuracyPlot.set_yticks(np.arange(0.5, 1, 0.1))
    accuracyPlot.grid()
    
    fig.tight_layout()
    fig.show()
    fig.savefig(saveDir)    
