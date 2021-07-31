import numpy as np

def getEmbedding(word2index,vocab_len,embed_vector_len,gloVec):
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