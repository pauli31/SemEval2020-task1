import numpy as np
from gensim.models import KeyedVectors
from numpy.linalg import norm


def load_word_vectors(src_file_path, trg_file_path):
    src_emb = KeyedVectors.load_word2vec_format(src_file_path, binary=False)
    trg_emb = KeyedVectors.load_word2vec_format(trg_file_path, binary=False)

    return src_emb, trg_emb


def load_transform_matrix(transformation_matrix_path):
    return np.loadtxt(transformation_matrix_path)


def compare_sense(sense_word, src_emb, trg_emb, transformation_matrix, topn=10):
    similarity = 0.0

    try:
        src_vec = src_emb[sense_word]
        trg_vec = trg_emb[sense_word]
    except:  # here if key misssing
        print('Not present word:' + sense_word)
        print("Similarity:0.0000")
        print('--------------')
        return similarity, None, None

    transformed_vec = np.dot(src_vec, transformation_matrix)

    # similarity after transformation between the target word in source and target space
    similarity = compute_cosine_sim(transformed_vec, trg_vec)

    # 10 most similar words to sense word in a target space
    most_similar_to_original_word = trg_emb.wv.most_similar(positive=[sense_word], negative=[], topn=topn)
    most_similar_to_original_word.append((sense_word, 1.0))

    # 10 most similar words to a transformed vector (from source space)
    most_similar_to_transformed_vector = trg_emb.wv.similar_by_vector(transformed_vec, topn=topn+1)

    return similarity, most_similar_to_original_word, most_similar_to_transformed_vector


def compute_cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))





