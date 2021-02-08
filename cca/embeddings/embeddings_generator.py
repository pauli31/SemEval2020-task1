import logging
import os
import pathlib

from gensim.models import FastText

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim.models.callbacks import CallbackAny2Vec

from gensim.models import Word2Vec
from config import EMBEDDINGS_EXPORT_PATH, ENGLISH_TEST_CORPUS_1, ENGLISH_TEST_CORPUS_2, \
    GERMAN_TEST_CORPUS_1, GERMAN_TEST_CORPUS_2, SWEDISH_TEST_CORPUS_1, SWEDISH_TEST_CORPUS_2, LATIN_TEST_CORPUS_1, \
    LATIN_TEST_CORPUS_2

# https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
# https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92
# https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
# https://radimrehurek.com/gensim/models/fasttext.html


DIM_SIZE = 300
WINDOW = 5
ITER = 1
NEGATIVE = 5

WORKERS = 7
MIN_COUNT = 5

# METHOD = 'w2v'
METHOD = 'fasttext'

ALGORITHM = 'skipgram'
# ALGORITHM = 'cbow'


# just a hack, not a solution
logging.root.level = logging.ERROR

def main():
    # train_english()
    train_german()
    # train_latin()
    # train_swedish()



def perform_train_and_save(note, dataset_name, sentences, export_dir, path_to_data, use_file=False):
    print('Training using method:' + str(METHOD))

    if METHOD == 'w2v':
        model, name = train_word2vec(sentences, dataset_name, DIM_SIZE, WORKERS, WINDOW, ITER, NEGATIVE,
                                     MIN_COUNT, note=note, algorithm=ALGORITHM)
    elif METHOD == 'fasttext':
        model, name = train_fasttext(sentences, dataset_name, path_to_data, DIM_SIZE, WORKERS, WINDOW, ITER,
                                     NEGATIVE, MIN_COUNT, ALGORITHM, note=note, use_file=use_file)
    else:
        raise Exception('Unknown method')

    print('Model trained')

    save_model(model, name, export_dir, visualize=False)


class MonitorCallback(CallbackAny2Vec):
    def __init__(self, test_words):
        self._test_words = test_words

    def on_epoch_end(self, model):
        try:
            print("Model loss:", model.get_latest_training_loss())  # print loss
        except Exception as e:
            print("Error:" + str(e))

        for word in self._test_words:  # show wv logic changes
            print(model.wv.most_similar(word))
            print('Word: ' + str(word) + ' vec:' + str(model.wv[word]))


def train_fasttext(sentences, dataset_name, path_to_data, dim_size, workers, window, iter,
                   negative, min_count, algorithm, note=None, use_file=False):

    name = generate_file_name('fasttext', dataset_name, dim_size, window, iter, min_count, algorithm, note=note)

    # monitor = MonitorCallback(["Abend", "sein", "und"])  # monitor with demo words
    # monitor = MonitorCallback(["Abend", "sein", "und"])  # monitor with demo words
    print("Training FastText model...")
    print('Name:' + str(name))

    if algorithm == 'cbow':
        sg = 0
    elif algorithm == 'skipgram':
        sg = 1
    else:
        raise Exception("Unknown algorithm:" + str(algorithm))
    if use_file is True:
        model = FastText(corpus_file=path_to_data, min_count=min_count, size=dim_size, workers=workers, window=window, sg=sg, iter=iter,
                         negative=negative, callbacks=[], max_vocab_size=None,)
    else:
        model = FastText(sentences=sentences, min_count=min_count, size=dim_size, workers=workers, window=window, sg=sg,
                         iter=iter, negative=negative, callbacks=[], max_vocab_size=None)

    # model_ft = fasttext.train_unsupervised(path_to_data, minCount=min_count, dim=dim_size, thread=workers, ws=window,
    #                                     model=algorithm, epoch=iter, neg=negative)
    # tmp_path = os.path.join(TMP_DIR, "tmp_" + name)
    # save_fast_text_binary(tmp_path, model_ft)
    # model = KeyedVectors.load_word2vec_format(tmp_path, binary=False)

    print("Model trained")

    # info
    print_info(model)

    # queen king test
    # perform_queen_king_test(model)

    return model, name


def save_fast_text_binary(path, model):
    # get all words from model
    words = model.get_words()

    with open(path, 'w', encoding='utf-8') as file_out:

        # the first line must contain number of total words and vector dimension
        file_out.write(str(len(words)) + " " + str(model.get_dimension()) + "\n")

        # line by line, you append vectors to VEC file
        for w in words:
            v = model.get_word_vector(w)
            vstr = ""
            for vi in v:
                vstr += " " + str(vi)

            file_out.write(w + vstr + '\n')


def train_word2vec(sentences, dataset_name, dim_size, workers, window, iter,
                   negative, min_count, algorithm, note=None):

    name = generate_file_name('w2v', dataset_name, dim_size, window, iter, min_count, algorithm, note=note)

    print("Training Word2Vec model...")
    print('Name:' + str(name))

    if algorithm == 'cbow':
        sg = 0
    elif algorithm == 'skipgram':
        sg = 1
    else:
        raise Exception("Unkown algorithm:" + str(algorithm))

    model = Word2Vec(sentences, min_count=MIN_COUNT, size=dim_size, workers=workers, window=window, sg=sg, iter=iter, negative=negative)
    print("Model trained")

    # info
    print_info(model)

    # queen king test
    # perform_queen_king_test(model)

    return model, name


def save_model(model, name, export_dir, visualize=False):
    # path_bin = os.path.join(export_dir, name + '.bin')
    path_vec = os.path.join(export_dir, name + '.vec')

    # model.wv.save(path_bin)

    # we check that the model was properly saved
    # model = KeyedVectors.load(path_bin)

    # save as txt
    model.wv.save_word2vec_format(path_vec, binary=False)


def generate_file_name(model_name, dataset_name, dim_size, window, iter, min_count, algorithm, note=None):
    str_name = ''
    if note is not None:
        str_name = note + '_'
    str_name += model_name + '.' + str(algorithm) + '.' + dataset_name + '.' + str(dim_size) + '_window-' + str(window) + '_iter-' + str(iter) + "_min-count-" + str(min_count)
    return str_name


def print_info(model):
    # summarize the loaded model
    print("Model:" + str(model))

    # summarize vocabulary
    # print("Vocab:" + str(list(model.wv.vocab)))


def perform_queen_king_test(model):
    try:
        print("Peforming: (king - man) + woman")
        print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))
    except Exception as e:
        print(e)

    print("-------")

    try:
        print("Peforming: (computer - human) + interface")
        print(model.most_similar(positive=['computer', 'human'], negative=['interface'], topn=1))
    except Exception as e:
        print(e)

    try:
        print("Most similar to king:")
        print(model.most_similar('king'))
    except Exception as e:
        print(e)

    try:
        print("Italian:")
        print("Performing: (Re - uomo) + donna")
        print(model.most_similar(positive=['Re', 'uomo'], negative=['donna'], topn=1))

        print("Performing: (re - uomo) + donna")
        print(model.most_similar(positive=['re', 'uomo'], negative=['donna'], topn=1))
    except Exception as e:
        print(e)

    try:
        print("Most similar to Re:")
        print(model.most_similar('Re'))
    except Exception as e:
        print(e)

    try:
        print("Most similar to re:")
        print(model.most_similar('re'))
    except Exception as e:
        print(e)

    try:
        print("Most similar to uomo:")
        print(model.most_similar('uomo'))
    except Exception as e:
        print(e)

    try:
        print("Most similar to donna:")
        print(model.most_similar('donna'))
    except Exception as e:
        print(e)

    print("-------")

    try:
        print("Most similar to Abend:")
        print(model.most_similar('Abend'))
    except Exception as e:
        print(e)

    try:
        print(model.most_similar('abend'))
    except Exception as e:
        print(e)

    print("Most similar to auto:")
    try:
        print(model.most_similar('Auto'))
    except Exception as e:
        print(e)

    try:
        print(model.most_similar('auto'))
    except Exception as e:
        print(e)

    print("Most similar to Apfel:")
    try:
        print(model.most_similar('Apfel'))
    except Exception as e:
        print(e)

    try:
        print(model.most_similar('apfel'))
    except Exception as e:
        print(e)



def load_test_data(path, lower_case=False):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.rstrip() for line in lines]
    if lower_case is True:
        lines = [line.lower() for line in lines]
        path = os.path.splitext(path)[0] + "_lower.txt"
        with open(path, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')

    # make list of lists
    tokenized_sent = [tokenize(line) for line in lines]
    return tokenized_sent, path



def prepare_sentences(data_df, title_col, text_col, use_sentences=False, lower_case=False):
    sentences = []

    # pripadne gensim.utils.simple_preprocess()

    for index, row in data_df.iterrows():
        text = row[text_col]
        title = row[title_col]

        if lower_case is True:
            text = text.lower()
            title = title.lower()

        title = title.replace('\n', ' ')

        if '\n' in title:
            print('Word contain new line:' + str(title))

        if use_sentences is True:
            title_tokens = tokenize(title)
            # append title as one sentence
            sentences.append(title_tokens)

            text_lines = text.split('\n')
            for line in text_lines:
                line_tokens = tokenize(line)
                sentences.append(line_tokens)
        else:
            text = str(title) + ' ' + str(text)
            text = text.replace('\n', ' ')
            text_tokens = tokenize(text)
            sentences.append(text_tokens)

    return sentences


def tokenize(string):
    return string.split(' ')


def train_german(export_dir=EMBEDDINGS_EXPORT_PATH):
    train_german_t1(export_dir)
    train_german_t2(export_dir)

def train_swedish(export_dir=EMBEDDINGS_EXPORT_PATH):
    train_swedish_t1(export_dir)
    train_swedish_t2(export_dir)

def train_latin(export_dir=EMBEDDINGS_EXPORT_PATH):
    train_latin_t1(export_dir)
    train_latin_t2(export_dir)

def train_english(export_dir=EMBEDDINGS_EXPORT_PATH):
    train_english_t1(export_dir)
    train_english_t2(export_dir)


def train_english_t2(export_dir):
    english_t2_corpus_dir = os.path.join(export_dir, 'english_corpus_2')
    pathlib.Path(english_t2_corpus_dir).mkdir(parents=True, exist_ok=True)

    english_t2, path = load_test_data(ENGLISH_TEST_CORPUS_2)
    perform_train_and_save(None, 'english_corpus2', english_t2, english_t2_corpus_dir, path)

def train_english_t1(export_dir):
    english_t1_corpus_dir = os.path.join(export_dir, 'english_corpus_1')
    pathlib.Path(english_t1_corpus_dir).mkdir(parents=True, exist_ok=True)

    english_t1, path = load_test_data(ENGLISH_TEST_CORPUS_1)
    perform_train_and_save(None, 'english_corpus1', english_t1, english_t1_corpus_dir, path)


def train_german_t1(export_dir):
    german_t1_corpus_dir = os.path.join(export_dir, 'german_corpus_1')
    pathlib.Path(german_t1_corpus_dir).mkdir(parents=True, exist_ok=True)

    german_t1, path = load_test_data(GERMAN_TEST_CORPUS_1)
    perform_train_and_save(None, 'german_corpus1', german_t1, german_t1_corpus_dir, path, use_file=True)

    # german_t1_lower, path_lower = load_test_data(GERMAN_TEST_CORPUS_1, lower_case=True)
    # perform_train_and_save('lower', 'german_corpus1', german_t1_lower, german_t1_corpus_dir, path_lower, use_file=True)


def train_german_t2(export_dir):
    german_t2_corpus_dir = os.path.join(export_dir, 'german_corpus_2')
    pathlib.Path(german_t2_corpus_dir).mkdir(parents=True, exist_ok=True)

    german_t2, path = load_test_data(GERMAN_TEST_CORPUS_2)
    perform_train_and_save(None, 'german_corpus2', german_t2, german_t2_corpus_dir, path, use_file=True)

    # german_t2_lower, path_lower = load_test_data(GERMAN_TEST_CORPUS_2, lower_case=True)
    # perform_train_and_save('lower', 'german_corpus2', german_t2_lower, german_t2_corpus_dir, path_lower, use_file=True)



def train_latin_t1(export_dir):
    latin_t1_corpus_dir = os.path.join(export_dir, 'latin_corpus_1')
    pathlib.Path(latin_t1_corpus_dir).mkdir(parents=True, exist_ok=True)

    latin_t1, path = load_test_data(LATIN_TEST_CORPUS_1)
    perform_train_and_save(None, 'latin_corpus1', latin_t1, latin_t1_corpus_dir, path)

    # mala neresim
    # latin_t1_lower, path_lower = load_test_data(LATIN_TEST_CORPUS_1, lower_case=True)
    # perform_train_and_save('lower','latin_corpus1', latin_t1_lower, latin_t1_corpus_dir, path_lower)


def train_latin_t2(export_dir):
    latin_t2_corpus_dir = os.path.join(export_dir, 'latin_corpus_2')
    pathlib.Path(latin_t2_corpus_dir).mkdir(parents=True, exist_ok=True)

    latin_t2, path = load_test_data(LATIN_TEST_CORPUS_2)
    perform_train_and_save(None, 'latin_corpus2', latin_t2, latin_t2_corpus_dir, path)

    # mala neresim
    # latin_t2_lower, path_lower = load_test_data(LATIN_TEST_CORPUS_2, lower_case=True)
    # perform_train_and_save('lower', 'latin_corpus2', latin_t2_lower, latin_t2_corpus_dir, path_lower)


def train_swedish_t1(export_dir):
    swedish_t1_corpus_dir = os.path.join(export_dir, 'swedish_corpus_1')
    pathlib.Path(swedish_t1_corpus_dir).mkdir(parents=True, exist_ok=True)

    swedish_t1, path = load_test_data(SWEDISH_TEST_CORPUS_1)
    perform_train_and_save(None, 'swedish_corpus1', swedish_t1, swedish_t1_corpus_dir, path)



def train_swedish_t2(export_dir):
    swedish_t2_corpus_dir = os.path.join(export_dir, 'swedish_corpus_2')
    pathlib.Path(swedish_t2_corpus_dir).mkdir(parents=True, exist_ok=True)

    swedish_t2, path = load_test_data(SWEDISH_TEST_CORPUS_2)
    perform_train_and_save(None, 'swedish_corpus2', swedish_t2, swedish_t2_corpus_dir, path)


if __name__ == '__main__':
    main()
