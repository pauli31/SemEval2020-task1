import os
import pathlib
import shutil
import sys

import numpy as np
import scipy.stats as ss
from gensim.models import KeyedVectors

from config import EMBEDDINGS_EXPORT_PATH, TMP_DIR, ENGLISH_TEST_TARGET_WORDS, TEST_DATA_RESULTS_DIR, \
    GERMAN_TEST_TARGET_WORDS, LATIN_TEST_TARGET_WORDS, SWEDISH_TEST_TARGET_WORDS, TEST_DATA_TRUTH_ANSWER_TASK_1, \
    TEST_DATA_TRUTH_ANSWER_TASK_2, SWEDISH_TEST_GOLD_TASK_1, SWEDISH_TEST_GOLD_TASK_2, LATIN_TEST_GOLD_TASK_1, \
    LATIN_TEST_GOLD_TASK_2, GERMAN_TEST_GOLD_TASK_1, GERMAN_TEST_GOLD_TASK_2, ENGLISH_TEST_GOLD_TASK_2, \
    ENGLISH_TEST_GOLD_TASK_1
from data.post_eval_data.scoring_program.evaluation_official import spearman_official, accuracy_official
from sense_comparator import load_transform_matrix, compare_sense


def main():
    general_folder = 'post-test'
    task_1_dir, task_2_dir, folder_to_zip, zip_file = init_folders(general_folder)

    reverse_emb = True
    use_nearest_neigbh = False
    use_bin_thld = True
    emb_type = 'w2v'
    emb_dim = 100
    window = 5
    iter = 5

    acc_list = []
    rho_list = []

    # #
    acc, rho = run_english_default(task_1_dir, task_2_dir, reverse_emb, use_bin_thld, use_nearest_neigbh, emb_type, emb_dim, window, iter)
    acc_list.append(acc)
    rho_list.append(rho)

    acc, rho = run_german_default(task_1_dir, task_2_dir, reverse_emb, use_bin_thld, use_nearest_neigbh, emb_type, emb_dim, window, iter)
    acc_list.append(acc)
    rho_list.append(rho)

    acc, rho = run_latin_default(task_1_dir, task_2_dir, reverse_emb, use_bin_thld, use_nearest_neigbh, emb_type, emb_dim, window, iter)
    acc_list.append(acc)
    rho_list.append(rho)

    acc, rho = run_swedish_default(task_1_dir, task_2_dir, reverse_emb, use_bin_thld, use_nearest_neigbh, emb_type, emb_dim, window, iter)
    acc_list.append(acc)
    rho_list.append(rho)

    acc_avg = round(np.mean(acc_list), 3)
    rho_avg = round(np.mean(rho_list), 3)

    print('Type' + '\t' + 'avg acc/rank' + '\t' + 'english' + '\t' + 'german' + '\t' + 'latin'+ '\t' + 'swedish' + '\t' + 'reverse emb'
          + '\t' + 'emb_type' + '\t' + 'emb_dim' + '\t' + 'window' + '\t' + 'iter' + '\t' + 'use bin thld' + '\t' + 'use nearest neigh')
    print("Binary overview" + '\t' + str(acc_avg) +
          '\t' + str(acc_list[0]) + '\t' + str(acc_list[1]) + '\t' + str(acc_list[2]) + '\t' + str(acc_list[3]) + '\t'
          + str(reverse_emb) + '\t' + emb_type + '\t' + str(emb_dim) + '\t' + str(window) + '\t' + str(iter)
          + '\t' + str(use_bin_thld) + '\t' + str(use_nearest_neigbh))


    print('Rank overview' + '\t' + str(rho_avg) +
          '\t' + str(rho_list[0]) + '\t' + str(rho_list[1]) + '\t' + str(rho_list[2]) + '\t' + str(rho_list[3]) + '\t'
          + str(reverse_emb) + '\t' + emb_type + '\t' + str(emb_dim) + '\t' + str(window) + '\t' + str(iter))


    # zip_folder(folder_to_zip, zip_file)
    #
    # compute_spearman_between_res()
    # evaluate_submission_results()


def evaluate_submission_results():
    submissions = ['default', 'default_binary_threshold', 'default_reveresed_binary_threshold', 'default_reversed',
                   'LDA-100', 'LDA-100-globalThreshold', 'map-ort-i', 'map-ort-i-globalThreshold', 'map-unsup', 'map-unsup-globalThreshold']
    languages = ['english', 'german', 'latin', 'swedish']
    for sub in submissions:
        print('-' * 70)
        print('-' * 70)
        print('-' * 70)
        print('Evalaluating submission named:', sub)
        for lang in languages:
            print('-' * 50)
            print('Lang:' + lang)
            binary_gold_file = os.path.join(TEST_DATA_TRUTH_ANSWER_TASK_1, lang + '.txt')
            binary_pred_file = os.path.join(TEST_DATA_RESULTS_DIR, sub, 'answer','task1', lang + '.txt')

            rank_gold_file = os.path.join(TEST_DATA_TRUTH_ANSWER_TASK_2, lang + '.txt')
            rank_pred_file = os.path.join(TEST_DATA_RESULTS_DIR, sub, 'answer', 'task2', lang + '.txt')
            my_rho, my_pval = compute_spearman(rank_gold_file, rank_pred_file, print_res=False)
            print('My results: Rho:' + str(my_rho) + ' p-value:' + str(my_pval))

            off_rho, off_pval = spearman_official(rank_gold_file, rank_pred_file)
            print('Official results: Rho:' + str(off_rho) + ' p-value:' + str(off_pval))

            acc_official = accuracy_official(binary_gold_file, binary_pred_file)
            print('Official accuracy:' + str(acc_official))



def run_swedish_default(task_1_dir, task_2_dir, reverse_emb, use_bin_thld, use_nearest_neigbh,
                        emb_type, emb_dim, window, iter):
    print('-' * 70)
    print('Swedish')
    save_file = os.path.join(task_2_dir, 'swedish.txt')
    save_file_binary = os.path.join(task_1_dir, 'swedish.txt')

    # config
    # corp1_emb_file = 'w2v.swedish_corpus1.100_window-5_iter-5_min-count-5.vec'
    # corp2_emb_file = 'w2v.swedish_corpus2.100_window-5_iter-5_min-count-5.vec'
    corp1_emb_file = emb_type + '.swedish_corpus1.' + str(emb_dim) + '_window-' + str(window) + '_iter-' + str(
        iter) + '_min-count-5.vec'
    corp2_emb_file = emb_type + '.swedish_corpus2.' + str(emb_dim) + '_window-' + str(window) + '_iter-' + str(
        iter) + '_min-count-5.vec'


    corp_1_emb = os.path.join(EMBEDDINGS_EXPORT_PATH, 'swedish_corpus_1', corp1_emb_file)
    corp_2_emb = os.path.join(EMBEDDINGS_EXPORT_PATH, 'swedish_corpus_2', corp2_emb_file)

    target_words = SWEDISH_TEST_TARGET_WORDS
    rho, acc, bin_thld, min_neighb_cnt = compare(corp_1_emb, corp_2_emb, target_words, SWEDISH_TEST_GOLD_TASK_1, SWEDISH_TEST_GOLD_TASK_2,
                               reverse_emb, use_bin_thld, use_nearest_neigbh, save_file_ranks=save_file, save_file_binary=save_file_binary,
                               one_minus=True)

    print('Config:')
    print(str(reverse_emb) + '\t' + 'swedish' + '\t' + corp1_emb_file + '\t' + corp2_emb_file + '\t' + str(min_neighb_cnt) +'\t' + str(bin_thld) +
          '\t' + str(acc) + '\t' + str(rho))

    return acc, rho

def run_latin_default(task_1_dir, task_2_dir, reverse_emb, use_bin_thld, use_nearest_neigbh,
                      emb_type, emb_dim, window, iter):
    print('-' * 70)
    print('Latin')
    save_file = os.path.join(task_2_dir, 'latin.txt')
    save_file_binary = os.path.join(task_1_dir, 'latin.txt')

    # corp1_emb_file = 'w2v.latin_corpus1.100_window-5_iter-5_min-count-5.vec'
    # corp2_emb_file = 'w2v.latin_corpus2.100_window-5_iter-5_min-count-5.vec'
    corp1_emb_file = emb_type + '.latin_corpus1.' + str(emb_dim) + '_window-' + str(window) + '_iter-' + str(
        iter) + '_min-count-5.vec'
    corp2_emb_file = emb_type + '.latin_corpus2.' + str(emb_dim) + '_window-' + str(window) + '_iter-' + str(
        iter) + '_min-count-5.vec'

    corp_1_emb = os.path.join(EMBEDDINGS_EXPORT_PATH, 'latin_corpus_1', corp1_emb_file)
    corp_2_emb = os.path.join(EMBEDDINGS_EXPORT_PATH, 'latin_corpus_2', corp2_emb_file)

    target_words = LATIN_TEST_TARGET_WORDS
    rho, acc, bin_thld, min_neighb_cnt = compare(corp_1_emb, corp_2_emb, target_words, LATIN_TEST_GOLD_TASK_1, LATIN_TEST_GOLD_TASK_2,
                               reverse_emb, use_bin_thld, use_nearest_neigbh, save_file_ranks=save_file, save_file_binary=save_file_binary,
                               one_minus=True)

    print('Config:')
    print(str(reverse_emb) + '\t' + 'latin' + '\t' + corp1_emb_file + '\t' + corp2_emb_file + '\t' + str(
        min_neighb_cnt) + '\t' + str(bin_thld) +
          '\t' + str(acc) + '\t' + str(rho))

    return acc, rho


def run_german_default(task_1_dir, task_2_dir, reverse_emb, use_bin_thld, use_nearest_neigbh,
                       emb_type, emb_dim, window, iter):
    print('-' * 70)
    print('German')
    save_file = os.path.join(task_2_dir, 'german.txt')
    save_file_binary = os.path.join(task_1_dir, 'german.txt')

    # corp1_emb_file = 'w2v.german_corpus1.100_window-5_iter-5_min-count-5.vec'
    # corp2_emb_file = 'w2v.german_corpus2.100_window-5_iter-5_min-count-5.vec'
    corp1_emb_file = emb_type + '.german_corpus1.' + str(emb_dim) + '_window-' + str(window) + '_iter-' + str(
        iter) + '_min-count-5.vec'
    corp2_emb_file = emb_type + '.german_corpus2.' + str(emb_dim) + '_window-' + str(window) + '_iter-' + str(
        iter) + '_min-count-5.vec'

    corp_1_emb = os.path.join(EMBEDDINGS_EXPORT_PATH, 'german_corpus_1', corp1_emb_file)
    corp_2_emb = os.path.join(EMBEDDINGS_EXPORT_PATH, 'german_corpus_2', corp2_emb_file)

    target_words = GERMAN_TEST_TARGET_WORDS
    rho, acc, bin_thld, min_neighb_cnt = compare(corp_1_emb, corp_2_emb, target_words, GERMAN_TEST_GOLD_TASK_1, GERMAN_TEST_GOLD_TASK_2,
                                 reverse_emb, use_bin_thld, use_nearest_neigbh, save_file_ranks=save_file, save_file_binary=save_file_binary,
                                 one_minus=True)

    print('Config:')
    print(str(reverse_emb) + '\t' + 'german' + '\t' + corp1_emb_file + '\t' + corp2_emb_file + '\t' + str(
        min_neighb_cnt) + '\t' + str(bin_thld) +
          '\t' + str(acc) + '\t' + str(rho))

    return acc, rho


def run_english_default(task_1_dir, task_2_dir, reverse_emb, use_bin_thld, use_nearest_neigbh,
                        emb_type, emb_dim, window, iter):
    print('-' * 70)
    print('English')
    save_file = os.path.join(task_2_dir, 'english.txt')
    save_file_binary = os.path.join(task_1_dir, 'english.txt')

    # corp1_emb_file = 'w2v.english_corpus1.100_window-5_iter-5_min-count-5.vec'
    # corp2_emb_file = 'w2v.english_corpus2.100_window-5_iter-5_min-count-5.vec'
    corp1_emb_file = emb_type + '.english_corpus1.' + str(emb_dim) + '_window-' + str(window) + '_iter-' + str(iter) + '_min-count-5.vec'
    corp2_emb_file = emb_type + '.english_corpus2.' + str(emb_dim) + '_window-' + str(window) + '_iter-' + str(iter) + '_min-count-5.vec'


    corp_1_emb = os.path.join(EMBEDDINGS_EXPORT_PATH, 'english_corpus_1', corp1_emb_file)
    corp_2_emb = os.path.join(EMBEDDINGS_EXPORT_PATH, 'english_corpus_2', corp2_emb_file)

    target_words = ENGLISH_TEST_TARGET_WORDS
    rho, acc, bin_thld, min_neighb_cnt = compare(corp_1_emb, corp_2_emb, target_words, ENGLISH_TEST_GOLD_TASK_1, ENGLISH_TEST_GOLD_TASK_2,
                       reverse_emb, use_bin_thld, use_nearest_neigbh, save_file_ranks=save_file, save_file_binary=save_file_binary,
                        one_minus=True)

    print('Config:')
    print(str(reverse_emb) + '\t' + 'english' + '\t' + corp1_emb_file + '\t' + corp2_emb_file + '\t' + str(
        min_neighb_cnt) + '\t' + str(bin_thld) +
          '\t' + str(acc) + '\t' + str(rho))

    return acc, rho


def compare(src_emb_path, trg_emb_path, target_words_path, gold_file_task1, gold_file_task2, reverse, use_binary_threshold,
            use_nearest_neigbhrs,
            xform=None, max_links=100000, run_spearman=True, save_file_ranks=None, save_file_binary=None,
            one_minus=False, topn=100):
    # delete_tmp_dir()

    # reversing
    if reverse is True:
        tmp_path = src_emb_path
        src_emb_path = trg_emb_path
        trg_emb_path = tmp_path


    print("Running comparison for topn:" + str(topn) + " min_neighbours_count:" + str(use_nearest_neigbhrs) +" use binary threshold:" + str(use_binary_threshold))

    # load embeddings and target words
    src_emb, trg_emb = load_word_vectors(src_emb_path, trg_emb_path)
    target_words_dict, target_words = load_target_words(target_words_path, load_labels=False)

    run_transform = False
    if xform is None:
        # transformation matrix
        xform = os.path.basename(src_emb_path) + '-' + os.path.basename(trg_emb_path) + '.xform'
        xform = os.path.join(TMP_DIR, xform)
        run_transform = True


    # file with results
    output_file = os.path.basename(src_emb_path) + '-' + os.path.basename(trg_emb_path) + '.ranks'
    output_file = os.path.join(TMP_DIR, output_file)

    output_file_binary = os.path.basename(src_emb_path) + '-' + os.path.basename(trg_emb_path) + '.binary'
    output_file_binary = os.path.join(TMP_DIR, output_file_binary)


    trans_dict_path = os.path.join(TMP_DIR, 'trans.dict')
    build_transform_dict(src_emb,trg_emb,trans_dict_path, target_words_dict)

    if run_transform is True:
        if sys.platform == 'linux':
            cpsep = ':'
        else:
            cpsep = ";"
        # but even with cpsep, this command is system-dependent...

        exit_code = os.spawnvp(os.P_WAIT,
                               '/usr/lib/jvm/java-11-openjdk-amd64/bin/java', ['-Xms6000000000', '-cp',
                                                           "CrossLingualSemanticSpaces-0.0.1-SNAPSHOT-jar-with-dependencies.jar" +
                                                           cpsep + "CCAjar.jar", 'clss.CCA',
                                                           src_emb_path, trg_emb_path, trans_dict_path, xform,  str(max_links)])
        if exit_code != 0:
            print('?exit_code from java=', exit_code, file=sys.stderr)
            sys.exit(exit_code)


    trans_matrix = load_transform_matrix(xform)
    # similarities used for generating output file
    rank_similarities = []

    # original similarities
    similarities_unchanged = []
    similarities_to_orig_word = []
    similarities_to_trans_vec = []
    binar_change = []
    neighbrs_inter_sizes = []


    for target_word in target_words:
        # print("Word:" + str(target_word), end='')
        sim, sim_to_orig_word, sim_to_trans_vec = compare_sense(target_word, src_emb, trg_emb, trans_matrix, topn)

        # compute intersection of nearest neigbhrs
        neighbrs_inter_sizes.append(compute_inter_size(sim_to_orig_word, sim_to_trans_vec))

        similarities_unchanged.append(sim)
        similarities_to_orig_word.append(sim_to_orig_word)
        similarities_to_trans_vec.append(sim_to_trans_vec)

        if one_minus is True:
            sim = 1 - sim
        rank_similarities.append(sim)

    binary_threshold = None
    min_neighbours_count = None

    # check for presence of wordsout file, and build it if necessary
    if use_nearest_neigbhrs is True and use_binary_threshold is True:
        raise Exception("I can compute only one at once")

    #     druha nejvetsi hodnota, pokud licha tak +1 a tu vydelim dvema
    # second highest value, if odd then +1 and divide by two
    # default
    # en - 62, I took  31
    # de - 38, I took  19
    # la - 39, I took  18 -- can change resutls
    # swe - 35, I took  17
    # default reversed
    # en - 62, I took 31
    # de - 39, I took 19
    # la - 61, I took  30
    # sw - 41, I took  20

    if use_nearest_neigbhrs is True:
        print("Computing decide_binary_neighbours")
        # if the max number is there two times we still take the second largest value
        set_list = set(neighbrs_inter_sizes)
        set_list.remove(max(set_list))
        second_largest = int(max(set_list)/2)
        min_neighbours_count = second_largest
        print("Second largest is:" + str(second_largest))


    # compute average similarity which will be the threshold
    if use_binary_threshold is True:
        print("Computing binary_threshold")
        avg_sim = np.average(similarities_unchanged)
        avg_sim = round(avg_sim, 3)
        print('similarity average:' + str(avg_sim))
        binary_threshold = avg_sim


    # iterate again over words and compute binary task
    for target_word, sim, sim_to_orig_word, sim_to_trans_vec, nearest_neigbh_size in zip(
            target_words, similarities_unchanged, similarities_to_orig_word, similarities_to_trans_vec, neighbrs_inter_sizes):
        if use_binary_threshold is True:
            binar_change.append(decide_binary_change_threshold(sim, binary_threshold))

        if use_nearest_neigbhrs is True:
            binar_change.append(decide_binary_neighbours(nearest_neigbh_size, min_neighbours_count))


    # write to tmp folder
    with open(output_file, 'w') as f:
        for word, sim in zip(target_words, rank_similarities):
            f.write(word + '\t' + str(sim) + '\n')

    # write binary predictions to tmp folder
    with open(output_file_binary, 'w') as f:
        for word, clazz in zip(target_words, binar_change):
            f.write(word + '\t' + str(clazz) + '\n')

    if save_file_ranks is not None:
        with open(save_file_ranks, 'w') as f:
            for word, sim in zip(target_words, rank_similarities):
                f.write(word + '\t' + str(sim) + '\n')

    # save binary predictions
    if save_file_binary is not None:
        with open(save_file_binary, 'w') as f:
            for word, clazz in zip(target_words, binar_change):
                f.write(word + '\t' + str(clazz) + '\n')

    if run_spearman is True:
        rho, pval = compute_spearman(gold_file_task2, output_file, print_res=False)
        acc = accuracy_official(gold_file_task1, output_file_binary)
        # print("task1 \t task2")
        # print(str(acc), str(rho))

    return round(rho,3), round(acc,3), binary_threshold, min_neighbours_count




def compute_spearman_between_res():
    t = TEST_DATA_RESULTS_DIR
    from os.path import join
    # tasks_paths = [join(t, 'default'), join(t, 'default_reversed'),
    #                join(t, 'default_binary_threshold'), join(t, 'default_reveresed_binary_threshold')]
    tasks_paths = [join(t, 'default'), join(t, 'default_reversed'), join(t, 'LDA-100'), join(t, 'map-ort-i'), join(t, 'map-unsup')]
    tasks_paths = [join(path, 'answer', 'task2') for path in tasks_paths]

    tasks_paths_english = [join(path, 'english.txt') for path in tasks_paths]
    tasks_paths_german = [join(path, 'german.txt') for path in tasks_paths]
    tasks_paths_latin = [join(path, 'latin.txt') for path in tasks_paths]
    tasks_paths_swedish = [join(path, 'swedish.txt') for path in tasks_paths]

    tasks_tuples = [('English', tasks_paths_english), ('German', tasks_paths_german), ('Latin', tasks_paths_latin), ('Swedish', tasks_paths_swedish)]

    for (lang, paths_list) in tasks_tuples:
        print('Computing correlation between our results for ' + lang)
        for base_path in paths_list:
            print('Solution:'+ str(base_path.split('/')[-4]))
            print('#####')
            for tmp_path in paths_list:
                print(str(tmp_path.split('/')[-4]))
                compute_spearman(base_path, tmp_path)
                print('----------------')

            print('-------------------------------')
        print('#################################')
        print('#################################')
        print('#################################')
    pass


def compute_inter_size(sim_to_orig_word, sim_to_trans_vec):
    orig_words = [tup[0] for tup in sim_to_orig_word]
    trans_words = [tup[0] for tup in sim_to_trans_vec]

    orig_words = set(orig_words)
    trans_words = set(trans_words)

    inters = orig_words.intersection(trans_words)
    inter_size = len(inters)
    # print(" inter size:" + str(inter_size))

    return inter_size


def decide_binary_neighbours(nearest_neigbh_size, min_neighbours_count):

    if nearest_neigbh_size >= min_neighbours_count:
        return 0
    else:
        return 1



def decide_binary_change_threshold(similarity, threshold):
    # print(" sim:" + str(similarity))
    if similarity >= threshold:
        return 0
    else:
        return 1


def compute_spearman(file_gold_path, file_pred_path, print_res=True):
    gold_words_dict, _ = load_target_words(file_gold_path)
    pred_words_dict, _ = load_target_words(file_pred_path)

    if(len(gold_words_dict) != len(pred_words_dict)):
        raise Exception("Word dictionaries do not match")

    gold_list = list(gold_words_dict.keys())
    gold_list.sort()
    pred_list = list(pred_words_dict.keys())
    pred_list.sort()

    if len(gold_list) != len(pred_list):
        print(len(pred_list), '!=', len(pred_list))
        raise Exception("Word dictionaries do not match")

    ranks_gold = []
    ranks_pred = []

    for gold, pred in zip(gold_list, pred_list):
        ranks_gold.append(gold_words_dict[gold])
        ranks_pred.append(pred_words_dict[pred])

    rho, pval = ss.spearmanr(ranks_gold, ranks_pred)
    if print_res:
        print('Rho:' + str(rho) + ' p-value:' + str(pval))

    return rho, pval


def delete_tmp_dir():
    tmp_dir = TMP_DIR
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(TMP_DIR)

def build_transform_dict(src_emb, trg_emb, trans_dict_path, target_words_dict):
    src_vocab = set(src_emb.vocab.keys())
    trg_vocab = set(trg_emb.vocab.keys())

    intersection = src_vocab.intersection(trg_vocab)

    with open(trans_dict_path, 'w', encoding='utf-8') as f:
        for word in intersection:
            if not word.strip():
                continue
            # we want exlcude the target words
            if target_words_dict is not None:
                if word in target_words_dict:
                    continue

            f.write(word + '\t' + word + '\n')


def load_target_words(target_words_path, load_labels=True):
    with open(target_words_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    trg_dict = {}
    trg_words = []

    for line in lines:
        word = line.split()[0]
        if load_labels is True:
            label = line.split()[1]
        else:
            label = 0.5
        word = word.strip()
        trg_words.append(word)
        trg_dict[word] = label

    # print('Loaded :' + str(len(trg_words)) + ' target words from file:' + str(target_words_path))

    return trg_dict, trg_words


def load_word_vectors(src_file_path, trg_file_path):
    src_emb = KeyedVectors.load_word2vec_format(src_file_path, binary=False)
    trg_emb = KeyedVectors.load_word2vec_format(trg_file_path, binary=False)

    return src_emb, trg_emb


def init_folders(general_dir):
    task_1 = os.path.join(TEST_DATA_RESULTS_DIR, general_dir, 'answer', 'task1')
    task_2 = os.path.join(TEST_DATA_RESULTS_DIR, general_dir, 'answer', 'task2')

    zip_file = 'UWB_' + general_dir
    zip_file = os.path.join(TEST_DATA_RESULTS_DIR, zip_file)

    folder_to_zip = os.path.join(TEST_DATA_RESULTS_DIR, general_dir)
    pathlib.Path(task_1).mkdir(parents=True, exist_ok=True)
    pathlib.Path(task_2).mkdir(parents=True, exist_ok=True)

    return task_1, task_2, folder_to_zip, zip_file


def zip_folder(folder_to_zip, zip_file):
    shutil.make_archive(zip_file, 'zip', folder_to_zip)