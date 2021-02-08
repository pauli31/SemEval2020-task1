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

    # ed
    reverse_emb = False
    use_nearest_neigbh = True
    use_bin_thld = False
    emb_type = 'w2v'
    emb_dim = 25
    window = 5
    iter = 5

    # Set both to False to reproduce results stated in the papers
    mean_centering = False
    unit_vectors = False

    acc_list = []
    rho_list = []

    # #
    acc, rho = run_english_default(task_1_dir, task_2_dir, reverse_emb, use_bin_thld, use_nearest_neigbh, emb_type, emb_dim, window, iter, mean_centering, unit_vectors)
    acc_list.append(acc)
    rho_list.append(rho)

    acc, rho = run_german_default(task_1_dir, task_2_dir, reverse_emb, use_bin_thld, use_nearest_neigbh, emb_type, emb_dim, window, iter, mean_centering, unit_vectors)
    acc_list.append(acc)
    rho_list.append(rho)

    acc, rho = run_latin_default(task_1_dir, task_2_dir, reverse_emb, use_bin_thld, use_nearest_neigbh, emb_type, emb_dim, window, iter, mean_centering, unit_vectors)
    acc_list.append(acc)
    rho_list.append(rho)

    acc, rho = run_swedish_default(task_1_dir, task_2_dir, reverse_emb, use_bin_thld, use_nearest_neigbh, emb_type, emb_dim, window, iter, mean_centering, unit_vectors)
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
                        emb_type, emb_dim, window, iter, mean_centering, unit_vectors):
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
    rho, acc, bin_thld, min_neighb_cnt = compare(corp_1_emb, corp_2_emb, target_words, SWEDISH_TEST_GOLD_TASK_1,
                                                 SWEDISH_TEST_GOLD_TASK_2, reverse_emb, use_bin_thld, use_nearest_neigbh,
                                                 mean_centering, unit_vectors, save_file_ranks=save_file,
                                                 save_file_binary=save_file_binary, one_minus=True)

    print('Config:')
    print(str(reverse_emb) + '\t' + 'swedish' + '\t' + corp1_emb_file + '\t' + corp2_emb_file + '\t' + str(min_neighb_cnt) +'\t' + str(bin_thld) +
          '\t' + str(acc) + '\t' + str(rho))

    return acc, rho

def run_latin_default(task_1_dir, task_2_dir, reverse_emb, use_bin_thld, use_nearest_neigbh,
                      emb_type, emb_dim, window, iter, mean_centering, unit_vectors):
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
    rho, acc, bin_thld, min_neighb_cnt = compare(corp_1_emb, corp_2_emb, target_words, LATIN_TEST_GOLD_TASK_1,
                                                 LATIN_TEST_GOLD_TASK_2, reverse_emb, use_bin_thld, use_nearest_neigbh,
                                                 mean_centering, unit_vectors, save_file_ranks=save_file,
                                                 save_file_binary=save_file_binary, one_minus=True)

    print('Config:')
    print(str(reverse_emb) + '\t' + 'latin' + '\t' + corp1_emb_file + '\t' + corp2_emb_file + '\t' + str(
        min_neighb_cnt) + '\t' + str(bin_thld) +
          '\t' + str(acc) + '\t' + str(rho))

    return acc, rho


def run_german_default(task_1_dir, task_2_dir, reverse_emb, use_bin_thld, use_nearest_neigbh,
                       emb_type, emb_dim, window, iter, mean_centering, unit_vectors):
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
    rho, acc, bin_thld, min_neighb_cnt = compare(corp_1_emb, corp_2_emb, target_words, GERMAN_TEST_GOLD_TASK_1,
                                                 GERMAN_TEST_GOLD_TASK_2, reverse_emb, use_bin_thld, use_nearest_neigbh,
                                                 mean_centering, unit_vectors, save_file_ranks=save_file,
                                                 save_file_binary=save_file_binary, one_minus=True)

    print('Config:')
    print(str(reverse_emb) + '\t' + 'german' + '\t' + corp1_emb_file + '\t' + corp2_emb_file + '\t' + str(
        min_neighb_cnt) + '\t' + str(bin_thld) +
          '\t' + str(acc) + '\t' + str(rho))

    return acc, rho


def run_english_default(task_1_dir, task_2_dir, reverse_emb, use_bin_thld, use_nearest_neigbh,
                        emb_type, emb_dim, window, iter, mean_centering, unit_vectors):
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
    rho, acc, bin_thld, min_neighb_cnt = compare(corp_1_emb, corp_2_emb, target_words, ENGLISH_TEST_GOLD_TASK_1,
                                                 ENGLISH_TEST_GOLD_TASK_2, reverse_emb, use_bin_thld, use_nearest_neigbh,
                                                 mean_centering, unit_vectors, save_file_ranks=save_file,
                                                 save_file_binary=save_file_binary, one_minus=True)

    print('Config:')
    print(str(reverse_emb) + '\t' + 'english' + '\t' + corp1_emb_file + '\t' + corp2_emb_file + '\t' + str(
        min_neighb_cnt) + '\t' + str(bin_thld) +
          '\t' + str(acc) + '\t' + str(rho))

    return acc, rho


# TODO dokumentace compare metody
def compare(src_emb_path, trg_emb_path, target_words_path, gold_file_task1, gold_file_task2, reverse, use_binary_threshold,
            use_nearest_neigbhrs, mean_centering, unit_vectors,
            xform=None, max_links=100000, run_spearman=True, save_file_ranks=None, save_file_binary=None,
            one_minus=False, topn=100):
    """
    The function perform comparison for a two given word embeddings (representing two data/text corpora)
    a targets words and it decides how much the target words changed their meaning between the two word embeddings (semantic spaces)

    :param src_emb_path: path to the source word embeddings, must be in the well known word2vec format, i.e., first line contains
                        number of words and dimension of the embeddings, then each line start word and its corresponding
                        word vector
    :param trg_emb_path: path to the target word embeddings, must be in the well known word2vec format, same as for src_emb_path
    :param target_words_path: path to list of target words, one per line
    :param gold_file_task1: path to file with gold data for task 1 (i.e. binary classification task)
                            it is NON-mandatory parameter, it can be set None, must be set only if run_spearman=True
                            if you not use the function for other data than SemEval set it to None
    :param gold_file_task2: path to file with gold data for task 2 (i.e. ranking task), NON-mandatory parameter, same as
                            gold_file_task1
                            if you not use the function for other data than SemEval set it to None
    :param reverse: if set to True the source embeddings is swaped with the target, this should not significantly affect
                    the results
    :param use_binary_threshold: strategy for binary classification, if set to True the parameter use_nearest_neigbhrs
                                must be set to False.
                                This strategy takes all cosine similarities for all words and it computes the average,
                                the average is then used as a threshold. This strategy assumes that roughly half of the
                                target words changed their meaning and half did not.
                                If you plan to use it for other data than SemEval we suggest to use the continous score,
                                or set the threshold manually.
    :param use_nearest_neigbhrs: strategy for binary classification, if set to True the parameter use_binary_threshold
                                must be set to False.


    :param mean_centering: if true word vectors are centered around zero
    :param unit_vectors: if true word vectors are converted to unit vectors
    :param xform: path to transformation matrix, set to None to perform fresh transformation, which is recommended
    :param max_links: maximum number of links (size of vocabulary) that are used for
    :param run_spearman: set to True only if data for SemEval are used, it calculates the competitions scores, i.e.,
                        Spearman score and accuracy for the binary task
    :param save_file_ranks: file where results for ranks will be written, the continuous scores
    :param save_file_binary: file where results for binary output will be written, the binary outputs
    :param one_minus:
    :param topn: number of top n most similar words for the binary strategies, keep

    :return: spearman score, accuracy, the used binary threshold if use_binary_threshold is used, use_binary_threshold if
             the use_nearest_neigbhrs strategy is used
    """

    # reversing
    if reverse is True:
        tmp_path = src_emb_path
        src_emb_path = trg_emb_path
        trg_emb_path = tmp_path

    print("Running comparison for topn:" + str(topn) + " min_neighbours_count:" + str(use_nearest_neigbhrs) +" use binary threshold:" + str(use_binary_threshold))

    # load embeddings and target words
    src_emb, trg_emb = load_word_vectors(src_emb_path, trg_emb_path)
    if mean_centering is True or unit_vectors is True:
        src_emb.vectors = normalize(src_emb.vectors, mean_centering, unit_vectors)
        trg_emb.vectors = normalize(trg_emb.vectors, mean_centering, unit_vectors)

    print("Src emb size:" + str(len(src_emb.vocab)))
    print("Trg emb size:" + str(len(trg_emb.vocab)))
    target_words_dict, target_words = load_target_words(target_words_path, load_labels=False)

    run_transform = False
    if xform is None:
        # transformation matrix
        xform = os.path.basename(src_emb_path) + '-' + os.path.basename(trg_emb_path) + '.xform'
        xform = os.path.join(TMP_DIR, xform)
        run_transform = True

        if os.path.exists(xform):
            try:
                os.remove(xform)
            except Exception as e:
                print("error when deleting xform:" + str(xform))
                print("error:" + str(e))

    # file with results
    output_file = os.path.basename(src_emb_path) + '-' + os.path.basename(trg_emb_path) + '.ranks'
    output_file = os.path.join(TMP_DIR, output_file)

    output_file_binary = os.path.basename(src_emb_path) + '-' + os.path.basename(trg_emb_path) + '.binary'
    output_file_binary = os.path.join(TMP_DIR, output_file_binary)


    trans_dict_path = os.path.join(TMP_DIR, 'trans.dict')
    build_transform_dict(src_emb,trg_emb,trans_dict_path, target_words_dict)

    if run_transform is True:
        import ccaxform1 as ccx  # this could be closer to other imports ...

        # get the transform
        trans_matrix = ccx.transform_KV(src_emb, trg_emb, target_words_dict, max_links=max_links)

        # could write out the xform file now, if we wanted
        np.savetxt(xform, trans_matrix, fmt="%.6f")
    else:
        # run_transform is false.  pick up xform from file, which
        # we have already checked for existence
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

def normalize(X, mean_centering=True, unit_vectors=True):
    """
    Normalize given ndarray

    :param X: ndarray representing semantic space,
                axis 0 (rows)       - vectors for words
                axis 1 (columns)    - elements of word vectors
    :param mean_centering: if true values are centered around zero
    :param unit_vectors: if true vectors are converted to unit vectors

    :return: normalized ndarray
    """
    if mean_centering is True:
        # mean vector and normalization
        mean = X.sum(0) / X.shape[0]
        X = X - mean

    if unit_vectors is True:
        # compute norm
        # norms = np.sqrt((X ** 2).sum(-1))[..., np.newaxis]
        norms = np.linalg.norm(X, axis=1).reshape((X.shape[0], 1))
        X = X / norms
    return X