#!/usr/bin/env python
import sys
import os
import os.path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
import codecs


# as per the metadata file, input and output directories are the arguments
# [_, input_dir, output_dir] = sys.argv

tasks = ['task1', 'task2']
languages = ['german', 'english', 'latin', 'swedish']
columns_task1 = ['ACC_ALL', 'ACC_GERMAN', 'ACC_ENGLISH', 'ACC_LATIN', 'ACC_SWEDISH']
columns_task2 = ['SPR_ALL', 'SPR_GERMAN', 'SPR_ENGLISH', 'SPR_LATIN', 'SPR_SWEDISH']
language2column_task1 = {('task1','all'):'ACC_ALL',('task1','german'):'ACC_GERMAN',('task1','english'):'ACC_ENGLISH',('task1','latin'):'ACC_LATIN',('task1','swedish'):'ACC_SWEDISH'}
language2column_task2 = {('task2','all'):'SPR_ALL',('task2','german'):'SPR_GERMAN',('task2','english'):'SPR_ENGLISH',('task2','latin'):'SPR_LATIN',('task2','swedish'):'SPR_SWEDISH'}

# Task 1

# accuracies = {}
# for language in languages:
#     # Load submission file
#     submission_file_name = language + '.txt'
#     submission_dir = os.path.join(input_dir, 'res/answer/task1')
#     submission_path = os.path.join(submission_dir, submission_file_name)
#     if not os.path.exists(submission_path):
#         message = "Error: Expected submission file '{0}', found files {1}"
#         sys.exit(message.format(submission_file_name, os.listdir(submission_dir)))
#     with codecs.open(submission_path, 'r', 'utf-8') as submission_file:
#         submission = {line.strip().split('\t')[0]:int(line.strip().split('\t')[1]) for line in submission_file}
#
#     # Load truth file
#     truth_file_name = language + '.txt'
#     truth_dir = os.path.join(input_dir, 'ref/task1')
#     truth_path = os.path.join(truth_dir, truth_file_name)
#     with codecs.open(truth_path, 'r', 'utf-8') as truth_file:
#         truth = {line.strip().split('\t')[0]:int(line.strip().split('\t')[1]) for line in truth_file}
#
#     # Check submission format
#     if set(submission.keys())!=set(truth.keys()) or len(submission.keys())!=len(truth.keys()):
#         message = "Error in '{0}': Submitted targets do not match gold targets."
#         sys.exit(message.format(truth_path))
#
#     if any((not (i==0 or i==1) for i in truth.values())):
#         message = "Error in '{0}': Submitted values contain values that are not equal to 0, 1."
#         sys.exit(message.format(truth_path))
#
#     # Get submitted values and true values
#     submission_values = [submission[target] for target in truth.keys()]
#     truth_values = [truth[target] for target in truth.keys()]
#
#     # Make results
#     acc = accuracy_score(truth_values, submission_values)
#     accuracies[language] = acc
#

def accuracy_official(gold_file, pred_file):
    # Load submission file
    if not os.path.exists(pred_file):
        message = "Error: Expected submission file '{0}', found files {1}"
        sys.exit(1)
    with codecs.open(pred_file, 'r', 'utf-8') as submission_file:
        submission = {line.strip().split('\t')[0]: int(line.strip().split('\t')[1]) for line in submission_file}

    # Load truth file
    with codecs.open(gold_file, 'r', 'utf-8') as truth_file:
        truth = {line.strip().split('\t')[0]:int(line.strip().split('\t')[1]) for line in truth_file}

    # Check submission format
    if set(submission.keys()) != set(truth.keys()) or len(submission.keys()) != len(truth.keys()):
        message = "Error in '{0}': Submitted targets do not match gold targets."
        sys.exit(message.format(gold_file))

    if any((not (i == 0 or i == 1) for i in truth.values())):
        message = "Error in '{0}': Submitted values contain values that are not equal to 0, 1."
        sys.exit(message.format(gold_file))

    # Get submitted values and true values
    submission_values = [submission[target] for target in truth.keys()]
    truth_values = [truth[target] for target in truth.keys()]

    # Make results
    acc = accuracy_score(truth_values, submission_values)

    return acc

def spearman_official(gold_file, pred_file):
    # Load submission file
    with codecs.open(pred_file, 'r', 'utf-8') as submission_file:
        submission = {line.strip().split('\t')[0]:float(line.strip().split('\t')[1]) for line in submission_file}

    # Load truth file
    with codecs.open(gold_file, 'r', 'utf-8') as truth_file:
        truth = {line.strip().split('\t')[0]:float(line.strip().split('\t')[1]) for line in truth_file}

        # Check submission format
    if set(submission.keys()) != set(truth.keys()) or len(submission.keys()) != len(truth.keys()):
        message = "Error in '{0}': Submitted targets do not match gold targets."
        sys.exit(message.format(gold_file))

    # Get submitted values and true values
    submission_values = [submission[target] for target in truth.keys()]
    truth_values = [truth[target] for target in truth.keys()]

    # Make results
    rho, p = spearmanr(submission_values, truth_values, nan_policy='raise')

    return rho, p

    # Task 2
# spearmans = {}
# for language in languages:
#     # Load submission file
#     submission_file_name = language + '.txt'
#     submission_dir = os.path.join(input_dir, 'res/answer/task2')
#     submission_path = os.path.join(submission_dir, submission_file_name)
#     if not os.path.exists(submission_path):
#         message = "Error: Expected submission file '{0}', found files {1}"
#         sys.exit(message.format(submission_file_name, os.listdir(submission_dir)))
#     with codecs.open(submission_path, 'r', 'utf-8') as submission_file:
#         submission = {line.strip().split('\t')[0]:float(line.strip().split('\t')[1]) for line in submission_file}
#
#     # Load truth file
#     truth_file_name = language + '.txt'
#     truth_dir = os.path.join(input_dir, 'ref/task2')
#     truth_path = os.path.join(truth_dir, truth_file_name)
#     with codecs.open(truth_path, 'r', 'utf-8') as truth_file:
#         truth = {line.strip().split('\t')[0]:float(line.strip().split('\t')[1]) for line in truth_file}
#
#     # Check submission format
#     if set(submission.keys())!=set(truth.keys()) or len(submission.keys())!=len(truth.keys()):
#         message = "Error in '{0}': Submitted targets do not match gold targets."
#         sys.exit(message.format(truth_path))
#
#     # Get submitted values and true values
#     submission_values = [submission[target] for target in truth.keys()]
#     truth_values = [truth[target] for target in truth.keys()]
#
#     # Make results
#     rho, p = spearmanr(submission_values, truth_values, nan_policy='raise')
#     spearmans[language] = rho
#
#
# # Make average results
# average_accuracy = np.mean([accuracies[language] for language in accuracies])
# accuracies['all'] = average_accuracy
# average_spearman = np.mean([spearmans[language] for language in spearmans])
# spearmans['all'] = average_spearman
#
# # Write output scores
# with open(os.path.join(output_dir, 'scores.txt'), 'a') as output_file:
#     # Task 1
#     for language in accuracies:
#         column = language2column_task1[('task1',language)]
#         score = accuracies[language]
#         output_file.write("{0}:{1}\n".format(column,score))
#     # Task 2
#     for language in spearmans:
#         column = language2column_task2[('task2',language)]
#         score = spearmans[language]
#         output_file.write("{0}:{1}\n".format(column,score))

