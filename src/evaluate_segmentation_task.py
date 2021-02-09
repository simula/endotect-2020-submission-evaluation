
import os
import cv2

import random
import warnings

from tqdm import tqdm
from glob import glob

from sklearn.metrics import jaccard_score, f1_score, recall_score, precision_score

import numpy as np
import pandas as pd


random.seed(0)
np.random.seed(0)

warnings.filterwarnings('ignore')

SUBMISSIONS_DIRECTORY_PATH = ''
GROUND_TRUTH_PATH = ''

RESULTS_DIRECTORY = ''

def calculate_metrics(y_true, y_pred):
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)

    score_jaccard = jaccard_score(y_true.reshape(-1), y_pred.reshape(-1), average='binary')
    score_f1 = f1_score(y_true.reshape(-1), y_pred.reshape(-1), average='binary')
    score_recall = recall_score(y_true.reshape(-1), y_pred.reshape(-1), average='binary')
    score_precision = precision_score(y_true.reshape(-1), y_pred.reshape(-1), average='binary', zero_division=0)

    return [score_jaccard, score_f1, score_recall, score_precision]

def get_filename(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]

def evaluate_submission(submission_path):

    submission_attributes = os.path.basename(submission_path).split('_')
    
    team_name = submission_attributes[1]
    task_name = submission_attributes[2]
    run_id    = os.path.splitext('_'.join(submission_attributes[3:]))[0]

    team_result_path = os.path.join(RESULTS_DIRECTORY, team_name, task_name, run_id)

    if not os.path.exists(team_result_path):
        os.makedirs(team_result_path)
        
    true_masks = sorted(glob(os.path.join(submission_path, "*")))
    pred_masks = sorted(glob(os.path.join(GROUND_TRUTH_PATH, "*")))

    filenames = os.listdir(submission_path)

    metrics_data = []
    SCORE = []

    with open(os.path.join(team_result_path, f'individual_metrics.csv'), 'w') as f:

        f.write(f'filename,jaccard,f1,recall,precision\n')

        for y_true, y_pred in zip(true_masks, pred_masks):
            name = y_true.split("/")[-1]

            assert get_filename(y_true) == get_filename(y_pred)

            y_true = cv2.imread(y_true, cv2.IMREAD_GRAYSCALE)
            y_true = y_true/255.0

            y_pred = cv2.imread(y_pred, cv2.IMREAD_GRAYSCALE)
            y_pred = y_pred/255.0

            score = calculate_metrics(y_true, y_pred)
            SCORE.append(score)
            f.write(f"{name},{score[0]:1.4f},{score[1]:1.4f},{score[2]:1.4f},{score[3]:1.4f}\n")
            print(f"{name} - Jaccard: {score[0]:1.4f} - F1: {score[1]:1.4f} - Recall: {score[2]:1.4f} - Precision: {score[3]:1.4f}")
            metrics_data.append([name, *score])

    SCORE = np.mean(SCORE, axis=0)

    with open(os.path.join(team_result_path, f'average_metrics.csv'), 'w') as f:
        f.write(f'metric,value\n')
        f.write(f'jaccard,{SCORE[0]:0.9f}\n')
        f.write(f'f1,{SCORE[1]:0.9f}\n')
        f.write(f'recall,{SCORE[2]:0.9f}\n')
        f.write(f'precision,{SCORE[3]:0.9f}\n')

    with open(os.path.join(RESULTS_DIRECTORY, f'segmentation_all_average_metrics.csv'), 'a') as f:
        f.write(f'{ team_name },{ task_name },{ run_id }')
        f.write(f',{ SCORE[0] },{ SCORE[1] },{ SCORE[2] },{ SCORE[3] }')
        f.write(f'\n')

if __name__ == "__main__":

    with open(os.path.join(RESULTS_DIRECTORY, f'segmentation_all_average_metrics.csv'), 'w') as f:
        f.write('team-name,task-name,run-id,jaccard,f1,recall,precision\n')

    for submission_path in os.listdir(SUBMISSIONS_DIRECTORY_PATH):

        print(f'Evaluating { submission_path }...')

        evaluate_submission(os.path.join(SUBMISSIONS_DIRECTORY_PATH, submission_path))