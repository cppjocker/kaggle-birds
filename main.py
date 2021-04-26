import os
import soundfile as sf

import matplotlib.pyplot as plt
import scipy
import scipy.signal
import numpy as np
import pandas as pd

import librosa
import librosa.display

def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()




def main():
    r_path = "."
    path = os.path.join(r_path, "./birdclef-2021/")

    train_labels = pd.read_csv(path + 'train_soundscape_labels.csv')
    train_meta = pd.read_csv(path + 'train_metadata.csv')
    test_data = pd.read_csv(path + 'test.csv')
    samp_subm = pd.read_csv(path + 'sample_submission.csv')


    labels = []
    for row in train_labels.index:
        labels.extend(train_labels.loc[row, 'birds'].split(' '))
    labels = list(set(labels))

    print('Number of unique bird labels:', len(labels))

    df_labels_train = pd.DataFrame(index=train_labels.index, columns=labels)
    for row in train_labels.index:
        birds = train_labels.loc[row, 'birds'].split(' ')
        for bird in birds:
            df_labels_train.loc[row, bird] = 1
    df_labels_train.fillna(0, inplace=True)

    # We set a dummy value for the target label in the test data because we will need for the Data Generator
    test_data['birds'] = 'nocall'

    df_labels_test = pd.DataFrame(index=test_data.index, columns=labels)
    for row in test_data.index:
        birds = test_data.loc[row, 'birds'].split(' ')
        for bird in birds:
            df_labels_test.loc[row, bird] = 1
    df_labels_test.fillna(0, inplace=True)



if __name__ == '__main__':
    main()