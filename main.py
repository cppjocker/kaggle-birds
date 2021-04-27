# %% [code]
import os
import soundfile as sf

import matplotlib.pyplot as plt
import scipy
import scipy.signal
import numpy as np
import pandas as pd

import librosa
import librosa.display

from sklearn.model_selection import train_test_split

from keras.utils import Sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPool1D, BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.applications import VGG19, VGG16, ResNet50


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


class DataGenerator(Sequence):

    def __init__(self, path, list_IDs, data, batch_size, data_length, num_labels):
        self.path = path
        self.list_IDs = list_IDs
        self.data = data
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.list_IDs))
        self.data_length = data_length
        self.num_labels = num_labels
        self.audio_length = 5

    def __len__(self):
        len_ = int(len(self.list_IDs) / self.batch_size)
        if len_ * self.batch_size < len(self.list_IDs):
            len_ += 1
        return len_

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y, spec_size = self.__data_generation(list_IDs_temp)
        # X = X.reshape((self.batch_size, 128, spec_size // 128 // 2))
        return X, y

    def get_spec_size(self, s_len):
        return 2 ** 14

    def get_half_spec_size(self, s_len):
        return self.get_spec_size(s_len) // 2 + 1

    def __data_generation(self, list_IDs_temp):
        spec_size = self.get_spec_size(self.data_length)
        half_spec_size = self.get_half_spec_size(self.data_length)

        X = np.zeros((self.batch_size, half_spec_size, 160))
        y = np.zeros((self.batch_size, self.num_labels))
        last_file = ''

        print(list_IDs_temp)

        for i, ID in enumerate(list_IDs_temp):
            prefix = str(self.data.loc[ID, 'audio_id']) + '_' + self.data.loc[ID, 'site']
            file_list = [s for s in os.listdir(self.path) if prefix in s]
            if len(file_list) == 0:
                # Dummy for missing test audio files
                audio_file_fft = np.zeros((half_spec_size, 160))
            else:
                file = file_list[0]  # [s for s in os.listdir(self.path) if prefix in s][0]
                if file != last_file:
                    audio_file, audio_sr = sf.read(self.path + file)

                last_file = file
                cur_sec = self.data.loc[ID, 'seconds']
                start_5sec = int((cur_sec - self.audio_length) / self.audio_length) * self.data_length
                end_5sec = int(cur_sec / self.audio_length) * self.data_length

                audio_file_chunk = audio_file[start_5sec:end_5sec]
                # _, audio_file_fft = scipy.signal.periodogram(audio_file_chunk, nfft=spec_size)
                _, _, audio_file_fft = scipy.signal.spectrogram(audio_file_chunk, nperseg=1000, noverlap=0,
                                                                nfft=spec_size)

                # audio_file_fft = audio_file_fft[0:-1]
                # scale data
                # audio_file_fft = (audio_file_fft - audio_file_fft.mean()) / audio_file_fft.std()
            X[i,] = audio_file_fft
            y[i,] = self.data.loc[ID, self.data.columns[5:]].values
        return X, y, spec_size


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
    labels = list( sorted( set(labels) ) )

    print(labels)

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

    train_labels = pd.concat([train_labels, df_labels_train], axis=1)
    test_data = pd.concat([test_data, df_labels_test], axis=1)

    train_idxs, val_idxs = train_test_split(list(train_labels.index), test_size=0.33)
    test_idxs = list(samp_subm.index)

    num_labels = len(labels)
    batch_size = 30
    data_length = 160000

    train_generator = DataGenerator(path + 'train_soundscapes/', train_idxs, train_labels, batch_size=batch_size,
                                    data_length=data_length, num_labels=num_labels)
    val_generator = DataGenerator(path + 'train_soundscapes/', val_idxs, train_labels, batch_size=batch_size,
                                  data_length=data_length, num_labels=num_labels)
    test_generator = DataGenerator(path + 'test_soundscapes/', test_idxs, test_data, batch_size=1,
                                   data_length=data_length, num_labels=num_labels)
    if False:
        model = Sequential()
        model.add(Conv1D(64, input_shape=(train_generator.get_half_spec_size(160000), 160), kernel_size=5, strides=4,
                         activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool1D(pool_size=(4)))
        model.add(Conv1D(64, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(num_labels, activation='sigmoid'))

        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['binary_accuracy'])
        model.summary()

        model.fit_generator(generator=train_generator, validation_data=val_generator, epochs=4)
        model.save("/kaggle/working/birds_model.h5")

    model = load_model('birds_model.h5')

    #results = model.evaluate(val_generator, verbose=1)
    #print('test loss, test acc:', results)

    y_pred = model.predict(test_generator, verbose=1)
    y_test = np.where(y_pred > 0.5, 1, 0)

    print(y_test.shape)

    for row in samp_subm.index:
        string = ''
        for col in range(len(y_test[row])):
            if y_test[row][col] == 1:
                print(col, y_pred[row][col], labels[col])
                if string == '':
                    string += labels[col]
                else:
                    string += ' ' + labels[col]
        if string == '':
            string = 'nocall'
        samp_subm.loc[row, 'birds'] = string

    output = samp_subm
    output.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()