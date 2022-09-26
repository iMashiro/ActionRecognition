import os
from src.scripts.utils import Utils
import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical, plot_model


class Train():
    def __init__(self):
        self.train_dataset_path = '../data/dataset/train/'
        self.test_dataset_path = '../data/dataset/test/'
        self.validation_dataset_path = '../data/dataset/validation/'
        self.labels_path = '../data/labels.txt'

        self.train_dataframe_path = '/media/hyago/pendrive/train.pkl' # no disk space, used a removible drive to load data.
        self.test_dataframe_path = '/media/hyago/pendrive/test.pkl' # no disk space, used a removible drive to load data.
        self.validation_dataframe_path = '../data/validation.pkl'

        self.train_dataframe = pd.read_pickle(self.train_dataframe_path)
        self.test_dataframe = pd.read_pickle(self.test_dataframe_path)
        #self.validation_dataframe = pd.read_csv(self.validation_dataframe_path)

        self.height = 64
        self.width = 64
        self.classes = 30

        self.classes_list = self.get_classes()

        self.utils = Utils(self.labels_path)

    def get_classes(self):
        columns_list = list(self.train_dataframe.columns)
        columns_list.remove('x')
        return columns_list

    def create_model(self):
        image_height, image_width = self.height, self.width
        model_output_size = self.classes

        model = Sequential()

        model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', input_shape = (image_height, image_width, 3)))
        model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(256, activation = 'relu'))
        model.add(BatchNormalization())
        model.add(Dense(model_output_size, activation = 'softmax'))

        model.summary()

        return model

    def create_dataframes(self):
        self.utils.create_frames_dataset(self.train_dataset_path, self.classes_list, 'train')
        self.utils.create_frames_dataset(self.test_dataset_path, self.classes_list, 'test')
        self.utils.create_frames_dataset(self.validation_dataset_path, self.classes_list, 'validation')

if __name__ == '__main__':
    train_class = Train()
    model = train_class.create_model()

    train_dataset = train_class.train_dataframe
    test_dataset = train_class.test_dataframe

    x = np.array(train_dataset['x'].to_list())
    y = np.array(train_dataset[train_class.classes_list].values.astype(np.float32))

    early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    print('Start Training')
    model.fit(x=x, y=y, epochs=10, batch_size=10, shuffle=True, validation_split=0.2, callbacks=[early_stop], verbose=1)

    x_test = np.array(test_dataset['x'].to_list())
    y_test = np.array(test_dataset[train_class.classes_list].values.astype(np.float32))

    model_evaluation_hist = model.evaluate(x_test, y_test)

    evaluate_loss, evaluate_acc = model_evaluation_hist
    print('Loss: ' + str(evaluate_loss) + ' | ' + 'Accuracy: ' + str(evaluate_acc))
    model.save('model_acc_' + str(evaluate_acc) + '.h5')
