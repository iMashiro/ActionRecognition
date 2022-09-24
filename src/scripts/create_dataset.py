from src.scripts.utils import Utils
import os
import random

class CreateDataset():
    def __init__(self):
        self.labels_path = '../data/labels.txt'
        self.utils = Utils(self.labels_path)

        self.raw_train_data_path = '../data/kinetics400/train.csv'
        self.raw_test_data_path = '../data/kinetics400/test.csv'
        self.raw_test_validate_path = '../data/kinetics400/validate.csv'
        self.destiny_path = '../data/'

        self.class_names = self.select_classes()

    def create_folders(self):
        os.mkdir(self.destiny_path + 'dataset')
        os.mkdir(self.destiny_path + 'dataset/train')
        os.mkdir(self.destiny_path + 'dataset/test')
        os.mkdir(self.destiny_path + 'dataset/validation')
        vector = []
        for cl in self.class_names:
            os.mkdir(self.destiny_path + 'dataset/train/' + cl.replace('\n', ''))
            os.mkdir(self.destiny_path + 'dataset/test/' + cl.replace('\n', ''))
            os.mkdir(self.destiny_path + 'dataset/validation/' + cl.replace('\n', ''))
            vector.append(cl.replace('\n', ''))
        self.class_names = vector

    def select_classes(self):
        vector = random.sample(self.utils.class_names, k=30)
        return vector

if __name__ == '__main__':
    create_dataset = CreateDataset()
    create_dataset.create_folders()
    create_dataset.utils.create_dataset(create_dataset.raw_train_data_path, create_dataset.destiny_path, 'train', create_dataset.class_names)
    create_dataset.utils.create_dataset(create_dataset.raw_test_data_path, create_dataset.destiny_path, 'test', create_dataset.class_names)
    create_dataset.utils.create_dataset(create_dataset.raw_test_validate_path, create_dataset.destiny_path, 'validation', create_dataset.class_names)
