import albumentations as alb
import pandas as pd
from pytube import YouTube
import pathlib
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
import cv2
import numpy as np
import time
import random

class Utils():
    def __init__(self, labels):
        self.transform_pipeline = alb.Compose([
            alb.Resize(64, 64, always_apply=True),
            alb.CenterCrop(64, 64, always_apply=True), #Pytorch optimal results
            alb.Normalize(mean = [0.43216, 0.394666, 0.37645],
                            std = [0.22803, 0.22145, 0.216989],
                            always_apply=True) #Dataset expected normalization
        ])
        self.iteration = 1

        self.train_amount = 100
        self.test_amount = 50
        self.validate_amount = 20

        with open(labels, 'r') as file:
            self.class_names = file.readlines()
            file.close()

    def get_video_number(self, path):
        file_count = 0
        for p in pathlib.Path(path).iterdir():
            if p.is_file():
                file_count += 1
        return file_count

    def cut_video(self, path, start, end, final_path):
        ffmpeg_extract_subclip(path, start, end, targetname=final_path)

    def download_video(self, data, destiny_path, dataset_type):
        print('Video number ' + str(self.iteration))
        downloaded = True
        try:
            final_path = destiny_path + 'dataset/' + str(dataset_type) + '/' + data['label'].replace(' ', '') + '/'
            save_path = final_path + data['label'].replace(' ', '') + '_'
            video_number = self.get_video_number(final_path)

            yt = YouTube("https://www.youtube.com/watch?v="+data['youtube_id'])
            yt.streams.filter(progressive=True,
                file_extension='mp4').order_by('resolution').desc().last().download(output_path=final_path,
                filename=data['label'].replace(' ', '') + '_' + str(video_number) + '.mp4')

            self.cut_video(save_path + str(video_number) + '.mp4', data['time_start'], data['time_end'], final_path + str(video_number) + '.mp4')
            os.remove(save_path + str(video_number) + '.mp4')
        except Exception as error:
            print(str(error))
            downloaded = False
        self.iteration += 1
        return downloaded

    def sample_videos(self, dataset_type, dataset):
        number_of_samples = 0
        if 'train' in dataset_type:
            number_of_samples = self.train_amount
        elif 'test' in dataset_type:
            number_of_samples = self.test_amount
        else:
            number_of_samples = self.validate_amount

        return dataset.groupby('label').sample(n=number_of_samples)

    def create_dataset(self, path, destiny_path, dataset_type, classes):
        dataset = pd.read_csv(path)
        dataset = dataset.loc[dataset.apply(lambda x: x.label.replace(' ', '') in classes, axis=1)]
        dataset = self.sample_videos(dataset_type, dataset)
        dataset['downloaded'] = dataset.apply(self.download_video, args=[destiny_path, dataset_type], axis=1)
        dataset.to_csv(destiny_path + dataset_type + '.csv')

    def extract_frames(self, video_path):
        frames_list = []
        image_height, image_width = 64, 64

        video_reader = cv2.VideoCapture(video_path)
        while True:
            ret, frame = video_reader.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (image_height, image_width))
            normalized_frame = resized_frame / 255
            frames_list.append(normalized_frame)

        if len(frames_list) == 0:
            print('It was not possible to extract frames.')
        video_reader.release()
        return frames_list

    def create_category_dict(self, frames, categories, category):
        dic = {'x': []}
        for cat in categories:
            dic[cat] = []

        for frame in frames:
            dic['x'].append(frame)
            for cat in categories:
                if cat == category:
                    dic[cat].append(1)
                else:
                    dic[cat].append(0)
        return dic

    def create_frames_dataset(self, path, classes, dataset_type):
        dataframe = pd.DataFrame(columns=['x'] + classes)
        for category in classes:
            print('Extracting data from class: ' + category)
            data_path = path + category + '/'
            videos_list = os.listdir(data_path)

            all_frames = []

            for file in videos_list:
                print('File: ' + file)
                video_path = data_path + file
                frames = self.extract_frames(video_path)
                all_frames.extend(frames)

            all_frames = np.asarray(random.sample(all_frames, k=1000))
            data = pd.DataFrame.from_dict(self.create_category_dict(all_frames, classes, category))
            dataframe = pd.concat([dataframe, data], ignore_index=True)
            print('Dataframe updated, new len: ' + str(len(dataframe)))
            time.sleep(5)
        dataframe.to_csv(dataset_type + '.csv')
