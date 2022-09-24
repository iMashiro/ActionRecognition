import albumentations as alb
import pandas as pd
from pytube import YouTube
import pathlib
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os

class Utils():
    def __init__(self, labels):
        self.transform_pipeline = alb.Compose([
            alb.Resize(128, 171, always_apply=True),
            alb.CenterCrop(112, 112, always_apply=True), #Pytorch optimal results
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


