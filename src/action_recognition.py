from src.scripts.utils import Utils

import torch
import torchvision
import cv2
import time
import numpy as np
import mediapipe as mp
import tensorflow as tf

class ActionRecognition():
    def __init__(self):
        self.labels_path = 'data/labels.txt'

        self.utils = Utils(self.labels_path)
        self.class_names = ['tossingsalad', 'playingkeyboard', 'jugglingballs',
                            'kitesurfing', 'wrappingpresent', 'ridingmechanicalbull',
                            'tappingguitar', 'eatingcake', 'drinkingshots', 'playingcontroller',
                            'dancinggangnamstyle', 'kickingfieldgoal', 'jogging', 'krumping',
                            'playingrecorder', 'dodgeball', 'shakinghead', 'playingtrombone',
                            'trimmingorshavingbeard', 'surfingwater', 'skijumping', 'snatchweightlifting',
                            'canoeingorkayaking', 'playingcards', 'fingersnapping', 'jumpstyledancing',
                            'playingtennis', 'climbingladder', 'bakingcookies', 'gettingatattoo']

        #Preparing the model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torchvision.models.video.r3d_18(pretrained=True, progress=True)
        self.model = self.model.eval().to(self.device)

        self.model_trained = tf.keras.models.load_model('scripts/model_acc_0.20.h5')

        #Mediapipe configuration
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic

    def print_output(self, output, desired):
        find_label = 0
        for j in range(0, len(output)):
            max_value = max(output[j])
            max_index = np.where(output[j] == max_value)[0][0]
            if self.class_names[max_index] == desired:
                find_label += 1
        print('Label desired found in ' + str(find_label) + ' of ' + str(len(output)))

    def get_video_dimensions(self, path):
        height, width = -1, -1
        video = cv2.VideoCapture(path)
        if not video.isOpened():
            print('Path error.')
        else:
            height = int(video.get(4))
            width = int(video.get(3))
        return video, height, width

    def create_save_video_object(self, path, height, width):
        obj = cv2.VideoWriter('output/'+ path.split('/')[-1].split('.')[0],
                                cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        return obj

    def classify_video_pytorch(self, video_path, video_length):
        video, height, width = self.get_video_dimensions(video_path)
        video_output = self.create_save_video_object(video_path, height, width)

        total_frames = 0
        total_fps = 0
        clips = []

        while video.isOpened():
            ret, frame = video.read()
            if ret:
                start_time = time.time()
                image = frame.copy()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.utils.transform_pipeline(image=frame)['image']

                clips.append(frame)
                if len(clips) == video_length:
                    with torch.no_grad():
                        input_frames = np.array(clips)
                        input_frames = np.expand_dims(input_frames, axis=0)
                        input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3)) #[1, 3, num_clips, height, width]
                        input_frames = torch.tensor(input_frames, dtype=torch.float32)
                        input_frames = input_frames.to(self.device)
                        outputs = self.model(input_frames)
                        _, preds = torch.max(outputs.data, 1)
                        label = self.utils.class_names[preds]

                    end_time = time.time()
                    fps = 1/(end_time-start_time)
                    total_fps += fps

                    total_frames += 1

                    wait_time = max(1, int(fps/4))
                    cv2.putText(image, label, (15, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 2, lineType=cv2.LINE_AA)

                    clips.pop(0)
                    cv2.imshow('image', image)
                    video_output.write(image)
                    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                        break
            else:
                break

        video.release()
        cv2.destroyAllWindows()

        average_fps = total_fps / total_frames
        print('Average FPS: ' + str(average_fps))

    def classify_video_tensorflow(self, video_path, expected_label):
        video, height, width = self.get_video_dimensions(video_path)

        frames = np.array(self.utils.extract_frames(video_path))
        outputs = self.model_trained.predict(frames)
        self.print_output(outputs, expected_label)

        video.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    act_recog = ActionRecognition()

    #act_recog.classify_video_pytorch(video_path='input/torch/archery.mp4', video_length=10)
    act_recog.classify_video_tensorflow('input/tensorflow/kickingfieldgoal.mp4', 'kickingfieldgoal')






