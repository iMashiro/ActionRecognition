from src.scripts.utils import Utils

import torch
import torchvision
import cv2
import time
import numpy as np
import mediapipe as mp

class ActionRecognition():
    def __init__(self):
        self.labels_path = 'data/labels.txt'

        self.utils = Utils(self.labels_path)

        #Preparing the model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torchvision.models.video.r3d_18(pretrained=True, progress=True)
        self.model = self.model.eval().to(self.device)

        #Mediapipe configuration
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic

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
                        label = self.utils.class_names[preds].strip()

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

    def process_videos_mediapipe(self, video_path=0):
        cap = cv2.VideoCapture(video_path)
        with self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                # Draw landmark annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                self.mp_drawing.draw_landmarks(
                    image,
                    results.face_landmarks,
                    self.mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles
                    .get_default_pose_landmarks_style())

                cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
        cap.release()

if __name__ == '__main__':
    act_recog = ActionRecognition()

    #act_recog.classify_video_pytorch(video_path='input/archery.mp4', video_length=10)
    act_recog.process_videos_mediapipe(video_path='input/archery.mp4')






