import mediapipe as mp
import os
import pandas as pd
import cv2
import shutil

class MediaPipeProcess():
    def __init__(self):
        self.train_path = '../data/dataset/train/'
        self.test_path = '../data/dataset/test/'
        self.validation_path = '../data/dataset/validation/'

        self.classes_names = self.get_classes_names(self.train_path)
        self.train_data = self.generate_videos_dataframe(self.train_path)
        self.test_data = self.generate_videos_dataframe(self.test_path)
        self.validation_data = self.generate_videos_dataframe(self.validation_path)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic

    def get_classes_names(self, path):
        classes = os.listdir(path)
        return classes

    def generate_videos_dataframe(self, path):
        folders = os.listdir(path)
        dataframe = pd.DataFrame(columns=['filename', 'class', 'video_path'])
        for folder in folders:
            files = os.listdir(path+folder)
            for file in files:
                new_row = pd.DataFrame.from_dict([{'filename': file,
                                                    'class': folder,
                                                    'video_path': path+folder+'/'+file}])
                dataframe = pd.concat([dataframe, new_row], ignore_index=True)
        return dataframe

    def generate_mediapipe_data(self, path):
        succeed = False
        copied_file_path = path[:-4] + '_copy.mp4'
        print('Copy file: ' + copied_file_path)
        shutil.copyfile(path, copied_file_path)
        cap = cv2.VideoCapture(path)
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (int(cap.get(3)), int(cap.get(4))))
        with self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    break

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
                succeed = True
                out.write(image)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
        cap.release()
        out.release()
        if not succeed:
            shutil.copyfile(copied_file_path, path)

        os.remove(copied_file_path)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    mediapipe_process = MediaPipeProcess()
    mediapipe_process.train_data['video_path'].apply(mediapipe_process.generate_mediapipe_data)
    mediapipe_process.test_data['video_path'].apply(mediapipe_process.generate_mediapipe_data)
    mediapipe_process.validation_data['video_path'].apply(mediapipe_process.generate_mediapipe_data)