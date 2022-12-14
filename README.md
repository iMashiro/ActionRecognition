# ActionRecognition
Code to detect human actions

Link: https://handtalk.notion.site/Reconhecimento-de-A-es-em-V-deo-d634f287e567471fbedcf9bdcf62fb1e

# PhraseClassification
An Action classification solution.

## Instructions:

### Steps to install requirements of the code:

First, create an virtualenv and activate it with the commands:
```
python3 -m venv env
source env/bin/activate
```
Then run the below code to install the src as root folder and install dependencies.
```
pip3 install -e .
```

Finally, you will need to execute the command, this will install the torch in lts version::

```
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cpu
```

## Results and informations

List of actions (total = 30): Tossingsalad, Playingkeyboard, Jugglingballs, Kitesurfing, Wrappingpresent, Ridingmechanicalbull, TappingGuitar, EatingCake, Drinkingshots, Playingcontroller, Dancinggangnamstyle, Kickingfieldgoal, Jogging, Krumping, Playingrecorder, Dodgeball, Shakinghead, Playingtrombone, Trimmingorshavingbeard, Surfingwater, Skijumping, Snatchweightlifting, Canoeingorkayaking, Playingcards, Fingersnapping, Jumpstyledancing, Playingtennis, Climbingladder, Bakingcookies, Gettingatattoo.

The train and test data were generated by selecting 30 random actions from the 400 actions dataset, and reducing the amount of videos for each of them (disk limitations).
For each class, 1000 frames were selected. Totalizing 30000 frames labeled for training.

Due to time and disk space limitations, and since it was my first experience creating a video classification pipeline, I was not able to fully complete the challenge, not being able to classificate a video from webcam and real time, just being able to classificate a video from a file.

The accuracy in training got in 62%, but with test data in avaluation, it just got 20%. With more disk space and time, probably the dataset could be improved for better performance.

To run the classification with the pre trained pytorch model, run the code action_recognition.py with the line 127 uncommented.

To run the classification with the obtained model with the created dataset with pre process with mediapipe, run the code action_recognition.py with the line 128 uncommented.

To change the file, you just need to change the video path in these lines.



