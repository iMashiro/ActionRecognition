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

Finally, to run the code with PyTorch, you will need to execute the command, this will install the torch in lts version with cuda if available:

```
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
```

