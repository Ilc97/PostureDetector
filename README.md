# PostureDetector

A program that detects a persons sitting posture as bad or good. In case of bad posture it alarms the person with a sound beep.

## Preview

## Requirements
[detectron 2](https://github.com/facebookresearch/detectron2) <br>
[openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

## Usage

Run the command:

```
python postureDetector.py [-h] [--sound] [--no-plot] [--log LOG]

Posture Detection Program

optional arguments:
  -h, --help  show this help message and exit
  --sound     Enable sound feedback
  --no-plot   Disable plotting
  --log LOG   Specify the log file name            


