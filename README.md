# PostureDetector

An algorithm that detects a person's sitting posture from a live camera as bad or good. In case of bad posture it alarms the person with a beep sound.

## Preview
![](https://github.com/Ilc97/PostureDetector/blob/master/detector_example.gif)

## Requirements
- usb webcam
- [detectron 2](https://github.com/facebookresearch/detectron2)<br>
- [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

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


