import cv2
import sys
import logging


sys.path.append("openpose/build/python")
from openpose import pyopenpose as op
from detectron2.utils.logger import setup_logger

setup_logger()
import numpy as np
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_img
import numpy as np
import time
from pygame import mixer
import matplotlib.pyplot as plt
import torch
import argparse


# Initialize the logger
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
# Check the cuDNN version
cudnn_version = torch.backends.cudnn.version()

# Create a figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
fig.suptitle("PostureDetector")
plot_text = fig.text(0, 0, "My text")
fig_text = plt.figtext(
    0.5,
    0.02,
    "",
    ha="center",
    va="center",
    fontsize=12,
    color="red",
    bbox=dict(facecolor="black", alpha=0.5),
)


# Setting OpenPose parameters
def set_params():
    params = dict()
    params["logging_level"] = 3
    params["net_resolution"] = "-320x176"
    params["model_pose"] = "BODY_25"
    params["disable_blending"] = False
    # Ensure you point to the correct path where models are located
    params["model_folder"] = ""
    params["face"] = False
    params["hand"] = False
    params["face_net_resolution"] = (
        "-1x-1"  # You can set the face_net_resolution to disable face detection
    )
    params["hand_net_resolution"] = (
        "-1x-1"  # You can set the hand_net_resolution to disable hand detection
    )
    params["disable_multi_thread"] = False
    params["number_people_max"] = 1
    return params


# Detectron2 package init
def init_detectron2():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    return cfg


# Process cam image
def process_image(loaded_model, openpose, predictor, cfg, args):
    # Reading image from camera
    cam = cv2.VideoCapture(0)  # 0 -> index of camera
    s, img_class = cam.read()
    img_class = cv2.flip(img_class, 1)

    new_width = img_class.shape[1]
    new_height = img_class.shape[0]

    # Create a new resized image using the calculated dimensions
    img = cv2.resize(img_class, (new_width, new_height))

    # Use OpenPose for pose detection
    datum = op.Datum()
    datum.cvInputData = (
        img  # Output keypoints and the image with the human skeleton blended on it
    )
    openpose.emplaceAndPop(op.VectorDatum([datum]))

    # Detectron2 for person segmentation
    outputs = predictor(img)
    pred_classes = outputs["instances"].pred_classes
    person_class_id = 0
    person_instances = outputs["instances"][
        pred_classes == person_class_id
    ]  # Filter instances that belong to the "person" class

    if (
        datum.poseKeypoints is not None
        and len(datum.poseKeypoints) > 0
        and len(person_instances) > 0
    ):
        logging.info("Detected a person.")
        handle_detection(person_instances, cfg, img, datum, loaded_model, cam, args)
    else:
        logging.info("Did not detect a person.")


def handle_detection(person_instances, cfg, img, datum, loaded_model, cam, args):

    # Get the binary mask of the person (use .to("cpu") to convert to CPU if necessary)
    person_mask = person_instances.pred_masks[0].to("cpu").numpy()

    # Create a black background of the same size as the input image
    black_background = np.zeros_like(img)

    # Set the pixels corresponding to the person's mask to white on the black background
    black_background[person_mask] = [255, 255, 255]  # Set to white

    white_pixels = np.argwhere(black_background == 255)
    x_max = 0
    # Find the white pixel with the highest x-coordinate (column index)
    if len(white_pixels) > 0:
        highest_x_pixel = white_pixels[np.argmax(white_pixels[:, 1])]
        # Extract the x and y values
        x_max, y = highest_x_pixel[1], highest_x_pixel[0]

    # Resizing and input preparation
    input_net_img = cv2.resize(black_background, (300, 100))
    X = keras_img.img_to_array(input_net_img)
    X = np.expand_dims(X, axis=0)

    v = Visualizer(
        datum.cvOutputData[:, :, ::-1],
        MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
        scale=1,
        instance_mode=ColorMode.SEGMENTATION,
    )
    out = v.draw_instance_predictions(person_instances.to("cpu"))
    output_img = out.get_image()[:, :, ::-1]
    white_color = [255, 255, 255]
    # Loop through the keypoints and set the corresponding pixels to white
    # 8th point is the waist, and the 16th point is the center of the head
    x1, y1, confidence = datum.poseKeypoints[0][8]  # Extract x, y, and confidence
    x1, y1 = int(x1), int(y1)
    output_img[y1, x1] = white_color
    output_img[y1, x_max] = (
        white_color  # Draw for the most furthest point on the right.
    )

    x2, y2, confidence = datum.poseKeypoints[0][18]  # Extract x, y, and confidence
    x2, y2 = int(x2), int(y2)
    output_img[y2, x2] = white_color
    output_img[y2, x_max] = (
        white_color  # Draw for the most furthest point on the right.
    )

    # Create a list of the points
    # points = [point1, point2, point3, point4]
    points = np.array([(x1, y1), (x_max, y1), (x_max, y2), (x2, y2)], dtype=np.float32)

    # Find the angle of rotation needed to align two specific points (e.g., points 0 and 1) with the x-axis
    angle = np.arctan2(points[3][1] - points[0][1], points[3][0] - points[0][0])
    mask = np.zeros_like(black_background)

    cv2.fillPoly(mask, [points.astype(np.int32)], (255, 255, 255))

    result = cv2.bitwise_and(black_background, mask)

    # Define the center of rotation
    center = (black_background.shape[1] // 2, black_background.shape[0] // 2)

    # Create a rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(angle), scale=1.0)

    # Rotate the result
    rotated_result = cv2.warpAffine(
        result,
        rotation_matrix,
        (black_background.shape[1], black_background.shape[0]),
    )
    rotated_result = cv2.transpose(rotated_result)
    rotated_result = cv2.flip(rotated_result, 0)
    x_min = 0
    white_pixels = np.argwhere(rotated_result == 255)
    if len(white_pixels) > 0:
        highest_x_pixel = white_pixels[np.argmin(white_pixels[:, 1])]

        # Extract the x and y values
        x_min, y = highest_x_pixel[1], highest_x_pixel[0]

    leftmost_point = x_min
    # Calculate the shift amount to move the leftmost point to the center
    image_center = np.array(rotated_result.shape[::-1]) / 2
    shift = image_center - leftmost_point

    # Create a translation matrix to perform the shift
    translation_matrix = np.float32([[1, 0, shift[0]], [0, 1, 0]])

    # Apply the translation to the image
    shifted_image = cv2.warpAffine(
        rotated_result,
        translation_matrix,
        (rotated_result.shape[1], rotated_result.shape[0]),
    )

    white_pixels = np.argwhere(rotated_result == 255)
    highest_y_pixel = white_pixels[np.argmin(white_pixels[:, 0])]
    lowest_y_pixel = white_pixels[np.argmax(white_pixels[:, 0])]

    cropped_image = shifted_image[highest_y_pixel[0] : lowest_y_pixel[0], :]
    # Crop the image to the right of the rightmost white point

    white_pixels = np.argwhere(cropped_image == 255)
    x_max = 0
    # Find the white pixel with the highest x-coordinate (column index)
    if len(white_pixels) > 0:
        highest_x_pixel = white_pixels[np.argmax(white_pixels[:, 1])]
        x_max, y = highest_x_pixel[1], highest_x_pixel[0]

    cropped_image = cropped_image[:, :x_max]

    height, width, _ = cropped_image.shape

    # Create a black background
    black_background1 = np.zeros((200, 200, 3), dtype=np.uint8)

    # Calculate the scaling factor
    scaling_factor = min(200 / width, 200 / height)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)

    # Resize the image while maintaining aspect ratio
    resized_image = cv2.resize(cropped_image, (new_width, new_height))
    # Calculate the position to paste the image at the center
    x_offset = (200 - new_width) // 2
    y_offset = (200 - new_height) // 2

    # Paste the resized image onto the black background
    black_background1[
        y_offset : y_offset + new_height, x_offset : x_offset + new_width
    ] = resized_image

    # Model input image
    input_image = np.expand_dims(black_background1, axis=0)

    val = loaded_model.predict(input_image)  # model predict
    posture_status = "Posture is normal"
    text_color = "green"

    if val < 0.5:
        logging.info("bad posture")
        posture_status = "Bad posture"
        text_color = "red"  # Red
        if args.sound:
            mixer.music.play()
    else:
        posture_status = "Posture is normal"
        text_color = "green"
        logging.info("good posture")

    # Clear previous images
    if not args.no_plot:
        ax1.clear()
        ax2.clear()
        # Display images
        ax1.imshow(output_img)
        ax1.set_title("OpenPose + Detectron2 ")
        ax1.axis("off")  # Hide axes

        # Display model input image
        ax2.imshow(black_background1)
        ax2.set_title("Model input")
        ax2.axis("off")
        fig_text.set_text(posture_status)
        fig_text.set_color(text_color)
        plt.draw()
        plt.pause(0.05)

    cam.release()


def main():

    parser = argparse.ArgumentParser(description="Posture Detection Program")
    parser.add_argument(
        "--sound",
        action="store_true",
        default=False,
        help="Enable sound feedback",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        default=False,
        help="Disable plotting",
    )

    parser.add_argument("--log", default="app.log", help="Specify the log file name")
    args = parser.parse_args()

    # Initialize logging
    logging.basicConfig(filename="app.log", level=logging.ERROR)

    # Init music
    mixer.init()
    mixer.music.load("beep.wav")

    # Load the detector model
    loaded_model = load_model("models/model.h5")

    # Create OpenPose object
    try:
        params = set_params()  # function to set OpenPose parameters
        openpose = op.WrapperPython()  # Create OpenPose wrapper object
        openpose.configure(params)  # Configure OpenPose with params
        openpose.start()  # Start OpenPose
    except Exception as e:
        logging.error(f"Error initializing OpenPose: {str(e)}")

    # Create Detectron2 object
    try:
        cfg = init_detectron2()
        predictor = DefaultPredictor(cfg)
    except Exception as e:
        logging.error(f"Error initializing Detectron2: {str(e)}")

    # Start the program
    try:
        while True:
            process_image(loaded_model, openpose, predictor, cfg, args)
    except KeyboardInterrupt:
        logging.log("Program interrupted by user.")
    except Exception as e:
        logging.error(f"Error during image processing: {str(e)}")


if __name__ == "__main__":
    main()
