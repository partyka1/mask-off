import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import tensorflow as tf
import base64

import face_detection

# Face Detector
import face_detection
import time

# processing images
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import math
import numpy

# Max enabled image width is set as 300. If greater we will resize the input images
BASEWIDTH = 1600
MASKED_COLOR = '#00b189'
NOT_MASKED_COLOR = '#365abd'

ROOT_DIR = Path().absolute()
FONT_TTF_LOC = str(Path(ROOT_DIR) / 'data' / 'fonts' / 'Arvo-Regular.ttf')


face_detector = face_detection.build_detector('RetinaNetMobileNetV1',
                                              confidence_threshold=.5,
                                              nms_iou_threshold=.3)

classifier_dir = Path(ROOT_DIR) / 'data' / 'classifier_model_weights'
classifier = tf.keras.models.load_model(classifier_dir / 'best.h5')


def resize_image(img):
    # Resize image by keeping the aspect ratio if image witdth is greater than BASEWIDTH
    if img.size[0] > BASEWIDTH:
        wpercent = (BASEWIDTH / float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((BASEWIDTH,hsize), Image.ANTIALIAS)
    return img

def detect_face(resized_img):
    open_cv_image = np.array(resized_img)
    open_cv_image = open_cv_image[:, :, ::-1].copy() # Convert RGB to BGR
    return face_detector.detect(open_cv_image)

def annotate_image(img, face_coords, classified_face_scores, classification_labels):
    pil_draw = ImageDraw.Draw(img)
    for idx, coords in enumerate(face_coords):
        x1, y1, x2, y2, _ = coords
        label = classification_labels[idx]
        color = MASKED_COLOR if label == 'masked' else NOT_MASKED_COLOR
        display_str = "{}: {:}%".format(label, math.ceil(classified_face_scores[idx] * 100))

        # Draw rectangle for detected face
        pil_draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label text box
        # portion of image width you want text width to be
        img_fraction = 0.2
        font_size = 5  # starting font size

        font = ImageFont.truetype(FONT_TTF_LOC, font_size)
        image_size = img.size[0]

        while font.getsize(display_str)[0] < img_fraction * image_size:
            # iterate until the text size is just larger than the criteria
            font_size += 1
            font = ImageFont.truetype(FONT_TTF_LOC, font_size)

        # Find coordinates of bounding text box
        w, h = font.getsize(display_str)
        pil_draw.rectangle([x1, y1, x1 + w, y1 + h], fill=color)
        pil_draw.text((x1, y1), display_str, font=font)
    return img



def classify_faces(img_raw, face_coords):
    classification_scores = []
    # Iterate over detected face coordinates to find
    for coords in face_coords:
        x1, y1, x2, y2, _ = coords
        cropped_face = img_raw.crop((x1, y1, x2, y2))
        img = np.float32(cropped_face)
        img = cv2.resize(img, (112, 112))
        preprocessed_img = tf.keras.applications.mobilenet.preprocess_input(img)
        preprocessed_img = preprocessed_img[np.newaxis, ...]
        pred = classifier.predict(preprocessed_img)[0][0]
        classification_scores.append(pred)
    return classification_scores


def convert_pil_to_base64(annotated_image, image_type):
    buffered = BytesIO()
    if image_type == 'jpg':
        annotated_image.save(buffered, format='jpeg')
    elif image_type == 'png':
        annotated_image.save(buffered, format='png')
    else:
        annotated_image.save(buffered, format=image_type)
    annotated_image_base64 = base64.b64encode(buffered.getvalue())
    return annotated_image_base64.decode('utf-8')

def predict_masked_faces(body):
    base64_image = body['image'].encode('utf-8')
    image_type = body['type']

    # Convert image from base64 to PIL Image and resize it to improve the performance
    img_raw = Image.open(BytesIO(base64.b64decode(base64_image)))

    # Resize image
    resized_img = resize_image(img_raw)

    # Detect face coordinates from the raw image
    face_coords = detect_face(resized_img)

    # Classify detected faces whether they have a mask or not
    classified_face_scores = classify_faces(resized_img, face_coords)

    # Find labels
    classification_labels = np.where(np.array(classified_face_scores) > 0.5, 'masked', 'not masked').tolist()

    # Annotate base image with detected faces and mask classification
    annotated_image = annotate_image(resized_img, face_coords, classified_face_scores, classification_labels)

    # Convert PIL image to base64
    annotated_image_base_64 = convert_pil_to_base64(annotated_image, image_type)

    # Convert score to string type to make it serializable
    classified_face_scores = [float(score) for score in classified_face_scores]

    return {
        'detected_face_coordinates': face_coords.tolist(),
        'detected_mask_scores': classified_face_scores,
        'detected_face_labels': classification_labels,
        'annotated_image': annotated_image_base_64,
        'image_type': image_type
    }


root_dir = Path().absolute()
data_dir = root_dir / 'data'
model_dir = data_dir / 'classifier_model_weights'

best_model = tf.keras.models.load_model(model_dir / 'best.h5')

cap = cv2.VideoCapture(0)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    cv2.imwrite('webcam.jpg', frame)


    with open('webcam.jpg', 'rb') as img_file:
        # We need to convert jpeg image into base64 string image to serialize the image.
        body = {
            'image': base64.b64encode(img_file.read()).decode('utf-8'),
            'type': 'jpg'
        }
        _time = time.time()
        response = predict_masked_faces(body)
        print(str(time.time() - _time))
    # Show Annotated Image to give an idea what model prediction looks like
    pilImg = Image.open(BytesIO(base64.b64decode(response['annotated_image']))).convert('RGB')
    open_cv_image = numpy.array(pilImg)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    cv2.imshow('on & on', open_cv_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()