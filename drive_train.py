import argparse
import base64
import json
import cv2
import os
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

from zimpy.camera_preprocessor import preprocess_image, flip_image

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    image_array = preprocess_image(image_array)
    # transformed_image_array = image_array
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
    # flipped_steering_angle = float(model.predict(np.fliplr(image_array)[None, :, :, :], batch_size=1))
    # mean_steering_angle = np.mean([steering_angle, flipped_steering_angle])
    # print('mean steering angle', mean_steering_angle)
    # steering_angle = mean_steering_angle
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.2
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
                        help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    print(args)

    # os.system("python training_test.py --epochs 2 --batch_size 128 --algo_mode 2 --repickle True")
    # os.system("python training_test.py --epochs 5 --batch_size 128 --algo_mode 3 --repickle True")
    os.system("python training_test.py --epochs 5 --batch_size 32 --algo_mode 3 --repickle True --lr 0.0001")

    with open(args.model, 'r') as jfile:
        the_json = json.load(jfile)
        print(json.loads(the_json))
        model = model_from_json(the_json)

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
