import argparse
import base64
import json

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
import cv2

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

y_mini = 0  # 60
y_maxi = 40  # 16#110
y_dim = y_maxi - y_mini

x_mini = 0  # 60#60
x_maxi = 80  # 260#260
x_dim = x_maxi - x_mini

IMG_WIDTH = 80  # 32
IMG_HEIGHT = 40  # 16


def test(img_arrray, x):
    return img_arrray * x


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
    # print(image_array.shape)

    # print (image_array.shape, 'b')
    # image_array = image_array[50:140, 0:360]

    # image_array = image_array[65:105, 60:260]
    # image_array = image_array[y_mini:y_maxi, x_mini: x_maxi]
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2YUV)
    image_array = cv2.resize(image_array, (IMG_WIDTH, IMG_HEIGHT))

    # image_array =image_array / 255.
    ########image_array = image_array - 0.5
    # image_array =normalize_set(image_array, False)

    r, g, b = cv2.split(image_array)
    r_flat, g_flat, b_flat = r.flatten(), g.flatten(), b.flatten()
    r_temp = r_flat / 255.
    r_norm = r_temp - 0.5
    g_temp = g_flat / 255.
    g_norm = g_temp / 0.5
    b_temp = b_flat / 255.
    b_norm = b_temp - 0.5

    # r_norm = -0.5 + ((r_flat - np.min(r_flat))/ (np.max(r_flat)-np.min(r_flat)))
    # g_norm = -0.5 + ((g_flat - np.min(g_flat))/ (np.max(g_flat)-np.min(g_flat)))
    # b_norm = -0.5 + ((b_flat - np.min(b_flat))/ (np.max(b_flat)-np.min(b_flat)))

    # g_norm = g_flat*1.
    # b_norm = b_flat*1.






    # r_norm32, g_norm32, b_norm32 =np.reshape(r_norm, (50,200)), np.reshape(g_norm, (50,200)),np.reshape(b_norm, (50,200))
    r_norm32, g_norm32, b_norm32 = np.reshape(r_norm, (y_maxi - y_mini, x_maxi - x_mini)), np.reshape(g_norm, (
    y_maxi - y_mini, x_maxi - x_mini)), np.reshape(b_norm, (y_maxi - y_mini, x_maxi - x_mini))
    image_array = cv2.merge((r_norm32, g_norm32, b_norm32))

    # print (image_array.shape)
    # image_array = cv2.resize(image_array, (160, 360))
    # image_array = resize_image(image_array, [160,360])

    # transformed_image_array = image_array[None, :, :, :]
    # transformed_image_array = image_array.reshape(( None,
    #                                           image_array.shape[0],
    #                                           image_array.shape[1],
    #                                           3))

    # print(transformed_image_array.shape)

    # image_array =normalize_set(image_array, False)

    transformed_image_array = image_array[None, :, :, :]

    ##print(transformed_image_array.shape)
    ###print(model.predict(transformed_image_array, batch_size=1), 'pred')


    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    print(steering_angle)
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.2
    # print(steering_angle, throttle)
    send_control(steering_angle, throttle)
    # send_control(-1, throttle)
    # send_control(b_norm[0], throttle)


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
    with open(args.model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
