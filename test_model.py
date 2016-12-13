import json
from keras.models import model_from_json

from zimpy.camera_preprocessor import predict_images

with open('model.json', 'r') as jfile:
	the_json = json.load(jfile)
	print(json.loads(the_json))
	model = model_from_json(the_json)

model.compile("adam", "mse")
weights_file = 'model.h5'
model.load_weights(weights_file)

# predict 3 images and compare accurach
predict_images(model)
