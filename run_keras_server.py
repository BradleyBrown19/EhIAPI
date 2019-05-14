from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from fastai import *
from fastai.vision import *
from PIL import Image
import tensorflow as tf
import numpy as np
import flask
import io

app = flask.Flask(__name__)
learn = load_learner('./models')

@app.route("/predict", methods=["POST"])
def predict():
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		print(flask.request.files)
		if flask.request.files.get("image"):
			print(image)
			image = flask.request.files["image"]

			image = open_image(image)

			image = image.resize(100)

			preds = learn.predict(image)
			data["predictions"] = []

			data["success"] = True

			index = int(preds[1])

			confidence = preds[2][index]

			prediction = str(preds[0])
			confidence = float(confidence)
						
			data["predictions"] = {'label': prediction, 'confidence': confidence}			
			

	# return the data dictionary as a JSON response
	response = flask.jsonify(data)
	response.headers.add('Access-Control-Allow-Origin', '*')
	print(response)
	return response

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	app.run()