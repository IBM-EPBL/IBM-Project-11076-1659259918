#Importing the Keras libraries and packages
from tensorflow.keras.models import load_model
from PIL import Image #used for manipulating image uploaded by the user.
import numpy as np #used for numerrical analysis
model = load_model('mnistCNN.h5')
img = Image.open('digits/digit9.png').convert("L") # convert image to monochrome
img = img.resize( (28, 28) ) # resizing of input image
im2arr = np.array(img) #converting to image
im2arr = im2arr.reshape(1, 28, 28, 1) #reshaping according to our requirement
y_pred = model.predict(im2arr) #predicting the results
print(y_pred)
import numpy as np
print(np.argmax(y_pred, axis=1)) #printing our Labels from first 4 images
