#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import tensorflow #open source used for both ML and DL for computation
from tensorflow.keras.datasets import mnist #mnist dataset
from tensorflow.keras.models import Sequential #it is a plain stack of layers
from tensorflow.keras import layers #A Layer consists of a tensor- in tensor-out computat ion funct ion
from tensorflow.keras.layers import Dense, Flatten #Dense-Dense Layer is the regular deeply connected r
#faltten -used fot flattening the input or change the dimension
from tensorflow.keras.layers import Conv2D #convolutional Layer
from keras.utils import np_utils #used for one-hot encoding
import matplotlib.pyplot as plt   #used for data visualization


# In[3]:


(x_train, y_train), (x_test, y_test)=mnist.load_data () #splitting the mnist data into train and test
print (x_train.shape)  #shape is used for give the dimens ion values #60000-rows 28x28-pixels
print (x_test.shape)
x_train[0]


# In[4]:


plt.imshow(x_train[3000]) 


# In[5]:


np.argmax(y_train[3000])


# In[6]:


#Reshaping Dataset
#Reshaping to format which CNN expects (batch, height, width, channels)
x_train=x_train.reshape (60000, 28, 28, 1).astype('float32')
x_test=x_test.reshape (10000, 28, 28, 1).astype ('float32')
#Applying One Hot Encoding
number_of_classes = 10  #storing the no of classes in a variable
y_train = np_utils.to_categorical (y_train, number_of_classes) #converts the output in binary format
y_test = np_utils.to_categorical (y_test, number_of_classes)


# In[7]:


#Add CNN Layers
#create model
model=Sequential ()
#adding modeL Layer
model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
#flatten the dimension of the image
model.add(Flatten())
#output layer with 10 neurons
model.add(Dense(number_of_classes,activation = 'softmax'))


# In[8]:


#Compile model
model.compile(loss= 'categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)


# In[9]:


#fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=32)


# In[16]:


# Final evaluation of the model
metrics = model.evaluate(x_test, y_test, verbose=0)
print("Metrics (Test loss &Test Accuracy) : ")
print(metrics)


# In[12]:


prediction=model.predict(x_test[6000:6001])
print(prediction)


# In[17]:


plt.imshow(x_test[6000])


# In[18]:


import numpy as np
print(np.argmax(prediction, axis=1)) #printing our Labels from first 4 images

np.argmax(y_test[6000:6001]) #printing the actual labels


# In[19]:


model.save('mnistCNN.h5')
get_ipython().system('tar -zcvf handwritten-digit-recognition-model_new.tgz mnistCNN.h5')


# In[19]:


get_ipython().system('pip install watson-machine-learning-client --upgrade')


# In[42]:


from ibm_watson_machine_learning import APIClient
credentials ={
    "url":"https://us-south.ml.cloud.ibm.com",
    "apikey":"qMWf3TNlOnECf3QWhXLLXXlSjol3iwYANNKEj1SvR4UQ"
}
client = APIClient(credentials)


# In[43]:


client.spaces.get_details()


# In[46]:


def guid_from_space_name(client,deploy):
  space = client.spaces.get_details()
  return (next(item for item in space['resources'] if item['entity']['name']==deploy)['metadata']['id'])


# In[44]:


client.spaces.get_details()


# In[54]:


space_uid = guid_from_space_name(client, 'sparks')
print("Space UID = " +space_uid)


# client.set.default_space(space_uid)

# In[55]:


client.set.default_space(space_uid)


# In[56]:


client.software_specifications.list()


# In[57]:


software_space_uid = client.software_specifications.get_uid_by_name('tensorflow_rt22.1-py3.9')
software_space_uid


# In[ ]:





# In[58]:


model_details = client.repository.store_model(model='handwritten-digit-recognition-model_new.tgz',meta_props={
    client.repository.ModelMetaNames.NAME:"CNN Digit recognition model",
    client.repository.ModelMetaNames.TYPE:"tensorflow_2.7",
    client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:software_space_uid
})


# In[30]:


model_details


# In[59]:


model_id = client.repository.get_model_id(model_details)
model_id


# In[61]:


client.repository.download(model_id,'DigitRecog_IBM_model1.tar.gz')


# In[62]:


ls


# In[63]:


#test
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import numpy as np
model = load_model("mnistCNN.h5")
import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='is_QZGPyU8oxZr3W-td-LCHXS3QPMaWArILi18FdSyGT',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.ap.cloud-object-storage.appdomain.cloud')

bucket = 'handwrittenimagerecognition-donotdelete-pr-8tlrnykut46vpi'
object_key = 'mnist-dataset-1024x424 (2).png'

streaming_body_1 = cos_client.get_object(Bucket=bucket, Key=object_key)['Body']

# Your data file was loaded into a botocore.response.StreamingBody object.
# Please read the documentation of ibm_boto3 and pandas to learn more about the possibilities to load the data.
# ibm_boto3 documentation: https://ibm.github.io/ibm-cos-sdk-python/
# pandas documentation: http://pandas.pydata.org/
img = Image.open(streaming_body_1).convert("L") # convert image to monochrome
img = img.resize( (28,28) ) # resizing of input image


# In[64]:


img


# In[65]:


im2arr = np.array(img) #converting to image
im2arr = im2arr.reshape(1, 28, 28, 1) #reshaping according to our requirement
pred = model.predict(im2arr)
print(pred)


# In[ ]:





# In[66]:


print(np.argmax(pred, axis=1)) #printing our Labels


# In[ ]:




