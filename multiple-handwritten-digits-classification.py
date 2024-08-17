import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
import matplotlib.pyplot as plt

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

from utils import *
np.set_printoptions(precision=2)

def my_softmax(z):
    ez=np.exp(z)
    a=ez/np.sum(ez)
    
X, y= load_data()       
# #Uncomment the following part to check if data is loaded correctly
# print ('The first element of X is: ', X[0])
# print ('The first element of y is: ', y[0,0])
# print ('The last element of y is: ', y[-1,0])
# print ('The shape of X is: ' + str(X.shape)) ##----> should output 5000,400
# print ('The shape of y is: ' + str(y.shape))  ##-----> should output 5000,1

tf.random.set_seed(1234) # for consistent results
model = Sequential(
    [               
        tf.keras.layers.InputLayer((400,)),
        tf.keras.layers.Dense(25, activation="relu", name="L1"),
        tf.keras.layers.Dense(15, activation="relu", name="L2"),
        tf.keras.layers.Dense(10, activation="linear", name="L3")
        
    ], name = "my_model" 
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)

history = model.fit(
    X,y,
    epochs=40
)

# # Uncomment the following part to compare the predictions vs the labels for a random sample of 64 digits. This takes a moment to run.
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# m, n = X.shape
# fig, axes = plt.subplots(8,8, figsize=(5,5))
# fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]
# widgvis(fig)
# for i,ax in enumerate(axes.flat):
#     # Select random indices
#     random_index = np.random.randint(m)
    
#     # Select rows corresponding to the random indices and
#     # reshape the image
#     X_random_reshaped = X[random_index].reshape((20,20)).T
    
#     # Display the image
#     ax.imshow(X_random_reshaped, cmap='gray')
    
#     # Predict using the Neural Network
#     prediction = model.predict(X[random_index].reshape(1,400))
#     prediction_p = tf.nn.softmax(prediction)
#     yhat = np.argmax(prediction_p)
    
#     # Display the label above the image
#     ax.set_title(f"{y[random_index,0]},{yhat}",fontsize=10)
#     ax.set_axis_off()
# fig.suptitle("Label, yhat", fontsize=14)
# plt.show()

##Uncomment the following part to check which images were NOT classified correctly
print( f"{display_errors(model,X,y)} errors out of {len(X)} images")
plt.show()
