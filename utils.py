import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid

def load_data():
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    return X, y

def widgvis(fig):
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    
# def display_errors(model,X,y):
#     f = model.predict(X)
#     yhat = np.argmax(f, axis=1)
#     doo = yhat != y[:,0]
#     idxs = np.where(yhat != y[:,0])[0]
#     if len(idxs) == 0:
#         print("no errors found")
#     else:
#         cnt = min(8, len(idxs))
#         fig, ax = plt.subplots(1,cnt, figsize=(5,1.2))
#         fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.80]) #[left, bottom, right, top]
#         widgvis(fig)

#         for i in range(cnt):
#             j = idxs[i]
#             X_reshaped = X[j].reshape((20,20)).T

#             # Display the image
#             ax[i].imshow(X_reshaped, cmap='gray')

#             # Predict using the Neural Network
#             prediction = model.predict(X[j].reshape(1,400))
#             prediction_p = tf.nn.softmax(prediction)
#             yhat = np.argmax(prediction_p)

#             # Display the label above the image
#             ax[i].set_title(f"{y[j,0]},{yhat}",fontsize=10)
#             ax[i].set_axis_off()
#             fig.suptitle("Label, yhat", fontsize=12)
           
#     return(len(idxs))    

def display_errors(model, X, y):
    f = model.predict(X)
    yhat = np.argmax(f, axis=1)
    idxs = np.where(yhat != y[:,0])[0]
    
    if len(idxs) == 0:
        print("No errors found")
        return 0
    else:
        n_errors = len(idxs)
        n_cols = 8
        n_rows = math.ceil(n_errors / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols/8, 1.2*n_rows))
        fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.95])  # [left, bottom, right, top]
        
        # Flatten the axes array
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes
        
        for i, idx in enumerate(idxs):
            X_reshaped = X[idx].reshape((20, 20)).T
            
            # Display the image
            axes[i].imshow(X_reshaped, cmap='gray')
            
            # Predict using the Neural Network
            prediction = model.predict(X[idx].reshape(1, 400))
            prediction_p = tf.nn.softmax(prediction)
            yhat_i = np.argmax(prediction_p)
            
            # Display the label above the image
            axes[i].set_title(f"{y[idx,0]},{yhat_i}", fontsize=10)
            axes[i].set_axis_off()
        
        # Turn off any unused subplots
        for j in range(n_errors, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Label, yhat", fontsize=12)
        return n_errors