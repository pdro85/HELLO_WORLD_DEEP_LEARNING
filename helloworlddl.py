import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# load dataset in mnist  x-> images(features)  y-> labels
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# show number of images and labels from training data
print("images training:", x_train.shape[0])
print("labels training:", len(y_train))

# show number of images and labels from test data
print("images test:", x_test.shape[0])
print("labels test:", len(y_test))

# show example image loaded number [n]
plt.ion() 
plt.imshow(x_train[9], cmap=plt.cm.binary)
plt.show(block = False)
#show example label from image loaded number [n]
print("label:", y_train[9])

# Set the print options for a NumPy array 
np.set_printoptions(precision=2, suppress=True, linewidth=120)
# prints the array corresponding to image number [n]
print(np.matrix(x_train[9]))

# Reshape the training data to 2D array where each row represents a flattened image
# 28x28 = 784 pixels , 60000 images
x_train = x_train.reshape((60000, 784))
# Reshape the test data in the same way as training data
# 28x28 = 784 pixels, 10000 images
x_test = x_test.reshape((10000, 784))
# Normalize pixel values of the data to a range between 0 and 1
x_train, x_test = x_train/255.0, x_test/255.0

# Define a sequential model with one dense layer that has 5 output neurons with softmax activation
# Softmax activation is used to output probabilities for each class
# The input shape is 784, which corresponds to the flattened image size
model = Sequential()
model.add(Dense(20, activation='softmax', input_shape=(784,))) 

# Print the summary of the model
model.summary()

# Compile the model with Adam optimizer, sparse_categorical_crossentropy loss,
# and accuracy as the evaluation metric
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 

# Train the model on training data for two epochs
model.fit(x_train, y_train, epochs=2)

# Evaluate the model on test data and print (verbose=2) the test loss and accuracy
model.evaluate(x_test,  y_test, verbose=2)

# Get the weights of the model
w = model.get_weights()
print ("layers:",len(w))
if len(w) > 0:
  # Convert the weights of the third layer to a numpy array
  w = np.asarray(w[0])
  # Visualize the first column of weights as an image
  plt.figure()
  # show number 4
  # blue color is a good zone
  plt.imshow(w[:, 4].reshape([28, 28]), cmap=plt.get_cmap('seismic_r'))
  plt.show(block=False)
else:
  print("insufficient layers")

#print array corresponding to image weights of number 4
np.set_printoptions(precision=0, suppress = True, linewidth=120)
print(np.matrix (255*(w[:,4].reshape([28,28]))))










input("Press key to exit...")

