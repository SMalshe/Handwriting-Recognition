import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

#Data loading
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Data normalization
x_train = x_train/255.0
x_test = x_test/255.0

#Creating the actual model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) #Alternate way to use relu
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

#Training the model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 3)

#Saving and loading model
model.save('handwritten.keras')
model = tf.keras.models.load_model('handwritten.keras')

#Evaluate Model
loss, accuracy = model.evaluate(x_test, y_test)

#Test on Custom Images
image_number = 0
while os.path.isfile(f"Digit{image_number}.png"):
    try:
        img = cv2.imread(f"Digit{image_number}.png")[:,:,0]
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        plt.imshow(img[0], cmap = plt.cm.binary)
        plt.show()
        print(f"The number is probably: {np.argmax(prediction)}")
    except FileNotFoundError:
        print("File wasn't found")
    except IndexError:
        print("Index out of range")
    except:
        print("Unexpected error:", sys.exc_info()[0])
    finally: image_number += 1

#Edge Cases
image_number = 10
while os.path.isfile(f"Digit_{image_number}.png"):
    try:
        img = cv2.imread(f"Digit_{image_number}.png")[:,:,0]
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        plt.imshow(img[0], cmap = plt.cm.binary)
        plt.show()
        print(f"The number is probably: {np.argmax(prediction)}")
    except FileNotFoundError:
        print("File wasn't found")
    except IndexError:
        print("Index out of range")
    except:
        print("Unexpected error:", sys.exc_info()[0])
    finally: image_number += 1