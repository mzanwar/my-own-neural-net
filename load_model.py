import time

import cv2
import numpy
import tensorflow as tf


def prepare(image_path, size):
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_array = cv2.resize(img_array, (size, size))
    final_array = resized_array.reshape(-1, size, size, 1)
    return final_array

if __name__ == '__main__':
    print("# loading model")
    model = tf.keras.models.load_model("./tf_models/tf-mnist.h5")
    for i in range(1000):
        prediction = model.predict(prepare("image.png", 28))
        print(prediction)
        print(numpy.argmax(prediction))
        time.sleep(1)
    # draw thickness

    # show image

