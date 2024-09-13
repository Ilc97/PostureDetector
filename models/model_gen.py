import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop


train = ImageDataGenerator(rescale=1 / 255)
validation = ImageDataGenerator(rescale=1 / 255)

train_dataset = train.flow_from_directory(
    "postureDataset/train/",
    target_size=(200, 200),
    interpolation="nearest",
    batch_size=16,
    class_mode="binary",
)

validation_dataset = train.flow_from_directory(
    "postureDataset/validation/",
    target_size=(200, 200),
    interpolation="nearest",
    batch_size=16,
    class_mode="binary",
)

train_dataset.class_indices

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            16, (3, 3), activation="relu", input_shape=(200, 200, 3)
        ),
        tf.keras.layers.MaxPool2D(2, 2),
        #
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2),
        #
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2),
        ##
        tf.keras.layers.Flatten(),
        ##
        tf.keras.layers.Dense(512, activation="relu"),
        ##
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    loss="binary_crossentropy",
    optimizer=RMSprop(learning_rate=0.001),
    metrics=["accuracy"],
)

model_fit = model.fit(
    train_dataset,
    steps_per_epoch=3,
    epochs=100,
    validation_data=validation_dataset,
    shuffle=True,
)

model.save("model.h5")
# Testing

dir_path = "postureDataset/testing/"

for i in os.listdir(dir_path):
    img = image.load_img(
        dir_path + "//" + i, target_size=(200, 200), interpolation="nearest"
    )

    X = image.img_to_array(img)
    X = X / 255
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])

    val = model.predict(images)
    if val < 0.5:
        print(val)
        print("bad")
    else:
        print(val)
        print("good")
