#!/usr/bin/env python3
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps
from segment import segment_image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras

def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y

def create_dataset(xs, ys, n_classes):
    ys = tf.one_hot(ys, depth=n_classes)
    return tf.data.Dataset.from_tensor_slices((xs, ys)).map(preprocess).shuffle(ys.shape[0]).batch(n_classes)

s = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z a b c d e f g h i j k l m n o p q r s t u v w x y z 0 1 2 3 4 5 6 7 8 9 . , : ? ! - _ ; ' ( ) [ ] { }"
chars_size = (len(s) + 1) // 2

chars = []
outs = []
fonts = [f.strip() for f in open("fonts.txt").read().strip().split("\n")]
print("Creating dataset...")
good = ""
msg = ""
for i, f in enumerate(fonts):
    font = ImageFont.truetype(f, 64)
    size = font.getsize(s)
    img = Image.new('RGB', size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), s, (0, 0, 0), font = font)
    img = img.convert("L").point(lambda p: p > 200 and 255)
    blobs = segment_image(img, resize=(16, 32), one_line=True, ol_rate=0.3, char_size_limit=(0, 200, 0, 200))
    print(len(blobs[0]), i, end=" ")
    if len(blobs[0]) == chars_size:
        msg = "success"
        for j in range(chars_size):
            chars.append(np.array(blobs[0][j], dtype=float))
            outs.append(j)
    else:
        msg = "failed"
    print(f, msg)
    print(str(i + 1) + "/" + str(len(fonts)), end="\r")
    if len(blobs[0]) == chars_size:
        pass
        #good += f + "\n"

print(good)

X = np.array(chars)
Y = np.array(outs)

train_dataset = create_dataset(X, Y, chars_size)

model = keras.Sequential([
    keras.layers.Reshape(target_shape=(16 * 32,), input_shape=(32, 16)),
    keras.layers.Dense(units=512, kernel_regularizer=keras.regularizers.l2(0.0001), activation="relu"),
    keras.layers.Dense(units=256, kernel_regularizer=keras.regularizers.l2(0.0001), activation="relu"),
    keras.layers.Dense(units=128, kernel_regularizer=keras.regularizers.l2(0.0001), activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=chars_size, activation="softmax"),
])

model.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

print("\nTraining the network...")
history = model.fit(
    train_dataset.repeat(),
    epochs=500,
    steps_per_epoch=500
)

for iter in range(10):
    test = np.random.randint(low=0, high=(len(chars) - 1))
    prediction = model.predict(np.expand_dims(chars[test], axis=0))
    print("Guess: ", s[np.argmax(prediction[0]) * 2], "- True:", s[outs[test] * 2])

model.save("network.h5")
