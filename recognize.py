#!/usr/bin/env python3

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from segment import segment_image

filename, extension = sys.argv[1].split(".")
threshold = 125


pic = Image.open(sys.argv[1]).convert("RGB")
sw, sh = pic.size
ratio = 1600 / sw
pic = pic.resize((int(sw * ratio), int(sh * ratio)))
grey = pic.convert("L")
bin = grey.point(lambda p: p > threshold and 255)

#bin.save(filename + "_" + extension + "_binary.png")
model = tf.keras.models.load_model('network.h5')
s = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:?!-_;'()[]{}"

blob = segment_image(bin, resize=(16, 32), put_space=True)
text = ""
for al, r in enumerate(blob):
    print(f"Processing line {al} of {len(blob)}", end="\r")
    for c in r:
        if type(c) != str:
            prediction = model.predict(np.expand_dims(np.array(c, dtype=float) / 255, axis=0))
            text += s[np.argmax(prediction[0])]
        else:
            text += " "
    text += "\n"
print()
file = open(filename + "_" + extension + ".txt", "w")
file.write(text)
file.close()
#it = 0
#for b in blob:
#    for c in b:
#        c.save("data/" + str(it) + ".png")
#        it += 1
