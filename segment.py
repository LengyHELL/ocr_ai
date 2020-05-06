#!/usr/bin/env python3

import numpy as np
from PIL import Image, ImageOps


def pad_to_height(img, ph):
    w, h = img.size
    delta_w = (ph // 2) - w
    delta_h = ph - h
    if delta_w < 0:
        delta_w = 0
    if delta_h < 0:
        delta_h = 0
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), 0)
    return ImageOps.expand(img, padding, fill=255)

def getOverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def blob_detect(bin, line_sep_limit):
    width, height = bin.size
    current_label = 1
    values = bin.load()
    labels = np.zeros((width, height), dtype=int)
    blobs = []
    lines = []

    prev_line = 0
    curr_line = 0
    for y in range(height):
        if (prev_line > curr_line) and (curr_line <= line_sep_limit):
            lines.append(blobs)
            blobs = []
        prev_line = curr_line
        curr_line = 0

        for x in range(width):
            if values[x, y] == 0:
                curr_line += 1

            if (values[x, y] == 0) and (labels[x, y] == 0):
                labels[x, y] = current_label
                queue = [[x, y]]
                t_blobs = [[x, y]]
                while(len(queue) > 0):
                    temp = queue[0]
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            if (i != 0) or (j != 0):
                                tx = i + temp[0]
                                ty = j + temp[1]
                                if (tx >= 0) and (tx < width) and (ty >= 0) and (ty < height):
                                    if (values[tx, ty] == 0) and (labels[tx, ty] == 0):
                                        labels[tx, ty] = current_label
                                        queue.append([tx, ty])
                                        t_blobs.append([tx, ty])
                    queue.pop(0)
                blobs.append(t_blobs)
                current_label += 1

    if len(blobs) > 0:
        lines.append(blobs)
    return lines

def segment_image(bin, char_size_limit=(5, 200, 5, 200), resize=(0, 0), line_sep_limit=5, sp_rate=0.3, ol_rate=0.5, one_line=False, put_space=False):
    lines = blob_detect(bin, line_sep_limit)

    images = []
    maxoflines = []
    maxch = 0
    avch = 0
    avcount = 0
    for l in lines:
        l.sort(key=lambda x: x[0])
        images_l = []
        skips = []
        prev = "none"
        maxl = 0
        for b in l:
            if b not in skips:
                skips.append(b)
                achar = np.array(b, dtype=float)
                cmin = np.amin(achar.T, axis=1)
                cmax = np.amax(achar.T, axis=1)
                cmax[0] += 1
                cmax[1] += 1
                for b2 in l:
                    ochar = np.array(b2, dtype=float)
                    omin = np.amin(ochar.T, axis=1)
                    omax = np.amax(ochar.T, axis=1)
                    omax[0] += 1
                    omax[1] += 1
                    ovl = getOverlap([cmin[0], cmax[0]], [omin[0], omax[0]])
                    size1 = cmax[0] - cmin[0]
                    size2 = omax[0] - omin[0]
                    if ((ovl / size1) >= ol_rate) or ((ovl / size2) >= ol_rate):
                        cmin[0] = min(omin[0], cmin[0])
                        cmin[1] = min(omin[1], cmin[1])
                        cmax[0] = max(omax[0], cmax[0])
                        cmax[1] = max(omax[1], cmax[1])
                        skips.append(b2)
                if put_space and (type(prev) != str):
                    omin, omax = prev
                    sp = cmin[0] - omax[0]
                    s_rate = sp / (((cmax[1] - cmin[1]) + (omax[1] - omin[1])) / 2)
                    if (s_rate >= sp_rate):
                        images_l.append("space")

                cw = cmax[0] - cmin[0]
                ch = cmax[1] - cmin[1]
                if ch > maxch:
                    maxch = ch
                if ch > maxl:
                    maxl = ch
                avch += ch
                avcount += 1
                cwl, cwu, chl, chu = char_size_limit
                if (cw > cwl) and (cw <= cwu) and (ch > chl) and (ch <= chu) :
                    images_l.append(bin.crop((cmin[0], cmin[1], cmax[0], cmax[1])))
                    prev = [cmin, cmax]

        if len(images_l) > 0:
            if one_line:
                images.extend(images_l)
            else:
                images.append(images_l)
                maxoflines.append(maxl)

    if one_line:
        images = [images]
        maxoflines.append(maxch)

    avch /= avcount

    if min(resize) > 0:
        for l in range(len(images)):
            for c in range(len(images[l])):
                if type(images[l][c]) != str:
                    images[l][c] = pad_to_height(images[l][c], int(maxoflines[l])).resize((16, 32))
    return images
