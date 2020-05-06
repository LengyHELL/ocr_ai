#!/usr/bin/env python3

import sys

def getdiff(a, b):
    return abs(a - b)

ori = [l.split(" ") for l in open(sys.argv[1]).read().strip().split("\n")]
est = [l.split(" ") for l in open(sys.argv[2]).read().strip().split("\n")]

score = 0
if len(ori) == len(est):
    score += len(ori)
    for i in range(len(ori)):
        if len(ori[i]) == len(est[i]):
            score += len(ori[i])
            for j in range(len(ori[i])):
                if len(ori[i][j]) == len(est[i][j]):
                    score += len(ori[i][j])
                    for k in range(len(ori[i][j])):
                        if ori[i][j][k] == est[i][j][k]:
                            score += 1
                else:
                    diff = getdiff(len(ori[i][j]), len(est[i][j]))
                    score += len(ori[i][j]) - diff
        else:
            diff = getdiff(len(ori[i]), len(est[i]))
            score += len(ori[i]) - diff
else:
    diff = getdiff(len(ori), len(est))
    score += len(ori) - diff

print(score)
