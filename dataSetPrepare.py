import argparse
import os
import random
from PIL import Image
import numpy as np
"""This python script will divide the videos into two sets(train & test)"""
"""Paths of pictures from the two sets will be written into file individually"""
"""Picture size used = 224*224*3"""



"""preparing"""
pathsDictionary = {
    'Deepfakes': 'picture_224/fake/Deepfakes/',
    'FaceSwap': 'picture_224/fake/FaceSwap/',
    'NeuralTextures': 'picture_224/fake/NeuralTextures/',
    'Face2Face': 'picture_224/fake/Face2Face/',
    'raw': 'picture_224/raw/youtube/'
}
videoNumber = dict()
for path in pathsDictionary:
    videoNumber[path] = 0

"""create indexes of the two set"""
def splitSet(target, splitRate=0.8):
    pictLists = os.listdir(pathsDictionary[target])
    for pictureName in pictLists:
        try:
            videoID = int(pictureName.split('-')[0])
            videoNumber[target] = max(videoID, videoNumber[target])
        except ValueError as e:
            print(pictureName)
            pass
    print(target)
    print(videoNumber[target])
    trainLength = round(videoNumber[target] * splitRate)
    testLength = videoNumber[target] - trainLength
    uList = list(range(1, videoNumber[target] + 1))
    random.shuffle(uList)
    testSet = uList[:testLength]
    trainSet = uList[testLength:]
    return {"name": target, "testSet": testSet, "trainSet": trainSet, "Num": videoNumber[target]}


"""generate the picture path of the two sets"""

def generateDataSet(target='Deepfakes', splitRate=0.8):
    paths = {'raw': 'picture_224/raw/youtube/', target: pathsDictionary[target]}
    train = []
    test = []
    for part in paths:
        splitRes = splitSet(part, splitRate)
        partLabel = 0 if (part == 'raw') else 1
        pictureList = os.listdir(paths[part])
        for pictureName in pictureList:
            try:
                videoID = int(pictureName.split('-')[0])
                if videoID in splitRes["trainSet"]:
                    train.append((paths[part] + pictureName, partLabel))
                elif videoID in splitRes["testSet"]:
                    test.append((paths[part] + pictureName, partLabel))
            except ValueError as e:
                pass
    random.shuffle(train)
    random.shuffle(test)
    return train, test





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=str, default="Deepfakes")
    parser.add_argument('-r', '--rate', type=int, default=0.8)
    arg = parser.parse_args()
    return arg



"""write the paths to local file"""
if __name__ == "__main__":
    args = parse_args()
    trainDataSet, testDataSet = generateDataSet(args.type,args.rate)
    if not os.path.isdir("dataSetDict_224/"):
        os.mkdir("dataSetDict_224/")
    if not os.path.isdir("dataSetDict_224/"+args.type):
        os.mkdir("dataSetDict_224/"+args.type)
    trainDataFile = open("dataSetDict_224/"+args.type+"/trainDataSet.txt", 'w')
    testDataFile = open("dataSetDict_224/"+args.type+"/testDataSet.txt", 'w')
    print(len(trainDataSet), len(testDataSet))
    for i in trainDataSet:
        pict,label = i
        trainDataFile.write(pict + " " + str(label) + "\n")
    for i in testDataSet:
        pict,label = i
        testDataFile.write(pict + " " + str(label) + "\n")