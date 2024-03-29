import argparse
from PIL import Image
import numpy as np
from tensorflow import keras
import tensorflow as tf
import tensorflow

"""This python script will evaluate models of each epoch."""
"""To speed up the process, we only use 10% of the test set(about 4400 samples)"""
"""picture size used = 224*224*3"""


testDataSet = []
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=str, default="Deepfakes")
    parser.add_argument('-m', '--model', type=str, default="Deepfakes")
    arg = parser.parse_args()
    return arg

def dataGen():
    data = []
    labels = []
    for i in range(len(testDataSet)):
        if i % 5 != 0:
            continue
        path, label = testDataSet[i]
        if label != 1:
            continue
        img = Image.open(path)
        img = img.resize((224, 224))
        img_ndarray = np.asarray(img, dtype='float32') / 255.0
        data.append(img_ndarray)
        labels.append(label)
    return np.array(data), np.array(labels)

def main(args):
    testDataFile = open("dataSetDict_224/" + args.target + "/testDataSet.txt")
    while True:
        line = testDataFile.readline()
        if not line:
            break
        record = line.split(" ")
        testDataSet.append((record[0], int(record[1])))
    print("Test Data Paths loaded: number =", len(testDataSet))

    for i in range(10):
        print("testing epoch =", i)
        model = keras.models.load_model("models_224/" + args.model + '/epoch' + str(i + 1) + '.h5')
        model.evaluate(x=dataGen()[0], y=dataGen()[1], verbose=1)



if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)