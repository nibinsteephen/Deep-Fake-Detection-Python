import argparse
import numpy as np
from PIL import Image
from tensorflow import keras

testDataSet = []

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=str, default="Deepfakes")
    arg = parser.parse_args()
    return arg

def dataGen():
    for data in testDataSet:
        path, label = data
        img = Image.open(path)
        img_ndarray = np.asarray(img, dtype='float32') / 255.0
        img_ndarray = img_ndarray.reshape((1, 224, 224, 3))
        yield (img_ndarray, label)

def main(args):
    testDataFile = open("dataSetDict_224/" + args.target + "/testDataSet.txt")
    while True:
        line = testDataFile.readline()
        if not line:
            break
        record = line.split(" ")
        testDataSet.append((record[0], int(record[1])))
    print("Test Data Paths loaded: number =", len(testDataSet))

    print("Testing epoch =", 10, '\n')
    model = keras.models.load_model("models_224/" + args.target + '/epoch10.h5')
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])
    test_data = np.array(list(dataGen())).astype(object)
    test_images = np.vstack(test_data[:, 0])
    test_labels = np.array(test_data[:, 1])
    model.evaluate(test_images, test_labels, verbose=1)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
