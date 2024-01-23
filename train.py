import argparse
import os
import time
import numpy as np
import tensorflow
import tensorflow as tf
from PIL import Image
from tensorflow import keras

trainDataSet = []

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=str, default="Deepfakes")
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch', type=int, default=32)
    parser.add_argument('-w', '--workers', type=int, default=1)
    arg = parser.parse_args()
    return arg

def model_load(target, epochNum):
    model_dir = "models_224/" + target
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if epochNum == 1:
        model = tf.keras.applications.MobileNet()
    else:
        model_path = os.path.join(model_dir, f"epoch{epochNum-1}.h5")
        model = keras.models.load_model(model_path)
    return model

def model_save(model, target, epochNum):
    model_dir = "models_224/" + target
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, f"epoch{epochNum}.h5")
    model.save(model_path)

def main(args):
    print("Start time =", time.time())

    trainDataFile = open("dataSetDict_224/" + args.target + "/trainDataSet.txt")
    while True:
        line = trainDataFile.readline()
        if not line:
            break
        record = line.split(" ")
        trainDataSet.append((record[0], int(record[1])))
    print("Training Data Paths loaded: number =", len(trainDataSet))

    data = tensorflow.data.Dataset.from_generator(dataGen, (tf.float32, tf.int32),
                                                  (tf.TensorShape([224, 224, 3]), tf.TensorShape([])))
    data = data.batch(args.batch)

    startTime = time.time()
    print("Start time =", time.time())

    for epoch in range(1, args.epochs + 1):
        model = model_load(args.target, epoch)
        model.compile(optimizer='adam',
                      loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        if args.workers > 1:
            model.fit(data, epochs=1, use_multiprocessing=True, workers=args.workers)
        else:
            model.fit(data, epochs=1)
        model_save(model, args.target, epoch)

    print("Time used =", time.time() - startTime)

def dataGen():
    for data in trainDataSet:
        path, label = data
        img = Image.open(path)
        img_ndarray = np.asarray(img, dtype='float64') / 255
        img_ndarray.resize(224, 224, 3)
        yield (img_ndarray, label)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
