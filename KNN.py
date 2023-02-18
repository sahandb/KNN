from random import randrange
import numpy as np
import glob
from numpy import linalg as la
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import operator
from mpl_toolkits.mplot3d import Axes3D


def gridDisplay(image_list, title):
    fig1, axes_array = plt.subplots(17, 10)
    fig1.suptitle(title)
    fig1.set_size_inches(17, 10)
    k = 0
    for row in range(17):
        for col in range(10):
            im = np.array(Image.fromarray(image_list[k]).resize((64, 64)))
            image_plot = axes_array[row][col].imshow(im, cmap='gray')
            axes_array[row][col].axis('off')
            k = k + 1
    plt.show()


image = []
flattened_images = []
names = []
labels = []


def labeling(nm):  # Labeld Picture
    if 'AN' in nm:
        return 1
    elif 'DI' in nm:
        return 2
    elif 'FE' in nm:
        return 3
    elif 'HA' in nm:
        return 4
    elif 'NE' in nm:
        return 5
    elif 'SA' in nm:
        return 6
    elif 'SU' in nm:
        return 7


def readFiles(path):
    faces = pd.DataFrame()
    for file in glob.iglob(path):  # './Jaffe/*.tiff'
        face = Image.open(file)
        face = face.resize((64, 64))
        faces = faces.append(pd.Series(np.array(face).flatten(), name=labeling(file)))
    return np.array(faces.values), faces.index


mat, index = readFiles('./Jaffe/*.tiff')

# reade images
for filename in glob.glob('jaffe/*.tiff'):
    im = Image.open(filename)
    names.append(im.filename)
    im = np.asarray(im.resize((64, 64)))
    im = np.asarray(im, dtype=float) / 255.0
    image.append(im)

# labels
for name in names:
    if 'AN' in name:
        labels.append(1)
    if 'DI' in name:
        labels.append(2)
    if 'FE' in name:
        labels.append(3)
    if 'HA' in name:
        labels.append(4)
    if 'NE' in name:
        labels.append(5)
    if 'SA' in name:
        labels.append(6)
    if 'SU' in name:
        labels.append(7)

for i in range(len(image)):
    u = image[i].flatten()
    flattened_images.append(u)


def project(data, k):
    A = np.array(data).T
    mean = np.mean(A, axis=0)
    zeroMean = A - mean
    cv = np.cov(zeroMean)
    U, S, V = la.svd(cv)
    dK = U[:, :k]
    projected = zeroMean.T @ dK
    return projected


dd = project(flattened_images, 2)
ddd = project(flattened_images, 3)

# stack x and y for dd
ddy = np.hstack((dd, np.array(labels).reshape(170, 1)))

# stack x and y for ddd
dddy = np.hstack((ddd, np.array(labels).reshape(170, 1)))


# split k fold
def crossValidationSplit(dataSet, nFold):
    dataSetSplit = list()
    dataSetCopy = list(dataSet).copy()
    foldSize = int(len(dataSet) / nFold)
    for _ in range(nFold):
        fold = list()
        while len(fold) < foldSize:
            index = randrange(len(dataSetCopy))
            fold.append(dataSetCopy.pop(index))
        dataSetSplit.append(fold)
    return dataSetSplit


# Calculate accuracy percentage
def accuracyMetric(actual, predicted):
    correct = 0
    for me in range(len(actual)):
        if actual[me] == predicted[me]:
            correct += 1
        return correct / float(len(actual)) * 100


# Evaluate alg using crossValidationSplit
def evaluateAlg(dataSet, alg, nFold, *args):
    folds = crossValidationSplit(dataSet, nFold)
    scores = list()
    for idx, fold in enumerate(folds):
        trainSet = list()
        for n_idx, n_fold in enumerate(folds):
            if n_idx != idx:
                trainSet.extend(n_fold)
        # print(len(trainSet))
        # exit()
        # testSet = fold[-1]
        testSet = list()
        for row in fold:
            rowCopy = list(row)
            testSet.append(rowCopy)
            rowCopy[-1] = None
        predicted = alg(trainSet, testSet, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracyMetric(actual, predicted)
        scores.append(accuracy)
    return scores


# Calculate Euclidean Dist
def euclideanDistance(row1, row2):
    distance = 0
    for m in range(len(row1) - 1):
        distance += (row1[m] - row2[m]) ** 2
    return np.sqrt(distance)


# Locate the most similar neighbors
def getNeighbors(train, testRow, k):
    distances = list()
    for trainRow in train:
        dist = euclideanDistance(trainRow, testRow)
        distances.append((trainRow, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = list()
    for n in range(k):
        neighbors.append(distances[n][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)  # Sorting it based on votes
    return sortedVotes[0][0]  # Please note we need the class for the top voted class, hence [0][0]#


def predictClassification(train, testRow, k):
    predictions = list()
    neighbors = getNeighbors(train, testRow, k)
    outputVal = getResponse(neighbors)
    predictions.append(outputVal)
    return predictions


# KNN alg
def KNN(train, test, k):
    predictions = list()
    for testRow in test:
        output = predictClassification(train, testRow, k)
        predictions.append(output)
    return predictions


def plot_2d(d2, index):
    laabels = {1: 'AN', 2: 'DI', 3: 'FE', 4: 'HA', 5: 'NE', 6: 'SA', 7: 'SU'}
    for j in range(7):
        items = []
        for ii in range(len(index)):
            if index[ii] == j:
                items.append(ii)
        x = np.array(d2[items])
        x1 = x[:, 0]
        x2 = x[:, 1]
        if len(x) > 0:
            plt.scatter(x1, x2, label=laabels[j])
    plt.title('2D')
    plt.legend()
    plt.show()


def plot_3d(d3, index):
    laabels = {1: 'AN', 2: 'DI', 3: 'FE', 4: 'HA', 5: 'NE', 6: 'SA', 7: 'SU'}
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for j in range(7):
        items = []
        for ii in range(len(index)):
            if index[ii] == j:
                items.append(ii)
        x = np.array(d3[items])
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        if len(x) > 0:
            ax.scatter(x1, x2, x3, zdir='z', s=20, label=laabels[j], depthshade=True)
    plt.title('3D')
    plt.legend()
    plt.show()


#######################################

scoresdd = evaluateAlg(ddy, KNN, 10, 1)
print('scores dd %s' % scoresdd)
print('mean accuracy dd %3f%%' % (sum(scoresdd) * 10 / 17))
plot_2d(dd, index)  # plot 2d

scoresddd = evaluateAlg(dddy, KNN, 10, 1)
print('scores ddd %s' % scoresddd)
print('mean accuracy ddd %3f%%' % (sum(scoresddd) * 10 / 17))
plot_3d(ddd, index)  # plot 3d
