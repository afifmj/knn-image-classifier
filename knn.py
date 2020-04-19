
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder #converting labels from string to int
from sklearn.model_selection import train_test_split #splitting the dataset
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

print("[INFO] loading images...")

#get the path of the dataset
imagePaths = list(paths.list_images(args['dataset']))

#preprocess all the images in the dataset to the same resolution of 32*32 pixels
sp = SimplePreprocessor(32,32)
sdl = SimpleDatasetLoader([sp])

#get the images and the labels
(data, labels) = sdl.load(imagePaths , verbose = 500)

#reshape the images from a 3d matrix to a 1d vector
data = data.reshape((data.shape[0], 3072))

print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

#convert the labels into integers
le = LabelEncoder()
labels = le.fit_transform(labels)

#split the dataset into training and testing data
(trainX , testX, trainY , testY) = train_test_split(data,labels,test_size = 0.25 , random_state = 42)
#X stores data   Y stores labels

#knn classifier
print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],n_jobs=args["jobs"])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX),target_names=le.classes_))