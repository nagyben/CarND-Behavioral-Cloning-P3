import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D, Dropout, SpatialDropout2D

def main():
    lines = []

    datafiles = ['beta_simulator_linux/data/driving_log.csv',
                 'data/driving_log.csv']

    for file in datafiles:
        with open(file) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)

    images = []
    steering = []

    for line in lines:
        source_path_center = line[0]
        image = cv2.imread(source_path_center)
        images.append(image)
        steering_angle = float(line[3])
        steering.append(steering_angle)

        # augmentation by image flipping
        images.append(np.fliplr(image))
        steering.append(-steering_angle)

        # augmentation by using right camera image
        source_path_right = line[2]
        image = cv2.imread(source_path_right)
        images.append(image)

        steering_angle = float(line[3]) - 0.2  # modify steering input
        steering.append(steering_angle)

    X_train = np.array(images)
    y_train = np.array(steering)

    model = Nvidia()

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

    model.save('model.h5')


def LeNet():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    # model.add(SpatialDropout2D(rate=0.3))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(rate=0.3))
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model


def Nvidia():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model

if __name__ == "__main__":
    main()