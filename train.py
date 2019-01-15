import csv
import cv2
import numpy as np
import sklearn
import sklearn.model_selection

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D, Dropout, SpatialDropout2D
from random import shuffle


def main():
    lines = []

    datafiles = ['data/driving_log.csv']

    for file in datafiles:
        with open(file) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)

    model = Nvidia(input_shape=(160, 320, 3))

    model.compile(loss='mse', optimizer='adam')
    X_train, y_train = get_all_images(lines)
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

    # train_samples, validation_samples = sklearn.model_selection.train_test_split(lines, test_size=0.2)
    #
    # train_generator = image_generator(train_samples)
    # validation_generator = image_generator(validation_samples)
    # model.fit_generator(train_generator, steps_per_epoch=len(train_samples), validation_data=validation_generator,
    #                     validation_steps=len(validation_samples))

    model.save('model.h5')


def get_all_images(samples):
    images = []
    steering = []

    for line in samples:
        source_path_center = line[0]
        image = cv2.imread(source_path_center)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        steering_angle = float(line[3])
        steering.append(steering_angle)

        # augmentation by image flipping
        images.append(np.fliplr(image))
        steering.append(-steering_angle)

    X_train = np.array(images)
    y_train = np.array(steering)

    return (X_train, y_train)


def image_generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            steering = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                image = cv2.imread(name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                angle = float(batch_sample[3])
                images.append(image)
                steering.append(angle)

                # augment by flipping horizontally
                image = np.fliplr(image)
                images.append(image)
                steering.append(-angle)

            X_train = np.array(images)
            y_train = np.array(steering)
            yield sklearn.utils.shuffle(X_train, y_train)


def LeNet(input_shape):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
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


def Nvidia(input_shape):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(rate=0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    return model


if __name__ == "__main__":
    main()
