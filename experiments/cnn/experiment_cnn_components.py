from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('th')


# Dataset
# Get classes which have at least 70 examples
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

num_samples, h, w = lfw_people.images.shape
X = lfw_people.data
y = lfw_people.target
num_classes = lfw_people.target_names.shape[0]

X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y,
                                                            test_size=0.3,
                                                            random_state=4246)


def shapeData(data, h=h, w=w):
    return data.reshape(data.shape[0], 1, h, w).astype('float32')


def normalize(data):
    return data / 255

X_train = normalize(shapeData(X_train))
X_test = normalize(shapeData(X_test))
y_train = np_utils.to_categorical(y_train_raw)
y_test = np_utils.to_categorical(y_test_raw)


# True Objective Model
def baseline_model(dropout_rate=0.25,
                   num_features=32,
                   feature_size=5,
                   pool_size=(2, 2),
                   fully_connected_size=256):
    """Baseline model with 1 convolution, 1 max pooling and 2 fully connected layers"""
    model = Sequential()
    model.add(Convolution2D(num_features, feature_size, feature_size,
                            border_mode='valid', input_shape=(1, h, w),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(fully_connected_size, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


# Objective function
def cnn_accuracy(x, verbose=1, print_summary=True, print_accuracy=False):
    print(x)
    model = baseline_model(dropout_rate=float(x[0, 0]),
                           num_features=int(x[0, 2]),
                           feature_size=int(x[0, 3]),
                           pool_size=(int(x[0, 4]), int(x[0, 4])))

    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              nb_epoch=int(x[0, 1]),
              batch_size=50,
              verbose=verbose)
    if print_summary:
        model.summary()

    if print_accuracy:
        print(1 - model.evaluate(X_test, y_test)[1])
        print()
    return 1 - model.evaluate(X_test, y_test)[1]  # returns 1 - accuracy %


# tweakable version of cnn_accuracy
def cnn_accuracy_base(verbose, summary, accuracy):
    return lambda x: cnn_accuracy(x,
                                  verbose=verbose,
                                  print_summary=summary,
                                  print_accuracy=accuracy)


# Objection domain
domain = [
    {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.01, 0.99)},
    {'name': 'num_epoch', 'type': 'discrete', 'domain': range(10, 41, 10)},
    {'name': 'num_features', 'type': 'discrete', 'domain': range(10, 101, 10)},
    {'name': 'feature_size', 'type': 'discrete', 'domain': range(2, 6)},
    {'name': 'pool_size', 'type': 'discrete', 'domain': range(1, 6)}]
#   {'name': 'fully_connected_size', 'type': 'discrete', 'domain': range(num_classes,301,50)}]