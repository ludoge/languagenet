import h5py
import numpy as np
from scipy import misc
import keras
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.layers import Dropout, Input, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Nadam, Adadelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

dataname = "spectrogramsV2"

with open('trainingdata/trainingData.csv') as f:
    file_to_label = {x.split('.')[0]: x.split(',')[-1] for x in f.read().split('\n')[1:]}
size = len(list(file_to_label))

file_list = list(file_to_label)

labels = list(set(file_to_label.values()))
num_classes = len(labels)
label_to_int = {labels[i]: i for i in range(num_classes)}

print(labels)
print("Number of unique languages:")
print(num_classes)

print("Reading files...")

# creating y_data

y_data = np.zeros([size, 1], dtype=np.int8)
for i in range(size):
    y_data[i] = label_to_int[file_to_label[file_list[i]]]

print(np.shape(y_data))

# converting to one-hot

y_data = to_categorical(y_data, num_classes)

print(np.shape(y_data))

# Creating x data; this can take 1-2 minutes

try:
    with h5py.File(f'{dataname}.h5', 'r') as hf:
        x_data = hf['x_data'][:]
except (IOError, EOFError, FileNotFoundError):
    tensor_list = []
    i = 0
    a0 = np.array(misc.imread(f'trainingdata/{dataname}/{list(file_to_label)[0]}.jpg'))
    x, y = np.shape(a0)
    shape = (size, x, y, 1)
    x_data = np.zeros(shape, dtype=np.int8)
    for filename in list(file_to_label):
        arr = np.array(misc.imread(f'trainingdata/{dataname}/{filename}.jpg'))
        arr = np.reshape(arr, (x, y, 1))
        # print(arr)
        x_data[i] = arr
        i += 1
        print(i)
    print(np.shape(x_data))
    n, h, w = list(np.shape(x_data))[0], list(np.shape(x_data))[1], list(np.shape(x_data))[2]
    x_data = np.reshape(x_data, (n, h, w, 1))
    with h5py.File(f'{dataname}.h5', 'w') as hf:
        hf.create_dataset('x_data', data=x_data)

if len(list(np.shape(x_data))) < 4:
    n, h, w = list(np.shape(x_data))[0], list(np.shape(x_data))[1], list(np.shape(x_data))[2]
    x_data = np.reshape(x_data, (n, h, w, 1))

print(np.shape(x_data))

x_data = x_data.astype('float32')
x_data /= 255

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=13376)

x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=4576)

print(np.shape(x_train), np.shape(x_test), np.shape(y_train), np.shape(y_test))


batch_size = 32

in_dim = (192,192,1)
out_dim = 176

i = Input(shape=in_dim)
m = Conv2D(16, (3, 3), activation='elu', padding='same')(i)
m = MaxPooling2D()(m)
m = Conv2D(32, (3, 3), activation='elu', padding='same')(m)
m = MaxPooling2D()(m)
m = Conv2D(64, (3, 3), activation='elu', padding='same')(m)
m = MaxPooling2D()(m)
m = Conv2D(128, (3, 3), activation='elu', padding='same')(m)
m = MaxPooling2D()(m)
m = Conv2D(256, (3, 3), activation='elu', padding='same')(m)
m = MaxPooling2D()(m)
m = Flatten()(m)
m = Dense(512, activation='elu')(m)
m = Dropout(0.5)(m)
m = Dense(512, activation='elu')(m)
m = Dropout(0.5)(m)
o = Dense(out_dim, activation='softmax')(m)

model = Model(inputs=i, outputs=o)
model.summary()

data_augmentation = False # causes MemoryError

if not data_augmentation:
    model.compile(loss='categorical_crossentropy', optimizer=Nadam(lr=1e-3), metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=4, verbose=1, validation_data=(x_val, y_val))
    model.compile(loss='categorical_crossentropy', optimizer=Nadam(lr=1e-4), metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=4, verbose=1, validation_data=(x_val, y_val))
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(lr=1e-4), metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=4, verbose=1, validation_data=(x_val, y_val))
else:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    datagen.fit(x_train)

    model.compile(loss='categorical_crossentropy', optimizer=Nadam(lr=1e-3), metrics=['accuracy'])
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=2, verbose=1, validation_data=(x_val, y_val))
    model.compile(loss='categorical_crossentropy', optimizer=Nadam(lr=1e-4), metrics=['accuracy'])
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=3, verbose=1,
                        validation_data=(x_val, y_val))

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = f'keras_spectrograms_trained_model_{dataname}.h5'

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
