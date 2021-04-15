"""
Main script - model train and predict

usage with optional arguments:

python model_train.py --predict

   -h, --help   - show this help message and exit
   --train      - activate train function: True if exists, else False
   --predict    - activate predict function: True if exists, else False


author: Danilchenko Vadim
email: vndanilchenko@gmail.com
"""
import argparse
parser = argparse.ArgumentParser(description='A tutorial of argparse!')
parser.add_argument("--train", action="store_true", help="- activate train function: True if exists, else False")
parser.add_argument("--predict", action="store_true", help="- activate predict function: True if exists, else False")


import os
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.metrics import classification_report
import numpy as np

# загрузим mnist dataset
from keras.datasets import mnist

class Base_model:

    def prepare_data(self, new_data=None):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # добавим новые данные в обучающую выборку
        if new_data:
            x_train_ = list(x_train)
            y_train_ = list(y_train)
            for i in range(len(new_data)):
                x_train_.append(new_data[i][0])
                y_train_.append(new_data[i][1])

            x_train = np.asarray(x_train_)
            y_train = np.asarray(y_train_)

        # приведем в категориальный формат
        self.y_train = to_categorical(y_train, num_classes=10)
        self.y_test = to_categorical(y_test, num_classes=10)

        self.x_train = x_train
        self.x_test = x_test

        print('x_train shape:', x_train.shape,
              '\ny_train shape:', y_train.shape,
              '\nx_test shape:', x_test.shape,
              '\ny_test shape:', y_test.shape)

    def compile(self):
        # напишем архитектуру модели
        inp_ = Input(shape=(28,28,1))
        x = Conv2D(32, 4, padding='same', activation='relu', name='conv2d_1')(inp_)
        x = Conv2D(64, 2, padding='same', activation='relu', name='conv2d_2')(x)
        x = MaxPooling2D()(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu', name='dense1')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu', name='dense2')(x)
        out = Dense(10, activation='softmax', name='output')(x)

        model = Model(inp_, out)
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        if os.path.isfile('./cls/weights/best_weight.hdf5'):
            model.load_weights('./cls/weights/best_weight.hdf5')

        print(model.summary())
        self.model = model


    def train(self, new_data=None):
        self.prepare_data(new_data)
        self.compile()
        self.model.fit(self.x_train,
                  self.y_train,
                  batch_size=256,
                  epochs=10,
                  validation_data=(self.x_test, self.y_test),
                  callbacks=ModelCheckpoint('./cls/weights/best_weight.hdf5', monitor='val_accuracy', mode='max', save_best_only=True, save_weights_only=True),
                  verbose=1)

        print(classification_report(np.argmax(self.y_test, axis=1), list(np.argmax(self.model.predict(self.x_test), axis=1))))
        return True


    def predict(self, X):
        if X.shape == (28, 28):
            X = X.reshape(1, 28, 28)
        try:
            res = self.model.predict(X)
        except:
            self.compile()
            res = self.model.predict(X)
        return [int(i) for i in list(np.argmax(res, axis=1))]


if __name__ == '__main__':
    model = Base_model()
    args = parser.parse_args()
    if args.predict:
        model.compile()
        model.prepare_data()
        print('prediction:', model.predict(model.x_test[0]))
    elif args.train:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        model.train([[x_train[0], y_train[0]]])
    else:
        print("please choose optional arguments:\n" 
                "   -h, --help   - show this help message and exit\n" 
                "   --train      - activate train function: True if exists, else False\n" 
                "   --predict    - activate predict function: True if exists, else False")
