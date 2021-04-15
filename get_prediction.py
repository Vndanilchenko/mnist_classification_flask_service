"""
Interact with Service using REST API

usage with optional arguments:

python get_prediction.py --predict

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


import json, requests, numpy
from model_train import Base_model
from keras.datasets import mnist

def get_response(X):
    """
    функция предсказания класс изображения mnist
    :param X: (28, 28) numpy.ndarray
    :return: {'prediction': [class]} dict
    """
    data = {'data': X.tolist() if isinstance(X, numpy.ndarray) else X}
    res = requests.post('http://127.0.0.1:8080/get_response', data=json.dumps(data))

    return res.json()

def retrain_model(X):
    """
    функция добавления обучающих данных в выборку и запуск переобучения
    :param X: [(28, 28) numpy.ndarray, class int]
    :return: {'train_result': True}
    """
    data = {'data_train': X}
    res = requests.post('http://127.0.0.1:8080/train_model', data=json.dumps(data))

    return res.json()


if __name__ == '__main__':
    model = Base_model()

    args = parser.parse_args()
    if args.predict:
        model.prepare_data()
        print(get_response(model.x_test[0]), '\ntrue value', numpy.argmax(model.y_test[0]))
    elif args.train:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        model.train([[x_train[0], y_train[0]]])
    else:
        print("please choose optional arguments:\n" 
                "   -h, --help   - show this help message and exit\n" 
                "   --train      - activate train function: True if exists, else False\n" 
                "   --predict    - activate predict function: True if exists, else False")
