"""
Сервис предназначен для выполнения предсказания класса mnist

примеры выполнения запроса к Сервису в get_prediction.py

Эндпоинты:
/get_response - предсказание класса изображения mnist
    :param data: (28, 28) numpy.ndarray
    :return: {'prediction': [class]} dict

/train_model - добавление обучающих данных в выборку и запуск переобучения
    :params: [(28, 28) numpy.ndarray, class int]
    :return: {'train_result': True}

author: Danilchenko Vadim
email: vndanilchenko@gmail.com
"""


import flask, json, requests
import numpy as np
from model_train import Base_model


app = flask.Flask(__name__)

@app.route('/')
def get_response():
    return 'hello, stranger! this is not what you are looking for'

@app.route('/get_response', methods=['POST'])
def get_response_():
    """
    эндпоинт для получения предсказания модели
    params: (28, 28) numpy.ndarray
    :return: {'prediction': [class]} dict
    """
    params = flask.request.get_json(force=True, silent=True)

    if params and 'data' in params:
        res = model.predict(np.asarray(params['data']))
    else:
        res = None

    return flask.Response(response=json.dumps({'prediction': res}), status=200, content_type='application/json; charset=utf-8')

@app.route('/train_model', methods=['POST'])
def train_model_():
    """
    эндпоинт дообучения модели
    :params: [(28, 28) numpy.ndarray, class int]
    :return: {'train_result': True}
    """
    params = flask.request.get_json(force=True, silent=True)

    if params and 'data_train' in params:
        res = model.train(params['data_train'])
    else:
        res = False

    return flask.Response(response=json.dumps({'train_result': res}), status=200, content_type='application/json; charset=utf-8')


if __name__ == '__main__':
    model = Base_model()
    model.compile()
    app.run(host='127.0.0.1', port=8080, debug=True)