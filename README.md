Сервис предназначен для выполнения предсказания класса датасета mnist

основная функция main.py

примеры выполнения запроса к Сервису в get_prediction.py

Эндпоинты:

<xml>

    /get_response - предсказание класса изображения mnist
        :param data: (28, 28) numpy.ndarray
        :return: {'prediction': [class]} dict

</xml>

<xml>

    /train_model - добавление обучающих данных в выборку и запуск переобучения
        :params: [(28, 28) numpy.ndarray, class int]
        :return: {'train_result': True}

</xml>

predict example

![image info](./screenshots/get_prediction%20--predict.png)

train example

![image info](./screenshots/get_prediction%20--train%20part1.png)

![image info](./screenshots/get_prediction%20--train%20part2.png)

![image info](./screenshots/get_prediction%20--train%20part3.png)








author: Danilchenko Vadim

email: vndanilchenko@gmail.com