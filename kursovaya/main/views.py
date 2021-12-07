from re import match
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import pandas as pd
from pandas import DataFrame
from sklearn import svm

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# Create your views here.


def main_logic(request):
    data = get_data('collegePlace.csv')
    dict = {
        'Age': data['Age'],
        'Gender': data['Gender'],
        'Stream': data['Stream'],
        'Internships': data['Internships'],
        'CGPA': data['CGPA'],
        'Hostel': data['Hostel'],
        'HistoryOfBacklogs': data['HistoryOfBacklogs'],
        'PlacedOrNot': data['PlacedOrNot'],
    }
    title = ''
    action = ''
    result = ''
    description = ''

    if request.GET['action'] == "sum":
        if request.GET['data'] == "Internships":
            title = 'Стажировки'
            action = 'Сумма'
            result = pd.Series(dict['Internships']).sum()
            description = 'Данные сформированы по стажировкам студентов. Выше мы видим операцию над данными а ' \
                          'снизу видим график по всем студентам '
        elif request.GET['data'] == "CGPA":
            title = 'CGPA'
            action = 'Сумма'
            result = pd.Series(dict['CGPA']).sum()
            description = 'Данные сформированы по баллам студентов. Выше мы видим операцию над данными а ' \
                          'снизу видим график по всем студентам '
        elif request.GET['data'] == "Hostel":
            title = 'Наличие жилья'
            action = 'Сумма'
            result = pd.Series(dict['Hostel']).sum()
            description = 'Данные сформированы по наличию жилья студентов. Выше мы видим операцию над данными ' \
                          'а снизу видим график по всем студентам '
        elif request.GET['data'] == "HistoryOfBacklogs":
            title = 'История невполненных работ'
            action = 'Сумма'
            result = pd.Series(dict['HistoryOfBacklogs']).sum()
            description = 'Данные сформированы по истории невыполненных работ студентов. Выше мы видим ' \
                          'операцию над данными а снизу видим график по всем студентам '
        elif request.GET['data'] == "PlacedOrNot":
            title = 'Размещение'
            action = 'Сумма'
            result = pd.Series(dict['PlacedOrNot']).sum()
            description = 'Данные сформированы по размещению студентов. Выше мы видим операцию над данными а ' \
                          'снизу видим график по всем студентам '
    elif request.GET['action'] == "average":
        if request.GET['data'] == "Internships":
            title = 'Стажировки'
            action = 'Среднее'
            result = pd.Series(dict['Internships']).mean()
            description = 'Данные сформированы по стажировкам студентов. Выше мы видим операцию над данными а ' \
                          'снизу видим график по всем студентам '
        elif request.GET['data'] == "CGPA":
            title = 'CGPA'
            action = 'Среднее'
            result = pd.Series(dict['CGPA']).mean()
            description = 'Данные сформированы по баллам студентов. Выше мы видим операцию над данными а ' \
                          'снизу видим график по всем студентам '
        elif request.GET['data'] == "Hostel":
            title = 'Наличие жилья'
            action = 'Среднее'
            result = pd.Series(dict['Hostel']).mean()
            description = 'Данные сформированы по наличию жилья студентов. Выше мы видим операцию над данными ' \
                          'а снизу видим график по всем студентам '
        elif request.GET['data'] == "HistoryOfBacklogs":
            title = 'История невполненных работ'
            action = 'Среднее'
            result = pd.Series(dict['HistoryOfBacklogs']).mean()
            description = 'Данные сформированы по истории невыполненных работ студентов. Выше мы видим ' \
                          'операцию над данными а снизу видим график по всем студентам '
        elif request.GET['data'] == "PlacedOrNot":
            title = 'Размещение'
            action = 'Среднее'
            result = pd.Series(dict['PlacedOrNot']).mean()
            description = 'Данные сформированы по размещению студентов. Выше мы видим операцию над данными а ' \
                          'снизу видим график по всем студентам '
    elif request.GET['action'] == "max":
        if request.GET['data'] == "Internships":
            title = 'Стажировки'
            action = 'Максимум'
            result = pd.Series(dict['Internships']).max()
            description = 'Данные сформированы по стажировкам студентов. Выше мы видим операцию над данными а ' \
                          'снизу видим график по всем студентам '
        elif request.GET['data'] == "CGPA":
            title = 'CGPA'
            action = 'Максимум'
            result = pd.Series(dict['CGPA']).max()
            description = 'Данные сформированы по баллам студентов. Выше мы видим операцию над данными а ' \
                          'снизу видим график по всем студентам '
        elif request.GET['data'] == "Hostel":
            title = 'Наличие жилья'
            action = 'Максимум'
            result = pd.Series(dict['Hostel']).max()
            description = 'Данные сформированы по наличию жилья студентов. Выше мы видим операцию над данными ' \
                          'а снизу видим график по всем студентам '
        elif request.GET['data'] == "HistoryOfBacklogs":
            title = 'История невполненных работ'
            action = 'Максимум'
            result = pd.Series(dict['HistoryOfBacklogs']).max()
            description = 'Данные сформированы по истории невыполненных работ студентов. Выше мы видим ' \
                          'операцию над данными а снизу видим график по всем студентам '
        elif request.GET['data'] == "PlacedOrNot":
            title = 'Размещение'
            action = 'Максимум'
            result = pd.Series(dict['PlacedOrNot']).max()
            description = 'Данные сформированы по размещению студентов. Выше мы видим операцию над данными а ' \
                          'снизу видим график по всем студентам '
    elif request.GET['action'] == "min":
        if request.GET['data'] == "Internships":
            title = 'Стажировки'
            action = 'Минимум'
            result = pd.Series(dict['Internships']).min()
            description = 'Данные сформированы по стажировкам студентов. Выше мы видим операцию над данными а ' \
                          'снизу видим график по всем студентам '
        elif request.GET['data'] == "CGPA":
            title = 'CGPA'
            action = 'Минимум'
            result = pd.Series(dict['CGPA']).min()
            description = 'Данные сформированы по баллам студентов. Выше мы видим операцию над данными а ' \
                          'снизу видим график по всем студентам '
        elif request.GET['data'] == "Hostel":
            title = 'Наличие жилья'
            action = 'Минимум'
            result = pd.Series(dict['Hostel']).min()
            description = 'Данные сформированы по наличию жилья студентов. Выше мы видим операцию над данными ' \
                          'а снизу видим график по всем студентам '
        elif request.GET['data'] == "HistoryOfBacklogs":
            title = 'История невполненных работ'
            action = 'Минимум'
            result = pd.Series(dict['HistoryOfBacklogs']).min()
            description = 'Данные сформированы по истории невыполненных работ студентов. Выше мы видим ' \
                          'операцию над данными а снизу видим график по всем студентам '
        elif request.GET['data'] == "PlacedOrNot":
            title = 'Размещение'
            action = 'Минимум'
            result = pd.Series(dict['PlacedOrNot']).min()
            description = 'Данные сформированы по размещению студентов. Выше мы видим операцию над данными а ' \
                          'снизу видим график по всем студентам '
    elif request.GET['action'] == "med":
        if request.GET['data'] == "Internships":
            title = 'Стажировки'
            action = 'Медиана'
            result = pd.Series(dict['Internships']).median()
            description = 'Данные сформированы по стажировкам студентов. Выше мы видим операцию над данными а ' \
                          'снизу видим график по всем студентам '
        elif request.GET['data'] == "CGPA":
            title = 'CGPA'
            action = 'Медиана'
            result = pd.Series(dict['CGPA']).median()
            description = 'Данные сформированы по баллам студентов. Выше мы видим операцию над данными а ' \
                          'снизу видим график по всем студентам '
        elif request.GET['data'] == "Hostel":
            title = 'Наличие жилья'
            action = 'Медиана'
            result = pd.Series(dict['Hostel']).median()
            description = 'Данные сформированы по наличию жилья студентов. Выше мы видим операцию над данными ' \
                          'а снизу видим график по всем студентам '
        elif request.GET['data'] == "HistoryOfBacklogs":
            title = 'История невполненных работ'
            action = 'Медиана'
            result = pd.Series(dict['HistoryOfBacklogs']).median()
            description = 'Данные сформированы по истории невыполненных работ студентов. Выше мы видим ' \
                          'операцию над данными а снизу видим график по всем студентам '
        elif request.GET['data'] == "PlacedOrNot":
            title = 'Размещение'
            action = 'Медиана'
            result = pd.Series(dict['PlacedOrNot']).median()
            description = 'Данные сформированы по размещению студентов. Выше мы видим операцию над данными а ' \
                          'снизу видим график по всем студентам '
    elif request.GET['action'] == "klaster":
        X = pd.DataFrame(data=get_data('collegePlace.csv'), columns=['Internships', 'CGPA', 'HistoryOfBacklogs'])
        Y = data['Stream']
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10)
        SVC_model = svm.SVC()
        KNN_model = KNeighborsClassifier(n_neighbors=5)
        SVC_model.fit(X_train, y_train)
        KNN_model.fit(X_train, y_train)
        SVC_prediction = SVC_model.predict(X_test)
        KNN_prediction = KNN_model.predict(X_test)

        matrix = list()
        for i in range(len(confusion_matrix(SVC_prediction, y_test))):
            for j in range(6):
                matrix.append(str(confusion_matrix(SVC_prediction, y_test)[i][j]))

        out = {
            'title': 'Оценка точности — простейший вариант оценки работы классификатора',
            'SVC_prediction': 'SVC prediction = ' + str(accuracy_score(SVC_prediction, y_test)),
            'KNN_prediction': 'KNN prediction = ' + str(accuracy_score(KNN_prediction, y_test)),
            'description': 'Матрица неточности',
            'matrix': list(matrix),
            'description2': 'Отчёт о классификации',
            'report': classification_report(KNN_prediction, y_test)
        }
        return JsonResponse(out, safe=True)


    out_dict = {
        'title': title,
        'action': action,
        'result': str(result),
        'description': description,
        'genres': list(data['Gender']),
        'ages': list(data['Age']),
        'streams': list(data['Stream']),
        'data': list(data[request.GET['data']])
    }
    return JsonResponse(out_dict, safe=True)


def get_data(filename) -> DataFrame:
    return pd.read_csv(filename)
