from joblib import load
import numpy as np

clf = load('medical.joblib')
Temp = input('Ведите температуру тела:\n')
Weight = input('Ведите вес:\n')
Height = input('Ведите рост:\n')
Diag = input('Ведите диагноз по МКБ10:\n')
BMI = int(Weight)/(int(Height)/100)**2

diag_list = [
             'A', 'B', 'C', 'D', 'E',
             'G', 'H', 'I', 'J', 'K', 
             'L','M', 'N', 'O', 'Q', 
             'R', 'S', 'T', 'V', 'Z'
            ]
Diag_cat = diag_list.index(Diag[0])


predict = clf.predict(np.array([[Temp, Weight, Height, BMI, Diag_cat]]))
if predict[0] == 0:
    print("\nНизкая вероятность летального исхода")
else:
    print("\nВысокая вероятность летального исхода")