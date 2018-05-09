from sklearn import metrics
import csv

volume_systole_predict = []
volume_diastole_predict = []
volume_true = []
volume_systole_true = []
volume_diastole_true = []

with open('r2/volume_diastole.csv', 'rt') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        volume_diastole_predict.append(float(row['Volume']))

with open('r2/volume_systole.csv', 'rt') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        volume_systole_predict.append(float(row['Volume']))

with open('r2/solution.csv', 'rt') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        volume_true.append(row['Volume'])

    for j in range(0, len(volume_true)):
        if (j % 2 == 0):
            volume_diastole_true.append(float(volume_true[j]))
        else:
            volume_systole_true.append(float(volume_true[j]))

diastole_r2 = metrics.r2_score(volume_diastole_true, volume_diastole_predict)
systole_r2 = metrics.r2_score(volume_systole_true, volume_diastole_predict)

print(diastole_r2)
print(systole_r2)