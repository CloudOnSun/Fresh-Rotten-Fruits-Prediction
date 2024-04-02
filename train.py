import pickle
from sklearn.svm import SVC
import numpy as np


file_train_Xdata = open("train_X.pkl", 'rb')
train_Xdata = np.array(pickle.load(file_train_Xdata),dtype='object')
print(len(train_Xdata))
file_train_Xdata.close()

file_train_ydata = open("train_y.pkl", 'rb')
train_ydata = np.array(pickle.load(file_train_ydata))
print(len(train_ydata))
file_train_ydata.close()

file_test_Xdata = open("test_X.pkl", 'rb')
test_Xdata = np.array(pickle.load(file_test_Xdata), dtype='object')
print(len(test_Xdata))
file_test_Xdata.close()

file_test_ydata = open("test_y.pkl", 'rb')
test_ydata = np.array(pickle.load(file_test_ydata))
print(len(test_ydata))
file_test_ydata.close()

print(type(train_Xdata))

model = SVC(kernel='linear')
model.fit(train_Xdata, train_ydata)

pred_y = model.predict(test_Xdata)
model.accuracy(test_ydata, pred_y)
