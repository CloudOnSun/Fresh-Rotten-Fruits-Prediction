import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import cv2
from PIL import Image
from sklearn.svm import SVC
from sklearn.metrics import recall_score

train_data_path = 'dataset_resized/Train'
test_data_path = 'dataset_resized/Test'
train_data_folders = []
test_data_folders = []
for folders in os.listdir(train_data_path):
    train_data_folders.append(os.path.join(train_data_path, folders))
for folders in os.listdir(test_data_path):
    test_data_folders.append(os.path.join(test_data_path, folders))

train_data_folders.sort()
test_data_folders.sort()
print(len(train_data_folders), len(test_data_folders))

train_categories = []
test_categories = []
for category in train_data_folders:
    train_categories.append(category[22:])
for category in test_data_folders:
    test_categories.append(category[21:])
print(train_categories, test_categories)


# train_fruits = []
# test_fruits = []
# for category in train_categories[:len(train_categories) // 2]:
#     train_fruits.append(category[5:])
# for category in test_categories[:len(test_categories) // 2]:
#     test_fruits.append(category[5:])
# print(train_fruits, test_fruits)

def preprocess(dataset):
    root_source = f'dataset_resized/{dataset}'
    features = []
    labels = []
    if dataset == 'Train':
        categories = train_categories
    else:
        categories = test_categories
    for cat in categories:
        source = os.path.join(root_source, cat)
        for image in os.listdir(source):
            img = Image.open(os.path.join(source, image))
            img_feature = np.array(img.resize((64, 64))).flatten()
            features.append(img_feature)
            labels.append(categories.index(cat))
    return features, labels


train_X, train_y = preprocess('Train')
test_X, test_y = preprocess('Test')

print(len(train_X), len(train_y))
print(len(test_X), len(test_y))

model = SVC(kernel='linear', verbose=True, max_iter=10)
model.fit(train_X, train_y)

pred_y = model.predict(test_X)
print(recall_score(test_y, pred_y))

# Saving the train and test data in working directory
train_Xdata = 'train_X.pkl'
train_ydata = 'train_y.pkl'
with open(train_Xdata, 'wb') as file:
    pickle.dump(train_X, file)

with open(train_ydata, 'wb') as file:
    pickle.dump(train_y, file)

test_Xdata = 'test_X.pkl'
test_ydata = 'test_y.pkl'
with open(test_Xdata, 'wb') as file:
    pickle.dump(test_X, file)

with open(test_ydata, 'wb') as file:
    pickle.dump(test_y, file)
