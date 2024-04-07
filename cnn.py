import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models
import tensorflow as tf


def preprocess(dataset, categories):
    root_source = f'dataset_resized/{dataset}'
    features = []
    labels = []
    for cat_idx, cat in enumerate(categories):
        source = os.path.join(root_source, cat)
        for image in os.listdir(source):
            img = load_img(os.path.join(source, image), target_size=(32, 32))
            features.append(img_to_array(img))
            labels.append(cat_idx)
    return features, labels


train_data_path = 'dataset_resized/Train'
test_data_path = 'dataset_resized/Test'
train_categories = sorted(os.listdir(train_data_path))
test_categories = sorted(os.listdir(test_data_path))

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(train_categories))
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

train_X, train_y = preprocess('Train', train_categories)
test_X, test_y = preprocess('Test', test_categories)

history = model.fit(np.array(train_X), np.array(train_y), epochs=10,
                    validation_data=(np.array(test_X), np.array(test_y)))
