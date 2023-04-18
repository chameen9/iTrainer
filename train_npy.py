import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import datetime

import os
import numpy as np
import tensorflow as tf

start_time = datetime.datetime.now()

# Specify the path to the directory containing the data files
data_dir = 'D:/Beng (Hons)/Project/Software/data/train/'

# Get a list of all the data files
data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]

# Initialize empty lists to store the data and labels
data = []
labels = []

# Loop over each data file and load the data
for f in data_files:
    # Load the data from the file
    d = np.load(f)

    # Extract the label from the filename
    label = os.path.basename(f).split('_')[0]

    # Append the data and label to the lists
    data.append(d)
    labels.append(label)

# Convert the data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Print the number of samples and the shape of the data arrays
print('Number of samples:', len(data))
print('Shape of data:', data.shape)
print('Shape of labels:', labels.shape)

# Convert the labels to integers
label_dict = {l: i for i, l in enumerate(np.unique(labels))}
y_train = np.array([label_dict[label] for label in labels])

# Convert the data to the format expected by the model
X_train = data.reshape(data.shape[0], 33, 3, 1)

# Convert the labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train)

# Define the model architecture
model = tf.keras.models.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(33, 3, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(label_dict), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Save the model
model.save('model.h5')


end_time = datetime.datetime.now
print('Time Tooked :', end_time-start_time)