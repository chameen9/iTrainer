import datetime
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Dropout
from keras.utils import to_categorical

start_time = datetime.datetime.now()
image_size = (128, 128)  # set the desired image size

# Load and preprocess the drive shot data
drive_images = []
drive_key_points = []

for i in range(70):
    # Load the image
    img_path = f"D:/Beng (Hons)/Project/Software/dataset/drive/{i}.png"
    img = cv2.imread(img_path)
    img = cv2.resize(img, image_size)  # resize the image
    drive_images.append(img)

    # Load the key points
    key_points_path = f"D:/Beng (Hons)/Project/Software/output/drive/{i}.npy"
    key_points_data = np.load(key_points_path)
    drive_key_points.append(key_points_data)

drive_images = np.array(drive_images)
drive_key_points = np.array(drive_key_points)

# Load and preprocess the sweep shot data
sweep_images = []
sweep_key_points = []

for i in range(70):
    # Load the image
    img_path = f"D:/Beng (Hons)/Project/Software/dataset/sweep/{i}.png"
    img = cv2.imread(img_path)
    img = cv2.resize(img, image_size)  # resize the image
    sweep_images.append(img)

    # Load the key points
    key_points_path = f"D:/Beng (Hons)/Project/Software/output/sweep/{i}.npy"
    key_points_data = np.load(key_points_path)
    sweep_key_points.append(key_points_data)

sweep_images = np.array(sweep_images)
sweep_key_points = np.array(sweep_key_points)

# Combine the drive and sweep shot data
images = np.concatenate([drive_images, sweep_images])
key_points = np.concatenate([drive_key_points, sweep_key_points])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, key_points, test_size=0.2,
                                                                        random_state=42)

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(33 * 3),
    Reshape((33, 3))
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto',
                                                   min_delta=0.0001)

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test),
                              callbacks=[early_stopping])

# Evaluate the model on the test set
loss, mae = model.evaluate(X_test, y_test)

print('')
print('-----------------------------------------')
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {mae}")

# Save the model
# model.save('drive_sweep_model.h5')

end_time = datetime.datetime.now()
print('Time Tooked:',end_time-start_time)