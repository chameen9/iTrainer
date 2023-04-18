import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import datetime

start_time = datetime.datetime.now()

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=30,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

train_data = train_datagen.flow_from_directory('D:\Beng (Hons)\Project\Software\dataset',
                                               target_size=(224, 224),
                                               batch_size=32,
                                               class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory('D:\Beng (Hons)\Project\Software\dataset',
                                             target_size=(224, 224),
                                             batch_size=32,
                                             class_mode='categorical')

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data, epochs=75, batch_size=32, validation_data=test_data)

test_loss, test_acc = model.evaluate(test_data, verbose=2)

print('')
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

end_time = datetime.datetime.now()
print('Time Tooked:', end_time-start_time)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

model.save('Image_Model_for_shots')
