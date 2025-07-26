import os


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Input, Dense


DATASET_PATH = "dataset"
num_classes = len(os.listdir(DATASET_PATH))
class_mode = "categorical"

train_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=24,
    subset="training"
)
val_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=24,
    subset="validation"
)

model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(num_classes, activation="softmax")
])

loss_function = "categorical_crossentropy"
model.compile(optimizer="adam", loss=loss_function, metrics=["accuracy"])

model.fit(train_data, validation_data=val_data, epochs=50)

test_loss, test_accuracy = model.evaluate(val_data)
model.save(f"new_clacifare_chess_neg.h5")