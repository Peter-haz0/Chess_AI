import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


model = load_model("new_clacifare_chess_neg.h5")
img_path = "Qb.png"
target_size = (128, 128)
img = image.load_img(img_path, target_size=target_size)
img_array = image.img_to_array(img)
img_array = img_array/255
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
num_classes = model.output_shape[-1]

# if num_classes == 1:
# prob = predictions[0][0]
# if prob >= 0.5:
#     print("black")
# elif prob < 0.5:
#     print("white")

class_index = np.argmax(predictions[0])
prob = predictions[0][class_index]

label = ["Чёрный слон",
         "Белый слон",
         "Чёрный король",
         "Белый король",
         "Чёрный конь",
         "Белый конь",
         "Чёрная пешка",
         "Белая пешка",
         "Чёрный ферзь",
         "Белый ферзь",
         "Чёрныя ладья",
         "Белая ладья"]

print(f"предсказаный класс: {label[class_index]}")
print(f"с вероятностью: {prob:.2f}")

# class_ind = train_data.class_indices
# print(class_ind)

