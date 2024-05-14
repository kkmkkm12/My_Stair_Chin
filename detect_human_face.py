import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# 사람의 얼굴을 리스트화 시켜야함
face_images = list()        # empty list
for i in range(15):
    file = "./faces/" + "img{0:02d}.jpg".format(i + 1)
    img = cv2.imread(file)
    if img is None:
        print("파일 없음")
        break

    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_images.append(img)

def plot_images(n_row:int, n_col:int, images:list[np.ndarray]) -> None:
    fig = plt.figure()
    (fig, ax) = plt.subplots(n_row, n_col, figsize=(n_col, n_row))
    for i in range(n_row):
        for j in range(n_col):
            axis = ax[i, j]
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)
            axis.imshow(images[i * n_col + j])
    #plt.show()
    return None

plot_images(n_row=3, n_col=5, images=face_images)


# 동물의 얼굴을 리스트화 시켜야함
animal_images = list()          # Empty List
for i in range(15):
    file = "./animals/" + "img{0:02d}.jpg".format(i + 1)
    img = cv2.imread(file)
    if img is None:
        print("파일 없음")
        break

    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    animal_images.append(img)


plot_images(n_row=3, n_col=5, images=animal_images)

# X train data 만들기
X = face_images + animal_images
y = [[1, 0]]*len(face_images) + [[0, 1]]*len(animal_images)

# CNN 만들기
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(input_shape=(64, 64, 3), kernel_size=(3, 3), filters=32),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=32),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=32),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),

    # Neural Network
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=2, activation='softmax'),
])

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X = np.array(X)
y = np.array(y)
history = model.fit(x=X, y=y, epochs=500)

# 테스트 얼굴을 리스트 화 시킨다

example_images = list()     # Empty List
for i in range(10):
    file = "./examples/" + "img{0:02d}.jpg".format(i + 1)
    img = cv2.imread(file)
    if img is None:
        print("파일 없음")
        break

    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    example_images.append(img)

example_images = np.array(example_images)
plot_images(2, 5, example_images)

predict_images = model.predict(example_images)
print(predict_images)

fig = plt.Figure()
(fig, ax) = plt.subplots(2, 5, figsize=(10, 4))
for i in range(2):
    for j in range(5):
        axis = ax[i, j]
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
        if predict_images[i * 5 + j][0] > 0.6:
            axis.imshow(example_images[i * 5 + j])

plt.show()