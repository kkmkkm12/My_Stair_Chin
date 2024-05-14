import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 계단을 리스트화 시켜야함
stair_images = list()        # empty list
for i in range(310):
    file = "./stairs/" + "img{0:02d}.jpg".format(i + 1)
    img = cv2.imread(file)
    if img is None:
        print(f"계단 파일 없음 : {i + 1}")
        break

    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    stair_images.append(img)

def plot_images(n_row:int, n_col:int, images:list[np.ndarray]) -> None:
    fig = plt.figure()
    (fig, ax) = plt.subplots(n_row, n_col, figsize=(n_col, n_row))
    for i in range(n_row):
        for j in range(n_col):
            axis = ax[i, j]
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)
            axis.imshow(images[i * n_col + j])
    # plt.show()
    return None

# plot_images(n_row=5, n_col=4, images=stair_images)


# 턱을 리스트화 시켜야함
chin_images = list()          # Empty List
for i in range(263):
    file = "./road_chin/" + "img{0:02d}.jpg".format(i + 1)
    img = cv2.imread(file)
    if img is None:
        print(f"턱 파일 없음 : {i + 1}")
        break

    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    chin_images.append(img)


# plot_images(n_row=11, n_col=4, images=chin_images)

# 길을 리스트화 시켜야함
road_images = list()
for i in range(290):
    file = "./road/" + "img{0:02d}.jpg".format(i + 1)
    img = cv2.imread(file)
    if img is None:
        print(f"길 파일 없음 : {i + 1}")
        break

    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    road_images.append(img)


# plot_images(n_row=3, n_col=5, images=road_images)

# X train data 만들기
X = stair_images + chin_images + road_images
y = [[1, 0, 0]]*len(stair_images) + [[0, 1, 0]]*len(chin_images) + [[0, 0, 1]]*len(road_images)

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
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=3, activation='softmax'),
])

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X = np.array(X)
y = np.array(y)
history = model.fit(x=X, y=y, epochs=100)

# 테스트 사진을 리스트 화 시킨다

test_images = list()     # Empty List
for i in range(9):
    file = "./Test/" + "img{0:02d}.jpg".format(i + 1)
    img = cv2.imread(file)
    if img is None:
        print("파일 없음")
        break

    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test_images.append(img)

test_images = np.array(test_images)
# plot_images(3, 3, test_images)

predict_images = model.predict(test_images)
print(predict_images)

fig = plt.Figure()
(fig, ax) = plt.subplots(3, 3, figsize=(10, 4))
for i in range(3):
    for j in range(3):
        axis = ax[i, j]
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
        if predict_images[i * 3 + j][0] > 0.6:
            axis.imshow(test_images[i * 3 + j])

plt.show()

model.save('stair4.h5')