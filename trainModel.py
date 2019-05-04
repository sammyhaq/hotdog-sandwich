import ImageTools
from sklearn.model_selection import train_test_split


def network(img_size):
    # CNN based off of https://github.com/commaai/research/blob/master/train_steering_model.py
    net = Sequential()
    net.add(Conv2D(8, 5, 5, border_mode='valid', input_shape=(img_size, img_size, 1)))
    net.add(Dropout(0.5))
    net.add(Activation('relu'))

    net.add(Conv2D(16, 3, 3))
    net.add(Dropout(0.5))
    net.add(Activation('relu'))

    net.add(Conv2D(32, 3, 3))
    net.add(Dropout(0.5))
    net.add(Activation('relu'))

    net.add(Flatten())
    net.add(Dense(240))

    net.add(Activation('relu'))
    net.add(Dense(120))

    net.add(Dense(2))

    net.add(Activation('softmax'))

    return net

def main():
    img_size = 64
    classSize = 1000

    # Loading Data
    hotdog_files = ImageTools.parseImagePaths('./img/hotdog/')
    food_files = ImageTools.parseImagePaths('./img/food/')
    sandwich_files = ImageTools.parseImagePaths('./img/sandwiches/')

    food_x, food_y = ImageTools.expandClass(food_files, 0, classSize, img_size)
    sandwich_x, sandwich_y = ImageTools.expandClass(sandwich_files, 1, classSize, img_size)

    # Arranging
    X = np.array(food_x, sandwich_x)
    y = np.array(food_y, sandwich_y)

    # Greyscaling and normalizing inputs to reduce features and improve comparability
    X = ImageTools.greyscaleImgs(x)
    X = ImageTools.normalizeImgs(x)

    # Train n' test:
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=np.random.randint(0, 100))
    model = network(img_size) # Calling of CNN
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    model.fit(X_train, y_train, nb_epoch=5, validation_split=0.1)


    # Saving model
    modelsave = open('model.pkl', 'wb')
    pickle.dump(model, modelsave)
    modelsave.close()
main()
