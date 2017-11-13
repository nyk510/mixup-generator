import numpy as np
from keras.preprocessing.image import ImageDataGenerator, Iterator, load_img, img_to_array


class MixupGenerator(Iterator):
    """
    Mixup を適用した画像を返すイテレータ
    """

    def __init__(self, image_paths, labels, batch_size=32, alpha=0.2, shuffle=True, datagen=None, seed=1):
        """
        :param list[str] image_names: 
        :param list labels: 
        :param int batch_size: 
        :param float alpha: 
        :param bool shuffle: 
        :param ImageDataGenerator datagen: 
        """
        self.image_paths = image_paths
        self.labels = labels
        self.datagen = datagen
        self.n_images = len(image_paths)
        super(MixupGenerator, self).__init__(self.n_images, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        for i, idx in enumerate(index_array):
            img = load_img(self.image_paths[idx])
            arr = img_to_array(img)

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.image_paths.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.image_paths[batch_ids[:self.batch_size]]
        X2 = self.image_paths[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.labels, list):
            y = []

            for y_train_ in self.labels:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.labels[batch_ids[:self.batch_size]]
            y2 = self.labels[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y
