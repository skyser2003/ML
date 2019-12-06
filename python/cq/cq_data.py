import glob, random, enum
from typing import List

import numpy as np
from imageio import imread

bg_r = 116
bg_g = 107
bg_b = 85


class CqDataType(enum.Enum):
    FACE = 1
    WHOLE = 2
    SD = 3


class CqDataMode(enum.IntEnum):
    NORMAL = 1
    BG_BLACK = 2
    GRAY_SCALE = 4


class CqData:
    def __init__(self, type: CqDataType, mode=CqDataMode.NORMAL, max_data=-1, scale_down=1, nchw=False):
        self.images = []
        self.flat_images = []
        self.width = -1
        self.height = -1
        self.num_channel = -1

        if type == CqDataType.FACE:
            dir = "cq_data/cq/face"
        elif type == CqDataType.WHOLE:
            dir = "cq_data/cq/whole"
        elif type == CqDataType.SD:
            dir = "cq_data/cq/sd"
        else:
            dir = "cq_data/cq/face"

        files = glob.glob(dir + "/*")
        if max_data != -1:
            random.shuffle(files)

        i = 0

        for f in files:
            if -1 < max_data <= i:
                break

            image = imread(f)

            new_image = self.remove_alpha(image)

            if mode & CqDataMode.NORMAL:
                pass
            else:
                if mode & CqDataMode.BG_BLACK:
                    new_image = self.set_bg_black(new_image, 3)

                if mode & CqDataMode.GRAY_SCALE:
                    new_image = self.set_gray_scale(new_image)

            if 1 < scale_down:
                new_image = self.scale_down(new_image, scale_down)

            self.width = new_image.shape[1]
            self.height = new_image.shape[0]
            self.num_channel = new_image.shape[2]

            new_image = np.divide(new_image, 255)

            if nchw is True:
                new_image = np.transpose(new_image, (2, 1, 0))

            self.images.append(new_image)
            self.flat_images.append(np.ndarray.flatten(new_image))

            i += 1

    @staticmethod
    def remove_alpha(image):
        new_image = np.ndarray((image.shape[0], image.shape[1], 3))

        for i in range(image.shape[0]):
            row = image[i]

            for j in range(row.shape[0]):
                pixel = row[j]

                if 4 <= len(pixel) and pixel[3] == 0:
                    new_image[i][j] = (0, 0, 0)
                else:
                    new_image[i][j] = (pixel[0], pixel[1], pixel[2])

        return new_image

    @staticmethod
    def set_bg_black(image, num_channel):
        new_image = np.ndarray((image.shape[0], image.shape[1], num_channel))

        for i in range(image.shape[0]):
            row = image[i]

            for j in range(row.shape[0]):
                pixel = row[j]

                if pixel[0] == bg_r and pixel[1] == bg_g and pixel[2] == bg_b:
                    new_image[i][j] = (0, 0, 0)
                else:
                    new_image[i][j] = (pixel[0], pixel[1], pixel[2])

        return new_image

    @staticmethod
    def set_gray_scale(image):
        new_image = np.ndarray((image.shape[0], image.shape[1], 1))

        for i in range(image.shape[0]):
            row = image[i]

            for j in range(row.shape[0]):
                pixel = row[j]

                new_image[i][j] = (int(pixel[0]) + int(pixel[1]) + int(pixel[2])) / 3

        return new_image

    @staticmethod
    def scale_down(image, scale):
        scale = int(scale)

        width = (image.shape[1] // scale) * scale
        height = (image.shape[0] // scale) * scale

        new_image = np.ndarray((height // scale, width // scale, image.shape[2]))

        for i in range(0, height, scale):
            row = image[i]

            for j in range(0, width, scale):
                pixel = row[j]

                new_image[i // scale][j // scale] = pixel

        return new_image

    def get_count(self):
        return len(self.images)

    def get_random_image(self, flatten: bool):
        images = self.__get_images_pool(flatten)

        index = random.randint(0, len(images) - 1)
        return images[index]

    def get_random_image_with_index(self):
        index = random.randint(0, len(self.flat_images) - 1)
        return index, self.flat_images[index]

    def get_images(self, indices: List[int], flatten: bool):
        images = self.__get_images_pool(flatten)

        ret = []

        for i in indices:
            image = images[i]
            ret.append(image)

        return np.array(ret)

    def get_ordered_batch(self, count: int, flatten: bool):
        ret_images = []

        images = self.__get_images_pool(flatten)

        num_images = len(images)

        end_index = count

        while True:
            ret_images.extend(images[0:end_index])

            if num_images < end_index:
                end_index -= num_images
            else:
                break

        return np.array(ret_images)

    def get_batch(self, count: int, flatten: bool):
        _, images = self.get_batch_with_index(count, flatten)
        return images

    def get_batch_with_index(self, count: int, flatten: bool):
        indices = self.get_random_indices(count)
        ret = []

        images = self.__get_images_pool(flatten)

        for i in indices:
            image = images[i]
            ret.append(image)

        return np.array(indices), np.array(ret)

    def get_random_indices(self, count):
        image_count = self.get_count()
        loop_count = count // image_count

        if count % image_count != 0:
            loop_count += 1

        indices = []

        for i in range(loop_count):
            for j in range(image_count):
                indices.append(j)

        random.shuffle(indices)

        return indices[:count]

    def get_image_width(self) -> int:
        return self.width

    def get_image_height(self) -> int:
        return self.height

    def get_channel_count(self) -> int:
        return self.num_channel

    def __get_images_pool(self, flatten: bool):
        images = self.flat_images if flatten else self.images
        return images


if __name__ == "__main__":
    data = CqData(CqDataType.WHOLE)
