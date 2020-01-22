import numpy as np
import matplotlib.pyplot as plt
from imageio import imsave
from skimage.transform import resize


class Plot_Reproduce_Performance():
    def __init__(self, DIR: str, n_img_x=8, n_img_y=8, img_w=28, img_h=28, resize_factor=1.0, nchw=False):
        self.DIR = DIR
        self.nchw = nchw

        assert n_img_x > 0 and n_img_y > 0

        self.n_img_x = n_img_x
        self.n_img_y = n_img_y
        self.n_tot_imgs = n_img_x * n_img_y

        assert img_w > 0 and img_h > 0

        self.img_w = img_w
        self.img_h = img_h

        assert resize_factor > 0

        self.resize_factor = resize_factor

    def save_pngs(self, images, num_channel, name='result.png'):
        num_sample = self.n_img_x * self.n_img_y

        if self.nchw:
            images = images.reshape(num_sample, num_channel, self.img_h, self.img_w)
            images = np.transpose(images, (0, 3, 2, 1))
        else:
            images = images.reshape(num_sample, self.img_h, self.img_w, num_channel)

        outputs = self._merge_pngs(images, [self.n_img_x, self.n_img_y])

        imsave(self.DIR + "/" + name, outputs)

    def save_single_image(self, image, num_channel, name="result.png"):
        if self.nchw:
            image = image.reshape(num_channel, self.img_h, self.img_w)
            image = np.transpose(image, (2, 1, 0))
        else:
            image = image.reshape(self.img_h, self.img_w, num_channel)

        imsave(self.DIR + "/" + name, image)

    def _merge_pngs(self, images, size):
        w, h, channel = images.shape[1], images.shape[2], images.shape[3]
        num_x, num_y = size

        w_ = int(w * self.resize_factor)
        h_ = int(h * self.resize_factor)

        img = np.zeros((num_x * w_, num_y * h_, channel))

        for idx, image in enumerate(images):
            i = int(idx % size[1])
            j = int(idx / size[1])

            image_ = resize(image, (w_, h_, channel), order=0)

            begin_x = i * w_
            end_x = (i + 1) * w_

            begin_y = j * h_
            end_y = (j + 1) * h_

            img[begin_x:end_x, begin_y:end_y] = image_

        img *= 255
        img = img.astype("uint8")

        return img

    def _merge(self, images, size):
        h, w = images.shape[1], images.shape[2]

        h_ = int(h * self.resize_factor)
        w_ = int(w * self.resize_factor)

        img = np.zeros((h_ * size[0], w_ * size[1]))

        for idx, image in enumerate(images):
            i = int(idx % size[1])
            j = int(idx / size[1])

            image_ = resize(image, (w_, h_))

            img[j * h_:j * h_ + h_, i * w_:i * w_ + w_] = image_

        return img


class Plot_Manifold_Learning_Result():
    def __init__(self, DIR, n_img_x=20, n_img_y=20, img_w=28, img_h=28, resize_factor=1.0, z_range=4):
        self.DIR = DIR

        assert n_img_x > 0 and n_img_y > 0

        self.n_img_x = n_img_x
        self.n_img_y = n_img_y
        self.n_tot_imgs = n_img_x * n_img_y

        assert img_w > 0 and img_h > 0

        self.img_w = img_w
        self.img_h = img_h

        assert resize_factor > 0

        self.resize_factor = resize_factor

        assert z_range > 0
        self.z_range = z_range

        self._set_latent_vectors()

    def _set_latent_vectors(self):
        # z1 = np.linspace(-self.z_range, self.z_range, self.n_img_y)
        # z2 = np.linspace(-self.z_range, self.z_range, self.n_img_x)
        #
        # z = np.array(np.meshgrid(z1, z2))
        # z = z.reshape([-1, 2])

        # borrowed from https://github.com/fastforwardlabs/vae-tf/blob/master/plot.py
        z = np.rollaxis(
            np.mgrid[self.z_range:-self.z_range:self.n_img_y * 1j, self.z_range:-self.z_range:self.n_img_x * 1j], 0, 3)
        # z1 = np.rollaxis(np.mgrid[1:-1:self.n_img_y * 1j, 1:-1:self.n_img_x * 1j], 0, 3)
        # z = z1**2
        # z[z1<0] *= -1
        #
        # z = z*self.z_range

        self.z = z.reshape([-1, 2])

    def save_images(self, images, name='result.jpg'):
        images = images.reshape(self.n_img_x * self.n_img_y, self.img_h, self.img_w)
        imsave(self.DIR + "/" + name, self._merge(images, [self.n_img_y, self.n_img_x]))

    def _merge(self, images, size):
        h, w = images.shape[1], images.shape[2]

        h_ = int(h * self.resize_factor)
        w_ = int(w * self.resize_factor)

        img = np.zeros((h_ * size[0], w_ * size[1]))

        for idx, image in enumerate(images):
            i = int(idx % size[1])
            j = int(idx / size[1])

            image_ = resize(image, (w_, h_))

            img[j * h_:j * h_ + h_, i * w_:i * w_ + w_] = image_

        return img

    # borrowed from https://github.com/ykwon0407/variational_autoencoder/blob/master/variational_bayes.ipynb
    def save_scattered_image(self, z, id, name='scattered_image.jpg'):
        N = 10
        plt.figure(figsize=(8, 6))
        plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
        plt.colorbar(ticks=range(N))
        axes = plt.gca()
        axes.set_xlim([-self.z_range - 2, self.z_range + 2])
        axes.set_ylim([-self.z_range - 2, self.z_range + 2])
        plt.grid(True)
        plt.savefig(self.DIR + "/" + name)


# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)
