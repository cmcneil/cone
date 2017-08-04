from abc import ABCMeta, abstractmethod
import numpy as np


class ConformalMapper:
    """
    A base class for disk->square/square->disk conformal mappings.
    See https://arxiv.org/abs/1509.06344 as a good resource.
    """
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self.res = kwargs['res']
        self.u_idx, self.v_idx = self.build_forward_map()
        self.x_idx, self.y_idx = self.build_backward_map()

    @abstractmethod
    def square_to_disk_fn(self, x, y):
        """
        Given a set of x, y (square) coordinates, returns the corresponding disk coordinates.
        This will be used to create the disk->square transform.

        Args:
            x, y: float. Square coordinates in [-1,  1]

        Returns:
            (float, float): in domain ret[0]**2 + ret[1]**2 < 1
        """
        pass

    @abstractmethod
    def disk_to_square_fn(self, u, v):
        """
        Given a set of u, v (disk) coordinates, returns the corresponding square coordinates.
        This will be used to create the square->disk transform.

        Args:
            u, v: float. Disk coordinates in [-1,  1]. For arguments outside u**2 + v**2 <= 1,
                return value doesn't matter.
        """
        pass

    def build_forward_map(self):
        """
        Returns:
            (np.array, np.array) These should be index arrays specifying the points in the
            DISK (in C order) that each point in the SQUARE maps to. Therefore, they should
            each be of length self.res**2, and they should be in the domain
            (u - self.res/2)**2 + (v - self.res/2)**2 < self.res / 2.
        """
        u_idx = []
        v_idx = []
        rad = self.res / 2.0
        for i in range(self.res):
            for j in range(self.res):
                x = float(i - rad) / rad
                y = float(j - rad) / rad
                u, v = self.square_to_disk_fn(x, y)
                u_idx.append(min(int((u + 1)*rad), self.res-1))
                v_idx.append(min(int((v + 1)*rad), self.res-1))
        u_idx = np.array(u_idx)
        v_idx = np.array(v_idx)
        return u_idx, v_idx

    def build_backward_map(self):
        """
        Returns:
            (np.array, np.array) These should be index arrays specifying the points in the
            SQUARE (in C order) that each point in the DISK maps to. They should
            each be of length self.res**2, and they should be in the domain
            [0, self.res]. Points in the reshaped disk will be set to 0 if outside the radius,
            so those values in the index arrays can be set to anything.
        """
        x_idx = []
        y_idx = []
        rad = self.res / 2.0
        for i in range(self.res):
            for j in range(self.res):
                u = float(i - rad) / rad
                v = float(j - rad) / rad
                x, y = self.disk_to_square_fn(u, v)
                x_idx.append(min(int((x + 1)*rad), self.res-1))
                y_idx.append(min(int((y + 1)*rad), self.res-1))
        x_idx = np.array(x_idx)
        y_idx = np.array(y_idx)
        return x_idx, y_idx

    def disk_to_square(self, image):
        """
        Args:
            image: np.array. Expected shape is (..., self.res, self.res)
        """
        assert np.shape(image)[-2:] == (self.res, self.res)
        return np.reshape(image[..., self.u_idx, self.v_idx], np.shape(image))

    def square_to_disk(self, image, outside_val=0):
        """
        Args:
            image: np.array. Expected shape is (..., self.res, self.res)
        """
        assert np.shape(image)[-2:] == (self.res, self.res)
        disk_im = np.reshape(image[..., self.x_idx, self.y_idx], np.shape(image))
        rad = self.res / 2.0
        print 'radius: ' + str(rad)
        for i in range(self.res):
            for j in range(self.res):
                if np.sqrt((i - rad)**2 + (j - rad)**2) > rad:
                    disk_im[i, j] = outside_val*np.ones(np.shape(image)[:-2])
        return disk_im


class FGSquircularMapper(ConformalMapper):
    """
    Implements the FGSquircular Conformal mapping. See See https://arxiv.org/abs/1509.06344
    """
    def __init__(self, **kwargs):
        super(FGSquircularMapper, self).__init__(**kwargs)

    def square_to_disk_fn(self, x, y):
        u = np.nan_to_num(x * np.sqrt(x**2 + y**2 - x**2 * y**2) / np.sqrt(x**2 + y**2))
        v = np.nan_to_num(y * np.sqrt(x**2 + y**2 - x**2 * y**2) / np.sqrt(x**2 + y**2))
        return (u, v)

    def disk_to_square_fn(self, u, v):
        x = np.nan_to_num(np.sign(u*v) / (v * np.sqrt(2)) * np.sqrt(
                    u**2 + v**2 - np.sqrt((u**2 + v**2) * (u**2 + v**2 - 4 * u**2 * v**2))))
        y = np.nan_to_num(np.sign(u*v) / (u * np.sqrt(2)) * np.sqrt(
                    u**2 + v**2 - np.sqrt((u**2 + v**2) * (u**2 + v**2 - 4 * u**2 * v**2))))
        return (x, y)
