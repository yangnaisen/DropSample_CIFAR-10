import numpy as np
from keras.utils import Sequence


class CIFAR10DataLoader(Sequence):
    def __init__(
            self,
            x_set,
            y_set,
            batch_size=512,
            crop_size=32,
            cutout_size=8,
            is_train=False,
    ):
        self.x, self.y = x_set, y_set
        self.crop_size = crop_size
        self.cutout_size = cutout_size
        self.batch_size = batch_size
        self.is_train = is_train

    def __len__(self):

        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def crop(self, x,center_crop = False):
        x = x.copy()
        if center_crop:
            x = x[4:4 + self.crop_size, 4:4 +
              self.crop_size]
        else:
            start_h = np.random.randint(0, x.shape[0] - self.crop_size)
            start_w = np.random.randint(0, x.shape[1] - self.crop_size)
            x = x[start_h:start_h + self.crop_size, start_w:start_w +
                self.crop_size]
        return x

    def flip_lr(self, x):
        x = x.copy()
        
        x = x[:, ::-1]
        return x

    def cutout(self, x):
        x = x.copy()
        
        start_h = np.random.randint(0, x.shape[0] - self.cutout_size)
        start_w = np.random.randint(0, x.shape[1] - self.cutout_size)
        x[start_h:start_h + self.cutout_size, start_w:start_w +
                self.cutout_size] = 0.0
        return x

    def preprocessing(self, x):
        x = self.crop(x)
        
            
        if np.random.random() > 0.5:
                x = self.cutout(x)
                
        if np.random.random() > 0.5:
                x = self.flip_lr(x)
   
        

        return x

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.is_train:
            batch_x = [self.preprocessing(item) for item in list(batch_x.copy())]

        return np.array(batch_x), np.array(batch_y)


def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)], mode='reflect')


cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    #x = x.astype('float16')
    #x -= 128
    #x /= 128
    return x
