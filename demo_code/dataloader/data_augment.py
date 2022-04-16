import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch
import torch.distributions as tdist
import numpy as np


class PairRandomCrop(transforms.RandomCrop):

    def __call__(self, image, label):

        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w)


class PairCompose(transforms.Compose):
    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(label)
        return img, label


class PairToTensor(transforms.ToTensor):
    def __call__(self, pic, label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic), F.to_tensor(label)
    
def random_noise_levels():
  """Generates random noise levels from a log-log linear distribution."""
  log_min_shot_noise = np.log(0.0001)
  log_max_shot_noise = np.log(0.012)
  log_shot_noise     = torch.FloatTensor(1).uniform_(log_min_shot_noise, log_max_shot_noise)
  shot_noise = torch.exp(log_shot_noise)

  line = lambda x: 2.18 * x + 1.20
  n    = tdist.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([0.26])) 
  log_read_noise = line(log_shot_noise) + n.sample()
  read_noise     = torch.exp(log_read_noise)
  return shot_noise, read_noise


def add_noise(image, shot_noise=0.01, read_noise=0.0005):
  """Adds random shot (proportional to image) and read (independent) noise."""

  #image    = image.permute(1, 2, 0) # Permute the image tensor to HxWxC format from CxHxW format
  variance = image * shot_noise + read_noise
  n        = tdist.Normal(loc=torch.zeros_like(variance), scale=torch.sqrt(variance)) 
  noise    = n.sample()
  out      = image + noise
  #out      = out.permute(2, 0, 1) # Re-Permute the tensor back to CxHxW format
  return out
