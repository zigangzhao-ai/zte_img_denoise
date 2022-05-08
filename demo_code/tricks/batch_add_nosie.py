from dataloader.data_process import read_image, normalization, inv_normalization, write_image, write_back_dng
import numpy as np
import random
import os


class NoiseModelBase:  # base class
    def __call__(self, y, params=None):
        if params is None:
            K, g_scale, saturation_level, ratio = self._sample_params()
        else:
            K, g_scale, saturation_level, ratio = params

        y = y * saturation_level
        y = y / ratio

        if 'P' in self.model:
            z = np.random.poisson(y / K).astype(np.float32) * K
        elif 'p' in self.model:
            z = y + np.random.randn(*y.shape).astype(np.float32) * np.sqrt(np.maximum(K * y, 1e-10))
        else:
            z = y

        if 'g' in self.model:
            z = z + np.random.randn(*y.shape).astype(np.float32) * np.maximum(g_scale, 1e-10)  # Gaussian noise

        z = z * ratio
        z = z / saturation_level
        return z


# Only support baseline noise models: G / G+P / G+P*
class NoiseModel(NoiseModelBase):
    def __init__(self, model='g', cameras=None, include=None, exclude=None, cfa='bayer'):
        super().__init__()
        assert cfa in ['bayer', 'xtrans']
        assert include is None or exclude is None
        self.cameras = cameras or ['CanonEOS5D4', 'CanonEOS70D', 'CanonEOS700D', 'NikonD850', 'SonyA7S2']

        if include is not None:
            self.cameras = [self.cameras[include]]
        if exclude is not None:
            exclude_camera = set([self.cameras[exclude]])
            self.cameras = list(set(self.cameras) - exclude_camera)

        self.param_dir = os.path.join('camera_params', 'release')

        # print('[i] NoiseModel with {}'.format(self.param_dir))
        print('[i] cameras: {}'.format(self.cameras))
        # print('[i] using noise model {}'.format(model))

        self.camera_params = {}

        for camera in self.cameras:
            self.camera_params[camera] = np.load(os.path.join(self.param_dir, camera + '_params.npy'), allow_pickle=True).item()

        self.model = model
        # self.raw_packer = RawPacker(cfa)

    def _sample_params(self):
        camera = np.random.choice(self.cameras)
        # print(camera)

        saturation_level = 16383 - 800
        profiles = ['Profile-1']

        camera_params = self.camera_params[camera]
        Kmin = camera_params['Kmin']
        Kmax = camera_params['Kmax']
        profile = np.random.choice(profiles)
        camera_params = camera_params[profile]

        # log_K = np.random.uniform(low=np.log(Kmin), high=np.log(Kmax))
        log_K = np.random.uniform(low=np.log(1e-1), high=np.log(1))
        # log_K = np.random.uniform(low=np.log(1e-1), high=np.log(1))

        log_g_scale = np.random.standard_normal() * camera_params['g_scale']['sigma'] * 1 + camera_params['g_scale']['slope'] * log_K + camera_params['g_scale']['bias']

        K = np.exp(log_K)
        g_scale = np.exp(log_g_scale)

        ratio = np.random.uniform(low=100, high=150)
        # ratio = np.random.uniform(low=100, high=300)

        return (K, g_scale, saturation_level, ratio)



if __name__ == '__main__':

    black_level = 1024
    white_level = 16383
    image_dir = '../data/train'

    input_list = os.listdir(os.path.join(image_dir, 'noisy'))
    ground_list = os.listdir(os.path.join(image_dir, 'ground_truth'))
    shuffle_list = list(range(len(input_list)*5))
    random.shuffle(shuffle_list)
    for i, idx in enumerate(shuffle_list):
        print('i:',i)
        if i >=100:
            break
        NoiseMaker = NoiseModel(include=(idx+1) % 5)
        label, height, width = read_image(os.path.join(image_dir, 'ground_truth', ground_list[idx // 5]))#ground truth轮流加噪声
        label = normalization(label, black_level, white_level)
        label = np.transpose(label.reshape(-1, height // 2, width // 2, 4), (0, 3, 1, 2))
        image = NoiseMaker(label)
        image = np.maximum(np.minimum(image, 1.0), 0)
        image = np.ascontiguousarray(image)

        result_data = image.transpose(0, 2, 3, 1)
        result_data = result_data.reshape(-1, height // 2, width // 2, 4)
        result_data = inv_normalization(result_data, black_level, white_level)
        result_write_data = write_image(result_data, height, width)
        write_back_dng(image_dir+'/noisy/1_noise.dng', image_dir+'/noisy/'+str(idx).zfill(5)+'_noise'+'.dng', result_write_data)

        result_data = label.transpose(0, 2, 3, 1)
        result_data = result_data.reshape(-1, height // 2, width // 2, 4)
        result_data = inv_normalization(result_data, black_level, white_level)
        result_write_data = write_image(result_data, height, width)
        write_back_dng(image_dir + '/noisy/1_noise.dng', image_dir + '/ground_truth/'+str(idx).zfill(5)+'_gt'+'.dng', result_write_data)

