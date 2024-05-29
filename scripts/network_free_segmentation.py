import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys
from torchvision import utils
import torchvision.transforms as transforms
import faiss
from natsort import natsorted

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.inference_utils import run_on_batch
from options.test_options import TestOptions
from models.stylegan2.model import Generator
from models.psp import pSp


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def select_bg_saliency(n_clusters, clusters, H, W, threshold=9e-7):
    bg_mask = torch.zeros((H, W))
    sigma = H // 2
    h1, h2 = np.meshgrid(np.arange(-H/2, H/2), np.arange(-W/2, W/2))
    hg = np.exp(-(h1 ** 2 + h2 ** 2) / (2 * sigma ** 2))
    saliency_map = hg / hg.sum()
    mean_values = []
    for i in range(n_clusters):
        mean_values.append(saliency_map[clusters == i].mean())
    stats = np.array(mean_values).argsort()
    bg_mask[clusters == stats[1]] = 255
    bg_mask[clusters == stats[2]] = 255
    return bg_mask


def get_avg_image(net):
    avg_image = net(net.latent_avg.unsqueeze(0),
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0]
    avg_image = avg_image.to('cuda').float().detach()
    return avg_image


def major_voting(x):
    x = torch.stack(x)
    threshold = x.shape[0] // 2
    x = x.clip(0, 1)
    mask = x.sum(axis=0)
    mask[mask >= threshold] = 255
    mask[mask < threshold] = 0
    return mask


def run():
    test_opts = TestOptions().parse()
    
    os.makedirs(test_opts.exp_dir, exist_ok=True)

    ckpt = torch.load(test_opts.restyle_psp_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = test_opts.restyle_psp_path
    opts['batch_size'] = 1
    
    opts = Namespace(**opts)

    encoder = pSp(opts)
    opts.resize_outputs = False
    encoder.eval()
    encoder.cuda()
    
    device = 'cuda'
    latent = 512
    n_mlp = 8
    generator = Generator(test_opts.image_size, latent, n_mlp, channel_multiplier=2).to(device)
    generator.eval()

    g_ema = Generator(test_opts.image_size, latent, n_mlp, channel_multiplier=2).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    ckpt_ = torch.load(test_opts.generator_path, map_location=lambda storage, loc: storage)
    g_ema.load_state_dict(ckpt_['g_ema'])
    g_ema.eval()

    with torch.no_grad():
        mean_latent = g_ema.mean_latent(4096)

    img_transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
                ])

    images_count = 0
    for file_name in tqdm(natsorted(os.listdir(test_opts.data_path))):
        original_image = Image.open(test_opts.data_path + file_name).convert("RGB")
        transformed_image = img_transforms(original_image)
        
        with torch.no_grad():
            avg_image = get_avg_image(encoder)
            result_batch, result_latents = run_on_batch(transformed_image.unsqueeze(0).cuda(), encoder, opts, avg_image)
            latents1 = torch.from_numpy(result_latents[0][4]).reshape((1, 18, 512)).cuda()

        bg_masks = []
        for _ in range(test_opts.n_repeat_kmeans):
            styled_imgs = []
            for _ in range(test_opts.n_stylemixed):
                sample_z = torch.randn((1, 512)).cuda()
                with torch.no_grad():
                    new_mixing_c = np.random.randint(6, 10)
                    _, latents2 = g_ema([sample_z], truncation=0.7, truncation_latent=mean_latent, return_latents=True)
                    latent = torch.cat((latents1[:, :new_mixing_c], latents2[:, new_mixing_c:]), 1)
                    sample_mix, _ = g_ema([latent], inject_index=0, input_is_latent=True)
                    styled_imgs.append(sample_mix[0])

            mixed = torch.concatenate(styled_imgs, axis=0)
            mixed = torch.permute(mixed, (1, 2, 0))

            y = mixed.reshape((-1, mixed.shape[2]))
            y = np.ascontiguousarray(y.detach().cpu().numpy())

            H, W = mixed.shape[0], mixed.shape[1]

            kmeans = faiss.Kmeans(d=y.shape[1], k=test_opts.n_clusters, niter=200, gpu=True)
            kmeans.train(y)
            D, clusters = kmeans.index.search(y, 1)
            clusters = clusters.reshape((H, W))
            bg_mask = select_bg_saliency(test_opts.n_clusters, clusters, H, W)
            bg_masks.append(bg_mask)

        bg_mask = major_voting(bg_masks)
    
        utils.save_image(
                        bg_mask,
                        f"{test_opts.exp_dir}/{str(images_count).zfill(6)}_mask.png",
                        nrow=1,
                        normalize=True,
                        value_range=(0, 256),
                    )
        
        
        images_count += 1


if __name__ == '__main__':
    run()