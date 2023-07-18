# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import tqdm
import argparse
import math
import random
import shutil
import sys
import os
import time
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import image_models

import yaml
import detectron2
import detectron2.data.transforms as T
import numpy as np
from detectron2.data import DatasetMapper
from detectron2.utils.events import EventStorage
from detectron2.data import detection_utils as utils
from detectron2.structures.masks import BitMasks
import copy
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.lmbda = lmbda
        self.zero = torch.zeros(1).to('cuda')
    
    def psnr(self, output, target):
        mse = torch.mean((output - target) ** 2)
        # if(mse == 0):
        #     return 100
        max_pixel = 1.
        psnr = 10 * torch.log10(max_pixel / mse)
        return torch.mean(psnr)

    def forward(self, output, target, mask=  None, lmbdamap=None):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if mask is not None:
            mse = self.mse(output['x_hat'], target)
            roi_mse = torch.mean(mse*mask.expand_as(target), [1,2,3])
            # roi_mse_de = torch.sum(mask.repeat(1,3,1,1),[1,2,3])
            out["mse_loss"] = roi_mse.mean()
            roi_mse = torch.mean(roi_mse*(lmbdamap.view(-1,)))
        else:
            out["mse_loss"] = self.mse(output["x_hat"], target)
        out["rdloss"] =  255**2 * roi_mse + out["bpp_loss"]
        
        out["psnr"] = self.psnr(torch.clamp(output["x_hat"],0,1), target)
        return out


class Metrics(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.zero = torch.zeros(1).to('cuda')
    
    def psnr(self, output, target):
        mse = torch.mean((output - target) ** 2, [1,2,3])
        # if(mse == 0):
        #     return 100
        max_pixel = 1.
        psnr = 10 * torch.log10(max_pixel / mse)
        return psnr

    def forward(self, output, target, mask=  None):
        N, _, H, W = target.size()
        out = {}
        num_pixels =  H * W

        bpp = torch.stack([(torch.log(likelihoods).reshape(N,-1).sum(-1) / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"].values()])
        bpp = bpp.sum(0)
        out["bpp_loss"] = bpp
        out["mse_loss"] = self.mse(output["x_hat"], target).mean()
        if mask is not None:
            out["roi_psnr"] = torch.mean(10 * torch.log10(1./(torch.sum(((torch.clamp(output["x_hat"],0,1)-target)*mask.repeat(1,3,1,1))**2)/torch.sum(mask.repeat(1,3,1,1)))))
            out['roi_mse'] = torch.mean(((torch.clamp(output["x_hat"],0,1)-target)*mask.repeat(1,3,1,1))**2)
            out["nroi_psnr"] = torch.mean(10 * torch.log10(1./(torch.sum(((torch.clamp(output["x_hat"],0,1)-target)*(1-mask).repeat(1,3,1,1))**2)/torch.sum((1-mask).repeat(1,3,1,1)))))
            out['nroi_mse'] = torch.mean(((torch.clamp(output["x_hat"],0,1)-target)*(1-mask).repeat(1,3,1,1))**2)
            out["roi_psnr_ind"] = 10 * torch.log10(1./(torch.sum(((torch.clamp(output["x_hat"],0,1)-target)*mask.repeat(1,3,1,1))**2, [1,2,3])/torch.sum(mask.repeat(1,3,1,1), [1,2,3])))
            out["nroi_psnr_ind"] = 10 * torch.log10(1./(torch.sum(((torch.clamp(output["x_hat"],0,1)-target)*(1-mask).repeat(1,3,1,1))**2, [1,2,3])/torch.sum((1-mask).repeat(1,3,1,1), [1,2,3])))
        out["psnr"] = self.psnr(torch.clamp(output["x_hat"],0,1), target)
        out["weighted_PSNR"] = 10 * torch.log10(1. / ((0.8*out['roi_mse']*num_pixels*3+0.2*out['nroi_mse']*num_pixels*3)/(0.8*(mask.sum())*3+ 0.2*((1-mask).sum())*3)))
        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def init(args):
    base_dir = f'{args.root}/{args.exp_name}/{args.quality_level}/'
    os.makedirs(base_dir, exist_ok=True)

    return base_dir


def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_dir)


def test_epoch(epoch, test_dataloader, model, criterion_rd, metrics, stage='test'):
    model.eval()
    device = next(model.parameters()).device
    lambda_list = [0.0018, 0.0051, 0.013, 0.03665, 0.0932]

    loss_am_mean = AverageMeter()
    with torch.no_grad():
        for n, lmbda in enumerate(lambda_list):
            loss_am = AverageMeter()
            bpp_loss = AverageMeter()
            mse_loss = AverageMeter()
            aux_loss = AverageMeter()
            psnr = AverageMeter()
            roipsnr = AverageMeter()
            nroipsnr = AverageMeter()
            roimse = AverageMeter()
            nroimse = AverageMeter()
            totalloss = AverageMeter()
            weighted_psnr = AverageMeter()
            for i, d in tqdm.tqdm(enumerate(test_dataloader),leave=False, total=len(test_dataloader.dataset)//test_dataloader.batch_size):
                codecinput = torch.stack([d[idx]['image'].float().div(255).to(device) for idx in range(len(d))])
                roimask = torch.zeros(codecinput.shape[0], codecinput.shape[2], codecinput.shape[3])
                for tt,dd in enumerate(d):
                    roimask[tt] = (torch.sum(BitMasks.from_polygon_masks(dd['instances'].get('gt_masks'), codecinput.shape[2],codecinput.shape[3]).tensor,0)>0).float()
                binary_roimask = roimask.unsqueeze(1).to(device)
                roimask = binary_roimask
                lmbda_norm = (np.log(lmbda)-min(np.log(lambda_list)))/(max(np.log(lambda_list))-min(np.log(lambda_list)))
                mask  = torch.zeros(codecinput.shape[0], 1, codecinput.shape[2], codecinput.shape[3], device=device).fill_(lmbda_norm)
                mask_decoder = mask[:,:,:codecinput.shape[2]//16,:codecinput.shape[3]//16]
                lmbda_mask  = torch.zeros(codecinput.shape[0], 1, codecinput.shape[2], codecinput.shape[3], device=device).fill_(lmbda)
                out_net = model(codecinput, mask, mask_decoder, roimask)
                out_criterion = criterion_rd(out_net, codecinput, roimask, torch.tensor([[lmbda]*codecinput.shape[0]]).to(device))
                total_loss = out_criterion['rdloss']

                out_metric = metrics(out_net, codecinput, binary_roimask)

                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                mse_loss.update(out_criterion["mse_loss"])
                psnr.update(out_criterion['psnr'])
                roipsnr.update(out_metric['roi_psnr'].mean())
                nroipsnr.update(out_metric['nroi_psnr'].mean())
                roimse.update(out_metric['roi_mse'].mean())
                nroimse.update(out_metric['nroi_mse'].mean())
                weighted_psnr.update(out_metric['weighted_PSNR'].mean())
                totalloss.update(total_loss)

            txt = f"{lmbda} || Bpp loss: {bpp_loss.avg:.4f} | weighted_PSNR: {weighted_psnr.avg:.3f}  | PSNR: {psnr.avg:.3f} | ROI PSNR: {roipsnr.avg:.4f} | NROI PSNR: {nroipsnr.avg:.4f} | ROI MSE: {roimse.avg:.4f} | NROI MSE: {nroimse.avg:.4f}"
            print(txt)
            loss_am_mean.update(loss_am.avg)
    model.train()
    return loss_am_mean.avg

def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
    torch.save(state, base_dir+filename)
    if is_best:
        shutil.copyfile(base_dir+filename, base_dir+"checkpoint_best_loss.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-c",
        "--config",
        default="config/vpt_default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        '--name', 
        default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), 
        type=str,
        help='Result dir name', 
    )
    given_configs, remaining = parser.parse_known_args(argv)
    with open(given_configs.config) as file:
        yaml_data= yaml.safe_load(file)
        parser.set_defaults(**yaml_data)
    args = parser.parse_args(remaining)
    return args


def main(argv):
    args = parse_args(argv)
    base_dir = init(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    setup_logger(base_dir + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')
    msg = f'======================= {args.name} ======================='
    logging.info(msg)
    for k in args.__dict__:
        logging.info(k + ':' + str(args.__dict__[k]))
    logging.info('=' * len(msg))

    detectron2.data.datasets.register_coco_instances('my_coco_2017_val',{},f'{args.dataset_path}/annotations/instances_val2017.json', f'{args.dataset_path}/val2017/')
    test_dataset = detectron2.data.get_detection_dataset_dicts('my_coco_2017_val')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    test_dataloader = detectron2.data.build_detection_test_loader(test_dataset, mapper=DatasetMapper(test_dataset, image_format='RGB', use_instance_mask=True, augmentations=[T.ResizeShortestEdge(512), T.CropTransform(0,0,512,512)]), num_workers=args.num_workers, batch_size=args.test_batch_size)

    net = image_models[args.model](quality=int(args.quality_level), prompt_config=args)
    net = net.to(device)

    rdcriterion = RateDistortionLoss()

    if args.checkpoint:  # load from previous checkpoint
        logging.info("Loading "+str(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if list(checkpoint["state_dict"].keys())[0][:7]=='module.':
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                new_state_dict[k[7:]] = v
        else:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                new_state_dict[k] = v
        net.load_state_dict(new_state_dict, strict=True)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
    
    loss = test_epoch(-1, test_dataloader, net, rdcriterion, Metrics(), 'test')

if __name__ == "__main__":
    main(sys.argv[1:])