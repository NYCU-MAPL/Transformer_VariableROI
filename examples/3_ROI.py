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
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
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


def filter_images_largerthan256(dataset_dicts):
    num_before = len(dataset_dicts)
    def valid(x):
        if x['height']>=256 and x['width']>=256:
                return True
        return False

    dataset_dicts = [x for x in dataset_dicts if valid(x)]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images smaller than 256x256. {} images left.".format(
            num_before - num_after, num_after
        )
    )
    return dataset_dicts


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.zero = torch.zeros(1).to('cuda')
    
    def psnr(self, output, target):
        mse = torch.mean((output - target) ** 2)
        if(mse == 0):
            return 100
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
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.lmbda = lmbda
        self.zero = torch.zeros(1).to('cuda')
    
    def psnr(self, output, target):
        mse = torch.mean((output - target) ** 2)
        if(mse == 0):
            return 100
        max_pixel = 1.
        psnr = 10 * torch.log10(max_pixel / mse)
        return torch.mean(psnr)
    
    def lpips(self, output, target):
        return torch.mean(self.lpips_vgg(output, target, normalize=True))

    def forward(self, output, target, mask=  None):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        bpp = torch.stack([(torch.log(likelihoods).reshape(N,-1).sum(-1) / (-math.log(2) * num_pixels)) for likelihoods in output["likelihoods"].values()])
        bpp = bpp.sum(0)
        out["bpp_loss"] = sum(bpp)
        out["mse_loss"] = self.mse(output["x_hat"], target).mean()
        if mask is not None:
            out["roi_psnr"] = torch.mean(10 * torch.log10(1./(torch.sum(((torch.clamp(output["x_hat"],0,1)-target)*mask.repeat(1,3,1,1))**2)/torch.sum(mask.repeat(1,3,1,1)))))
            out["nroi_psnr"] = torch.mean(10 * torch.log10(1./(torch.sum(((torch.clamp(output["x_hat"],0,1)-target)*(1-mask).repeat(1,3,1,1))**2)/torch.sum((1-mask).repeat(1,3,1,1)))))
        out["psnr"] = self.psnr(torch.clamp(output["x_hat"],0,1), target)
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


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )

    return optimizer, aux_optimizer


def _get_grid(size):
        x1 = torch.tensor(range(size[0]))
        x2 = torch.tensor(range(size[1]))
        grid_x1, grid_x2 = torch.meshgrid(x1, x2)

        grid1 = grid_x1.view(size[0], size[1], 1)
        grid2 = grid_x2.view(size[0], size[1], 1)
        grid = torch.cat([grid1, grid2], dim=-1)
        return grid

def train_one_epoch(
    model, criterion_rd, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, steps
):
    model.train()
    device = next(model.parameters()).device
    tqdm_emu = tqdm.tqdm(enumerate(train_dataloader), total=steps, leave=False)
    lambda_list = np.array([0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.0483, 0.0932])
    grid = _get_grid((256,256))
    for i, d in tqdm_emu:
        codecinput = torch.stack([d[idx]['image'].float().div(255).to(device) for idx in range(len(d))])
        roimask = torch.zeros(codecinput.shape[0], codecinput.shape[2], codecinput.shape[3])
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        lambda_masks=[]
        masks = []
        lmbdalist = []
        for idx,dd in enumerate(d):
            lmbda_norm = random.random()
            lmbda_norm = (idx%codecinput.shape[0])/codecinput.shape[0]+lmbda_norm*(1/codecinput.shape[0])
            lmbda = np.exp(lmbda_norm*(max(np.log(lambda_list))- min(np.log(lambda_list)))+min(np.log(lambda_list)))
            masks.append(torch.zeros(codecinput.shape[2], codecinput.shape[3]).fill_(lmbda_norm))
            lambda_masks.append(torch.zeros(codecinput.shape[2], codecinput.shape[3]).fill_(lmbda))
            lmbdalist.append(lmbda)
            
            # Random mask generation 
            # ref: Song et al., "Variable-rate deep image compression through spatially-adaptive feature transform," ICCV'21
            mask_prod = random.random()
            if mask_prod<=0.2:
                if random.random() < 0.01:
                    pass
                else:
                    roimask[idx][:] = (100 + 1) * random.random() / 100
            elif mask_prod<=0.5:
                bitmask = BitMasks.from_polygon_masks(dd['instances'].get('gt_masks'), codecinput.shape[2],codecinput.shape[3]).tensor
                if bitmask.shape[0]>0:
                    cat_rand = torch.rand((bitmask.shape[0],1,1))
                    roimask[idx] = torch.max(bitmask*cat_rand, dim=0).values
            elif mask_prod<=0.7:
                v1 = random.random() * 100
                v2 = random.random() * 100
                qmap = np.tile(np.linspace(v1, v2, codecinput.shape[2]), (codecinput.shape[2], 1)).astype(float)
                if random.random() < 0.5:
                    qmap = qmap.T
                roimask[idx] = torch.from_numpy(qmap)/100
            else:
                qmap = torch.zeros(codecinput.shape[2],codecinput.shape[3]).float()
                gaussian_num = int(1 + random.random() * 20)
                for _ in range(gaussian_num):
                    mu_x = codecinput.shape[2] * random.random()
                    mu_y = codecinput.shape[3] * random.random()
                    var_x = 2000 * random.random() + 1000
                    var_y = 2000 * random.random() + 1000

                    m = MultivariateNormal(torch.tensor([mu_x, mu_y]), torch.tensor([[var_x, 0], [0, var_y]]))
                    p = m.log_prob(grid)
                    kernel = torch.exp(p).numpy()
                    qmap += kernel
                qmap *= 100 / qmap.max() * (0.5 * random.random() + 0.5)
                roimask[idx] = qmap/100
        roimask = roimask.unsqueeze(1).to(device)
        lmbdalist = torch.tensor(lmbdalist).float().to(device)
        mask = torch.stack(masks).unsqueeze(1).to(device)
        mask_decoder = mask[:,:,:codecinput.shape[2]//16,:codecinput.shape[3]//16]
        lambda_masks = torch.stack(lambda_masks).unsqueeze(1).to(device)
        out_net = model(codecinput, mask, mask_decoder,roimask)
        out_criterion = criterion_rd(out_net, codecinput, roimask, lmbdalist)
        total_loss = out_criterion['rdloss']
        total_loss.backward()
        aux_loss = model.aux_loss()
        aux_loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        aux_optimizer.step()

        update_txt=f'[{i}/{steps}] | Loss: {total_loss.item():.3f} | MSE loss: {out_criterion["mse_loss"].item():.5f} | Bpp loss: {out_criterion["bpp_loss"].item():.4f}'
        tqdm_emu.set_postfix_str(update_txt, refresh=True)
        if i>=steps:
            break

def test_epoch(epoch, test_dataloader, model, criterion_rd, metrics, stage='test'):
    model.eval()
    device = next(model.parameters()).device
    lambda_list = [0.0018, 0.0051, 0.013, 0.03665, 0.0932]
    alpha_list  = [0]

    loss_am_mean = AverageMeter()
    for an , alpha in enumerate(alpha_list):
        with torch.no_grad():
            for n, lmbda in enumerate(lambda_list):
                loss_am = AverageMeter()
                bpp_loss = AverageMeter()
                mse_loss = AverageMeter()
                aux_loss = AverageMeter()
                psnr = AverageMeter()
                roipsnr = AverageMeter()
                nroipsnr = AverageMeter()
                totalloss = AverageMeter()
                for i, d in tqdm.tqdm(enumerate(test_dataloader),leave=False, total=len(test_dataloader.dataset)//test_dataloader.batch_size):
                    codecinput = torch.stack([d[idx]['image'].float().div(255).to(device) for idx in range(len(d))])
                    roimask = torch.zeros(codecinput.shape[0], codecinput.shape[2], codecinput.shape[3])
                    for tt,dd in enumerate(d):
                        roimask[tt] = (torch.sum(BitMasks.from_polygon_masks(dd['instances'].get('gt_masks'), codecinput.shape[2],codecinput.shape[3]).tensor,0)>0).float()
                    binary_roimask = roimask.unsqueeze(1).to(device)
                    roimask = (1-binary_roimask)*alpha+binary_roimask
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
                    roipsnr.update(out_metric['roi_psnr'])
                    nroipsnr.update(out_metric['nroi_psnr'])
                    totalloss.update(total_loss)

                txt = f"{alpha} | {n+1} || Bpp loss: {bpp_loss.avg:.4f} | PSNR: {psnr.avg:.3f} | ROI PSNR: {roipsnr.avg:.4f} | NROI PSNR: {nroipsnr.avg:.4f} "
                print(txt)
                loss_am_mean.update(loss_am.avg)
    model.train()
    return loss_am_mean.avg

def test_epoch_variable_kodak(epoch, test_dataloader, model, criterion_rd, metircs, stage='test'):
    model.eval()
    device = next(model.parameters()).device
    lambda_list = [0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.0483, 0.0932]
    alphas = [1]

    for an, alpha in enumerate(alphas):
        loss_am_mean = AverageMeter()
        with torch.no_grad():
            for n, lmbda in enumerate(lambda_list):
                loss_am = AverageMeter()
                bpp_loss = AverageMeter()
                psnr = AverageMeter()
                roipsnr = AverageMeter()
                nroipsnr = AverageMeter()
                totalloss = AverageMeter()
                for i, d in tqdm.tqdm(enumerate(test_dataloader),leave=False, total=len(test_dataloader.dataset)//test_dataloader.batch_size):
                    codecinput = d.to(device)
                    roimask = torch.ones(codecinput.shape[0], codecinput.shape[2], codecinput.shape[3])
                    roimask_binary = roimask.unsqueeze(1).to(device)
                    roimask = (1-roimask_binary)*alpha + roimask_binary
                    lmbda_norm = (np.log(lmbda)-min(np.log(lambda_list)))/(max(np.log(lambda_list))-min(np.log(lambda_list)))
                    mask  = torch.zeros(codecinput.shape[0], 1, codecinput.shape[2], codecinput.shape[3], device=device).fill_(lmbda_norm)
                    mask_decoder = mask[:,:,:codecinput.shape[2]//16,:codecinput.shape[3]//16] 
                    lmbda_mask  = torch.zeros(codecinput.shape[0], 1, codecinput.shape[2], codecinput.shape[3], device=device).fill_(lmbda)
                    out_net = model(codecinput, mask, mask_decoder, roimask)
                    out_rd = criterion_rd(out_net, codecinput, roimask_binary, torch.tensor([[lmbda]*codecinput.shape[0]]).to(device))
                    out_criterion = metircs(out_net, codecinput, roimask_binary)

                    bpp_loss.update(out_criterion["bpp_loss"])
                    psnr.update(out_criterion['psnr'])
                    roipsnr.update(out_criterion['roi_psnr'])
                    nroipsnr.update(out_criterion['nroi_psnr'])
                    totalloss.update(out_rd['rdloss'])
                    
                txt = f"{alpha} | {n+1} || Bpp loss: {bpp_loss.avg:.4f} | PSNR: {psnr.avg:.3f} | ROI PSNR: {roipsnr.avg:.5f} | NROI PSNR: {nroipsnr.avg:.5f}"
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
        "-T",
        "--TEST",
        action='store_true',
        help='Testing'
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

    if args.dataset=='coco':
        detectron2.data.datasets.register_coco_instances('my_coco_2017_train',{},f'{args.dataset_path}/annotations/instances_train2017.json', f'{args.dataset_path}/train2017/')
        detectron2.data.datasets.register_coco_instances('my_coco_2017_val',{},f'{args.dataset_path}/annotations/instances_val2017.json', f'{args.dataset_path}/val2017/')
        train_dataset = detectron2.data.get_detection_dataset_dicts('my_coco_2017_train')
        train_dataset = filter_images_largerthan256(train_dataset)
        val_dataset = detectron2.data.get_detection_dataset_dicts('my_coco_2017_val')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = detectron2.data.build_detection_train_loader(train_dataset, aspect_ratio_grouping=False, mapper=DatasetMapper(train_dataset, image_format='RGB', use_instance_mask=True, augmentations=[T.RandomCrop('absolute',(256,256),)]), num_workers=args.num_workers,total_batch_size=args.batch_size)
    val_dataloader = detectron2.data.build_detection_test_loader(val_dataset, mapper=DatasetMapper(val_dataset, image_format='RGB', use_instance_mask=True, augmentations=[T.ResizeShortestEdge(512), T.CropTransform(0,0,512,512)]), num_workers=args.num_workers, batch_size=args.test_batch_size)

    kodak_dataset = ImageFolder(args.kodak_path, split='', transform=transforms.ToTensor())
    kodak_dataloader = DataLoader(kodak_dataset,batch_size=1,num_workers=args.num_workers,shuffle=False,pin_memory=(device == "cuda"),)
    
    net = image_models[args.model](quality=int(args.quality_level), prompt_config=args)
    net = net.to(device)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35,70], gamma=0.5)
    rdcriterion = RateDistortionLoss()

    last_epoch = 0
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
        net.load_state_dict(new_state_dict, strict=False if not args.TEST else True)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
        
    best_loss = float("inf")
    tqrange = tqdm.trange(last_epoch, args.epochs)
    for epoch in tqrange:
        train_one_epoch(
            net,
            rdcriterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            args.steps,
        )
        loss = test_epoch(epoch, val_dataloader, net, rdcriterion, Metrics(),'val')
        _ = test_epoch_variable_kodak(epoch, kodak_dataloader, net, rdcriterion, Metrics(),'kodak')
        lr_scheduler.step()


        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            filename =  f'checkpoint.pth.tar'
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                base_dir,
                filename= filename 
            )
            if epoch%10==9:
                shutil.copyfile(base_dir+filename, base_dir+f"checkpoint_{epoch}.pth.tar")
    
    


if __name__ == "__main__":
    main(sys.argv[1:])

