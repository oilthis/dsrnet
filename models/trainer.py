import os
import time
import logging
import itertools
import math
import numpy as np
import random
from PIL import Image
import importlib
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import importlib
import sys
sys.path.append("..")
from dataloader.dataset import TrainSet, GuanDao, b100 #, TestSet , TestSet_multi, Urban100, Sun80
from utils import util, calculate_PSNR_SSIM
from models.modules import define_G
from models.losses import PerceptualLoss, AdversarialLoss
from dataloader import DistIterSampler, create_dataloader


class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.augmentation = args.data_augmentation
        self.device = torch.device('cuda' if len(args.gpu_ids) != 0 else 'cpu')
        args.device = self.device


        ## init dataloader
        if args.phase == 'train':
            self.train_dataset = GuanDao(self.args)
            if args.dist:
                dataset_ratio = 1
                train_sampler = DistIterSampler(self.train_dataset, args.world_size, args.rank, dataset_ratio)
                self.train_dataloader = create_dataloader(self.train_dataset, args, train_sampler)
            else:
                self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

        testset_ = getattr(importlib.import_module('dataloader.dataset'), args.testset, None)
        self.test_dataset = testset_(self.args)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

        ## init network
        self.net = define_G(args)
        if args.resume:
            self.load_networks('net', self.args.resume)
        if self.args.unet:
            from models.archs.UNet_arch import UNet
            self.unet = UNet(3, 1)
            self.unet.to(self.device)
        if self.args.distill:
            from models.archs.MASA_arch import MASA
            from models.AFD import AFD
            self.t = MASA(self.args, teacher='True')
            self.t.to(self.device)
            self.load_networks('t', self.args.t_resume)

            self.net.eval()
            self.t.eval() 
                
            self.args.qk_dim = 64
            h, w = 80, 112
            lr = torch.FloatTensor(1, 3, h, w).cuda()
            ref = torch.FloatTensor(1, 3, h*4, w*4).cuda()
            ref_down = torch.FloatTensor(1, 3, h, w).cuda()
            dep_lr = torch.FloatTensor(1, 1, h, w).cuda()
            dep_ref_down = torch.FloatTensor(1, 1, h, w).cuda()
            with torch.no_grad():
                fea_lr_lt, fea_dep_lr_lt, _ = self.t(lr, ref, ref_down, dep_lr, dep_ref_down)
                fea_lr_ls, fea_dep_lr_ls, _ = self.net(lr, ref, ref_down, dep_lr, dep_ref_down)
            self.args.s_shapes = [ft.size() for ft in fea_lr_lt]
            self.args.t_shapes = [fs.size() for fs in fea_lr_ls]
            self.args.n_t, self.args.unique_t_shapes = self.unique_shape(self.args.t_shapes)
            self.criterion_ad1 = AFD(self.args).to(self.device)
            self.args.s_shapes = [ft.size() for ft in fea_dep_lr_lt]
            self.args.t_shapes = [fs.size() for fs in fea_dep_lr_ls]
            self.args.n_t, self.args.unique_t_shapes = self.unique_shape(self.args.t_shapes)
            self.criterion_ad2 = AFD(self.args).to(self.device)
            # print(self.args)
            self.lambda_ad = args.lambda_ad
            self.lambda_kd = args.lambda_kd
            if args.rank <= 0:
                    logging.info('  using kd&ad loss...')

        if args.rank <= 0:
            logging.info('----- generator parameters: %f -----' % (sum(param.numel() for param in self.net.parameters()) / (10**6)))

        ## init loss and optimizer
        if args.phase == 'train':
            if args.rank <= 0:
                logging.info('init criterion and optimizer...')
            g_params = [self.net.parameters()]
            if args.unet:
                g_params.append(self.unet.parameters())
            if args.distill:
                g_params.append(self.criterion_ad1.parameters())
                g_params.append(self.criterion_ad2.parameters())

            self.criterion_mse = nn.MSELoss().to(self.device)
            if args.loss_mse:
                self.criterion_mse = nn.MSELoss().to(self.device)
                self.lambda_mse = args.lambda_mse
                if args.rank <= 0:
                    logging.info('  using mse loss...')

            if args.loss_l1:
                self.criterion_l1 = nn.L1Loss().to(self.device)
                self.lambda_l1 = args.lambda_l1
                if args.rank <= 0:
                    logging.info('  using l1 loss...')

            if args.unet:
                self.lambda_dep = args.lambda_dep
                if args.rank <= 0:
                    logging.info('  using dep loss...')

            if args.loss_adv:
                self.criterion_adv = AdversarialLoss(gpu_ids=args.gpu_ids, dist=args.dist, gan_type=args.gan_type,
                                                             gan_k=1, lr_dis=args.lr_D, train_crop_size=40)
                self.lambda_adv = args.lambda_adv
                if args.rank <= 0:
                    logging.info('  using adv loss...')

            if args.loss_perceptual:
                self.criterion_perceptual = PerceptualLoss(layer_weights={'conv5_4': 1.}).to(self.device)
                self.lambda_perceptual = args.lambda_perceptual
                if args.rank <= 0:
                    logging.info('  using perceptual loss...')

            self.optimizer_G = torch.optim.Adam(itertools.chain.from_iterable(g_params), lr=args.lr, weight_decay=args.weight_decay)
            self.scheduler = CosineAnnealingLR(self.optimizer_G, T_max=500)  # T_max=args.max_iter

            if args.resume_optim:
                self.load_networks('optimizer_G', self.args.resume_optim)
            if args.resume_scheduler:
                self.load_networks('scheduler', self.args.resume_scheduler)
            if args.resume_unet:
                self.load_networks('unet', self.args.resume_unet)


    def set_learning_rate(self, optimizer, epoch):
        current_lr = self.args.lr * 0.3**(epoch//550)
        optimizer.param_groups[0]['lr'] = current_lr
        if self.args.rank <= 0:
            logging.info('current_lr: %f' % (current_lr))

    def vis_results(self, epoch, i, images):
        for j in range(min(images[0].size(0), 5)):
            save_name = os.path.join(self.args.vis_save_dir, 'vis_%d_%s_%d.jpg' % (epoch, i, j))
            # temps = []
            # for imgs in images:
            #     temps.append(imgs[j])
            # temps = torch.stack(temps)
            # B = temps[:, 0:1, :, :]
            # G = temps[:, 1:2, :, :]
            # R = temps[:, 2:3, :, :]
            # temps = torch.cat([R, G, B], dim=1)
            torchvision.utils.save_image(images[0], save_name)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def prepare(self, batch_samples):
        for key in batch_samples.keys():
            if 'name' not in key and 'pad_nums' not in key:
                batch_samples[key] = Variable(batch_samples[key].to(self.device), requires_grad=False)

        if self.args.phase == 'train':
            if self.augmentation:
                # flip
                if np.random.randint(0, 2) == 1:
                    batch_samples['HR'] = torch.flip(batch_samples['HR'], dims=[2])
                    batch_samples['LR'] = torch.flip(batch_samples['LR'], dims=[2])
                    if self.args.depth:
                        batch_samples['LR_dep'] = torch.flip(batch_samples['LR_dep'], dims=[2])
                # if np.random.randint(0, 2) == 1:
                    batch_samples['Ref'] = torch.flip(batch_samples['Ref'], dims=[2])
                    batch_samples['Ref_down'] = torch.flip(batch_samples['Ref_down'], dims=[2])
                    if self.args.depth:
                        batch_samples['Ref_dep_down'] = torch.flip(batch_samples['Ref_dep_down'], dims=[2])
                if np.random.randint(0, 2) == 1:
                    batch_samples['HR'] = torch.flip(batch_samples['HR'], dims=[3])
                    batch_samples['LR'] = torch.flip(batch_samples['LR'], dims=[3])
                    if self.args.depth:
                        batch_samples['LR_dep'] = torch.flip(batch_samples['LR_dep'], dims=[3])
                # if np.random.randint(0, 2) == 1:
                    batch_samples['Ref'] = torch.flip(batch_samples['Ref'], dims=[3])
                    batch_samples['Ref_down'] = torch.flip(batch_samples['Ref_down'], dims=[3])
                    if self.args.depth:
                        batch_samples['Ref_dep_down'] = torch.flip(batch_samples['Ref_dep_down'], dims=[3])
                # rotate
                if np.random.randint(0, 2) == 1:
                    k = np.random.randint(0, 4)
                    batch_samples['HR'] = torch.rot90(batch_samples['HR'], k, dims=[2, 3])
                    batch_samples['LR'] = torch.rot90(batch_samples['LR'], k, dims=[2, 3])
                    if self.args.depth:
                        batch_samples['LR_dep'] = torch.rot90(batch_samples['LR_dep'], k, dims=[2, 3])
                # if np.random.randint(0, 2) == 1:
                #     k = np.random.randint(0, 4)
                    batch_samples['Ref'] = torch.rot90(batch_samples['Ref'], k, dims=[2, 3])
                    batch_samples['Ref_down'] = torch.rot90(batch_samples['Ref_down'], k, dims=[2, 3])
                    if self.args.depth:
                        batch_samples['Ref_dep_down'] = torch.rot90(batch_samples['Ref_dep_down'], k, dims=[2, 3])

        return batch_samples

    def unique_shape(self, s_shapes):
        n_s = []
        unique_shapes = []
        n = -1
        for s_shape in s_shapes:
            if s_shape not in unique_shapes:
                unique_shapes.append(s_shape)
                n += 1
            n_s.append(n)
        return n_s, unique_shapes

    def train(self):
        if self.args.rank <= 0:
            logging.info('training on  ...' + self.args.dataset)
            logging.info('%d training samples' % (self.train_dataset.__len__()))
            logging.info('the init lr: %f'%(self.args.lr))
        steps = 0
        self.net.train()

        if self.args.use_tb_logger:
            if self.args.rank <= 0:
                tb_logger = SummaryWriter(log_dir='tb_logger/' + self.args.name)

        self.best_psnr = 0
        self.augmentation = False  # disenable data augmentation to warm up the encoder

        for i in range(self.args.start_iter, self.args.max_iter):
            self.scheduler.step()
            logging.info('current_lr: %f' % (self.optimizer_G.param_groups[0]['lr']))
            t0 = time.time()
            for j, batch_samples in enumerate(self.train_dataloader):
                log_info = 'epoch:%03d step:%04d  ' % (i, j)

                ## prepare data
                batch_samples = self.prepare(batch_samples)
                LR = batch_samples['LR']
                Ref = batch_samples['Ref']
                Ref_down = batch_samples['Ref_down']

                ## forward
                if not self.args.depth:
                    output = self.net(LR, Ref, Ref_down)
                else:
                    LR_dep = batch_samples['LR_dep']
                    Ref_dep_down = batch_samples['Ref_dep_down']
                    if self.args.unet:
                        from models.archs.UNet_arch import UNet
                        self.unet = UNet(3, 1)
                        self.unet.to(self.device)
                        lr_dep = self.unet(LR)
                        ref_dep_down = self.unet(Ref_down)
                        output = self.net(LR, Ref, Ref_down, lr_dep, ref_dep_down)
                    elif self.args.distill:
                        fea_lr_lt, fea_dep_lr_lt, out_t = self.t(LR, Ref, Ref_down, LR_dep, Ref_dep_down)
                        fea_lr_ls, fea_dep_lr_ls, output = self.net(LR, Ref, Ref_down, LR_dep, Ref_dep_down)
                    else:
                        output = self.net(LR, Ref, Ref_down, LR_dep, Ref_dep_down)
                ## optimization
                loss = 0
                self.optimizer_G.zero_grad()
                
                if self.args.loss_kd:
                    kd_loss = self.criterion_l1(out_t, output)
                    kd_loss = kd_loss * self.lambda_kd
                    loss += kd_loss
                    log_info += 'kd_loss:%.06f ' % (kd_loss.item())

                if self.args.loss_ad:
                    ad_loss1 = self.criterion_ad1(fea_lr_ls, fea_lr_lt)
                    ad_loss2 = self.criterion_ad2(fea_dep_lr_lt, fea_dep_lr_ls)
                    ad_loss = (ad_loss1 + ad_loss2) * self.lambda_ad * 0.5
                    loss += ad_loss
                    log_info += 'ad_loss:%.06f ' % (ad_loss.item())

                if self.args.unet:
                    dep_loss = self.criterion_ad1(lr_dep, LR_dep) + self.criterion_ad2(ref_dep_down, Ref_dep_down) 
                    dep_loss = dep_loss * self.lambda_dep
                    loss += dep_loss
                    log_info += 'dep_loss:%.06f ' % (dep_loss.item())
                
                if self.args.loss_mse:
                    mse_loss = self.criterion_mse(output, batch_samples['HR'])
                    mse_loss = mse_loss * self.lambda_mse
                    loss += mse_loss
                    log_info += 'mse_loss:%.06f ' % (mse_loss.item())

                if self.args.loss_l1:
                    l1_loss = self.criterion_l1(output, batch_samples['HR'])
                    l1_loss = l1_loss * self.lambda_l1
                    loss += l1_loss
                    log_info += 'l1_loss:%.06f ' % (l1_loss.item())

                if self.args.loss_perceptual:
                    perceptual_loss, _ = self.criterion_perceptual(output, batch_samples['HR'])
                    perceptual_loss = perceptual_loss * self.lambda_perceptual
                    loss += perceptual_loss
                    log_info += 'perceptual_loss:%.06f ' % (perceptual_loss.item())

                if self.args.loss_adv:
                    adv_loss, d_loss = self.criterion_adv(output, batch_samples['HR'])
                    adv_loss = adv_loss * self.lambda_adv
                    loss += adv_loss
                    log_info += 'adv_loss:%.06f ' % (adv_loss.item())
                    log_info += 'd_loss:%.06f ' % (d_loss.item())

                log_info += 'loss_sum:%f ' % (loss.item())
                loss.backward()
                self.optimizer_G.step()

                ## print information
                if j % self.args.log_freq == 0:
                    t1 = time.time()
                    log_info += 'aug:%s ' % str(self.augmentation)
                    log_info += '%4.6fs/batch' % ((t1-t0)/self.args.log_freq)
                    if self.args.rank <= 0:
                        logging.info(log_info)
                    t0 = time.time()

                ## visualization
                if j % self.args.vis_freq == 0:
                    LR_sr = F.interpolate(batch_samples['LR'], scale_factor=self.args.sr_scale, mode='bicubic')
                    # vis_temps = [LR_sr, batch_samples['HR'], batch_samples['Ref'], output]
                    # self.vis_results(i, j, vis_temps)
                    self.vis_results(i, str(j)+'lr', [LR_sr])
                    self.vis_results(i, str(j)+'hr', [batch_samples['HR']])
                    self.vis_results(i, str(j)+'ref', [batch_samples['Ref']])
                    self.vis_results(i, str(j)+'sr', output)

                ## write tb_logger
                if self.args.use_tb_logger:
                    if steps % self.args.vis_step_freq == 0:
                        if self.args.rank <= 0:
                            if self.args.loss_mse:
                                tb_logger.add_scalar('mse_loss', mse_loss.item(), steps)
                            if self.args.loss_l1:
                                tb_logger.add_scalar('l1_loss', l1_loss.item(), steps)
                            if self.args.loss_perceptual:
                                if i > 5:
                                    tb_logger.add_scalar('perceptual_loss', perceptual_loss.item(), steps)
                            if self.args.loss_adv:
                                if i > 5:
                                    tb_logger.add_scalar('adv_loss', adv_loss.item(), steps)
                                    tb_logger.add_scalar('d_loss', d_loss.item(), steps)

                steps += 1

            ## save networks
            if i % self.args.save_epoch_freq == 0:
                if self.args.rank <= 0:
                    logging.info('Saving state, epoch: %d iter:%d' % (i, 0))
                    self.save_networks('net', i)
                    self.save_networks('optimizer_G', i)
                    self.save_networks('scheduler', i)
                    if self.args.unet:
                        self.save_networks('unet', i)

            # if not self.args.loss_adv:
            if i >= 0:
                self.args.phase = 'eval'
                psnr, ssim = self.evaluate()
                logging.info('psnr:%.06f   ssim:%.06f ' % (psnr, ssim))
                if psnr > self.best_psnr:
                    self.best_psnr = psnr
                    if self.args.rank <= 0:
                        logging.info('best_psnr:%.06f ' % (self.best_psnr))
                        logging.info('Saving state, epoch: %d iter:%d' % (i, 0))
                        self.save_networks('net', 'best')
                        self.save_networks('optimizer_G', 'best')
                        self.save_networks('scheduler', 'best')
                        if self.args.unet:
                            self.save_networks('unet', 'best')
                    ## start data augmentation
                    if i > 30:
                        self.augmentation = self.args.data_augmentation
                self.args.phase = 'train'

        ## end of training
        if self.args.rank <= 0:
            tb_logger.close()
            self.save_networks('net', 'final')
            logging.info('The training stage on %s is over!!!' % (self.args.dataset))


    def test(self):
        scale = self.args.sr_scale
        save_path = os.path.join(self.args.save_folder, 'output_imgs')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        self.net.eval()
        logging.info('start testing...')
        logging.info('%d testing samples' % (self.test_dataset.__len__()))
        num = 0

        total_time = 0
        PSNR = []
        SSIM = []
        with torch.no_grad():
            for batch, batch_samples in enumerate(self.test_dataloader):
                batch_samples = self.prepare(batch_samples)
                HR = batch_samples['HR']
                HR_pad = batch_samples['HR_pad']
                Ref = batch_samples['Ref']
                LR = batch_samples['LR']
                LR_pad = batch_samples['LR_pad']
                Ref_down = batch_samples['Ref_down']
                pad_nums = batch_samples['pad_nums']
                pad_t, pad_d, pad_l, pad_r = pad_nums

                if not self.args.depth:
                    output = self.net(LR_pad, Ref, Ref_down)
                else:
                    LR_dep = batch_samples['LR_dep']
                    Ref_dep_down = batch_samples['Ref_dep_down']
                    output = self.net(LR_pad, Ref, Ref_down, LR_dep, Ref_dep_down)
                if output.size() != HR.size():
                    _, _, h, w = HR.size()
                    output = output[:, :, pad_t:pad_t+h, pad_l:pad_l+w]

                to_y = True
                output_img = calculate_PSNR_SSIM.tensor2img(output)
                gt = calculate_PSNR_SSIM.tensor2img(HR)
                output_img = output_img[:, :, [2, 1, 0]]
                gt = gt[:, :, [2, 1, 0]]
                output_img = output_img.astype(np.float32) / 255.0
                gt = gt.astype(np.float32) / 255.0

                if to_y:
                    output_img = calculate_PSNR_SSIM.bgr2ycbcr(output_img, only_y=to_y)
                    gt = calculate_PSNR_SSIM.bgr2ycbcr(gt, only_y=to_y)

                psnr = calculate_PSNR_SSIM.calculate_psnr(output_img * 255, gt * 255)
                ssim = calculate_PSNR_SSIM.calculate_ssim(output_img * 255, gt * 255)
                PSNR.append(psnr)
                SSIM.append(ssim)
                logging.info('psnr: %.6f    ssim: %.6f' % (psnr, ssim))

                image_name = batch_samples['HR_name'][0]
                path = os.path.join(save_path, image_name)
                out_img = output[0].flip(dims=(0,)).clamp(0., 1.)
                gt_img = HR[0].flip(dims=(0,))

                # torchvision.utils.save_image(torch.stack([out_img, gt_img]), path)
                torchvision.utils.save_image(out_img, path)
                logging.info('saving %d_th image: %s' % (num, image_name))
                num += 1

        PSNR = np.mean(PSNR)
        SSIM = np.mean(SSIM)
        logging.info('--------- average PSNR: %.06f,  SSIM: %.06f' % (PSNR, SSIM))

    def evaluate(self):
        scale = self.args.sr_scale
        self.net.eval()
        logging.info('start testing...')
        logging.info('%d testing samples' % (self.test_dataset.__len__()))

        PSNR = []
        SSIM = []
        with torch.no_grad():
            for batch, batch_samples in enumerate(self.test_dataloader):

                batch_samples = self.prepare(batch_samples)
                HR = batch_samples['HR']
                HR_pad = batch_samples['HR_pad']
                Ref = batch_samples['Ref']
                LR = batch_samples['LR']
                LR_pad = batch_samples['LR_pad']
                Ref_down = batch_samples['Ref_down']
                pad_nums = batch_samples['pad_nums']
                pad_t, pad_d, pad_l, pad_r = pad_nums
                if self.args.distill:
                    if not self.args.depth:
                        _, _, output = self.net(LR_pad, Ref, Ref_down)
                    else:
                        LR_dep = batch_samples['LR_dep']
                        Ref_dep_down = batch_samples['Ref_dep_down']
                        _, _, output = self.net(LR_pad, Ref, Ref_down, LR_dep, Ref_dep_down)
                else:
                    if not self.args.depth:
                        output = self.net(LR_pad, Ref, Ref_down)
                    else:
                        output = self.net(LR_pad, Ref, Ref_down, LR_dep, Ref_dep_down)
                if output.size() != HR.size():
                    _, _, h, w = HR.size()
                    output = output[:, :, pad_t:pad_t+h, pad_l:pad_l+w]

                to_y = True
                output_img = calculate_PSNR_SSIM.tensor2img(output)
                gt = calculate_PSNR_SSIM.tensor2img(HR)
                output_img = output_img[:, :, [2, 1, 0]]
                gt = gt[:, :, [2, 1, 0]]
                output_img = output_img.astype(np.float32) / 255.0
                gt = gt.astype(np.float32) / 255.0

                if to_y:
                    output_img = calculate_PSNR_SSIM.bgr2ycbcr(output_img, only_y=to_y)
                    gt = calculate_PSNR_SSIM.bgr2ycbcr(gt, only_y=to_y)

                psnr = calculate_PSNR_SSIM.calculate_psnr(output_img * 255, gt * 255)
                ssim = calculate_PSNR_SSIM.calculate_ssim(output_img * 255, gt * 255)
                PSNR.append(psnr)
                SSIM.append(ssim)

        PSNR = np.mean(PSNR)
        SSIM = np.mean(SSIM)

        return PSNR, SSIM

    def save_image(self, tensor, path):
        img = Image.fromarray(((tensor/2.0 + 0.5).data.cpu().numpy()*255).transpose((1, 2, 0)).astype(np.uint8))
        img.save(path)

    def load_networks(self, net_name, resume, strict=False):
        load_path = resume
        network = getattr(self, net_name)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path, map_location=torch.device(self.device))
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        if 'optimizer' in net_name or 'scheduler' in net_name:
            network.load_state_dict(load_net_clean)
        else:
            network.load_state_dict(load_net_clean, strict=strict)


    def save_networks(self, net_name, epoch):
        network = getattr(self, net_name)
        save_filename = '{}_{}.pth'.format(net_name, epoch)
        save_path = os.path.join(self.args.snapshot_save_dir, save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        if not 'optimizer' and not 'scheduler' in net_name:
            for key, param in state_dict.items():
                state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)
