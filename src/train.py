import argparse
import datetime, time
import os, gc
import traceback

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
# from torchsampler import ImbalancedDatasetSampler
from tqdm.autonotebook import tqdm

from backbone import EfficientDetBackbone
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from efficientdet.loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, Params, boolean_string, createConfusionMatrix


def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('--config', type=str, default='config/202212/V01/V01.yml', required=True)
    parser.add_argument('--head_only', type=boolean_string, default=True,
                        help='whether finetunes only the regressor and the classifier, (in this project, bifpn is added head only: True casse)'
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--debug', type=boolean_string, default=False,
                        help='whether visualize the predicted boxes of training, '
                             'the output images will be in test/')
    args = parser.parse_args()

    return args

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def save_checkpoint(model, name, saved_path: str):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(saved_path, name))
    else:
        torch.save(model.model.state_dict(), os.path.join(saved_path, name))


class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss


def train(opt):
    now=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    params = Params(opt.config)
    
    # classes = tuple(params.obj_list)

    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    if params.load_weights.endswith('.pth'):
        params.saved_path = params.saved_path + f'/{params.project_name}/{now}'
    params.log_path = params.log_path + f'/{params.project_name}/tensorboard/'
    os.makedirs(params.log_path, exist_ok=True)
    os.makedirs(params.saved_path, exist_ok=True)

    training_params = {'batch_size': params.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': params.num_workers}

    val_params = {'batch_size': params.batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': params.num_workers}
    
    input_sizes = params.input_sizes
    training_set = CocoDataset(root_dir=os.path.join(params.ROOT, params.project_name), set=params.train_set,
                               transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                             Augmenter(),
                                                             Resizer(input_sizes[params.compound_coef])]))
                                                            
    training_generator = DataLoader(training_set, 
                                    **training_params)

    val_set = CocoDataset(root_dir=os.path.join(params.ROOT, params.project_name), set=params.val_set,
                          transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                        Resizer(input_sizes[params.compound_coef])]))
    val_generator = DataLoader(val_set, **val_params)

    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=params.compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))
    # Use confusion matrix function
    feed_model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=params.compound_coef,
                                      ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))

    # load last weights
    if params.load_weights is not None:
        if params.load_weights.endswith('.pth'):
            weights_path = params.load_weights
        else:
            weights_path = get_last_weights(params.saved_path)
        try:
            last_step = int(os.path.basename(weights_path).split('_')[-2].split('.')[0])
        except:
            last_step = 0

        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
        print('[Info] initializing weights...')
        init_weights(model)

    # freeze backbone if train head_only
    if opt.head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            # for ntl in ['EfficientNet', 'BiFPN']:
            for ntl in ['EfficientNet']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        feed_model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if params.num_gpus > 1 and params.batch_size // params.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        feed_model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    writer = SummaryWriter(params.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=opt.debug)

    if params.num_gpus > 0:
        model = model.cuda()
        if params.num_gpus > 1:
            model = CustomDataParallel(model, params.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)

    if params.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), params.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), params.lr, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    step = max(0, last_step)
    model.train()
    
    num_iter_per_epoch = len(training_generator)

    try:
        for epoch in range(params.num_epochs):
            epoch_start = time.time()
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            epoch_loss = []
            epoch_timer = 0
            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):
                iter_start = torch.cuda.Event(enable_timing=True)
                iter_end = torch.cuda.Event(enable_timing=True)
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    imgs = data['img']
                    annot = data['annot']

                    if params.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    optimizer.zero_grad()
                    iter_start.record()
                    cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    iter_end.record()
                    torch.cuda.synchronize()
                    iter_comp_dur = float(iter_start.elapsed_time(iter_end)) / 1000.0
                    epoch_timer += iter_comp_dur

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                            step, epoch, params.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                            reg_loss.item(), loss.item()))
                    writer.add_scalars('Loss', {'train': loss}, step)
                    writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                    writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)
                    
                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)
                    
                    step += 1
                 
                    if step % params.save_interval == 0 and step > 0:
                        save_checkpoint(model, f'efficientdet-d{params.compound_coef}_{epoch}_{step}.pth', saved_path=params.saved_path)
                        print('checkpoint...')

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue
            scheduler.step(np.mean(epoch_loss))
            epoch_end = time.time()

            print("####### Comp Time Cost for Epoch: {} is {}, os time: {}".format(epoch, epoch_timer, epoch_end - epoch_start))

            if epoch % params.val_interval == 0:
                model.eval()
                loss_regression_ls = []
                loss_classification_ls = []
                for iter, data in enumerate(val_generator):
                    with torch.no_grad():
                        imgs = data['img']
                        annot = data['annot']

                        if params.num_gpus == 1:
                            imgs = imgs.cuda()
                            annot = annot.cuda()

                        cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss = cls_loss + reg_loss
                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss_classification_ls.append(cls_loss.item())
                        loss_regression_ls.append(reg_loss.item())

                cls_loss = np.mean(loss_classification_ls)
                reg_loss = np.mean(loss_regression_ls)
                loss = cls_loss + reg_loss

                print(
                    'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                        epoch, params.num_epochs, cls_loss, reg_loss, loss))
                writer.add_scalars('Loss', {'val': loss}, step)
                writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
                writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)
                
                if loss + params.es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch

                    save_checkpoint(model, f'efficientdet-d{params.compound_coef}_{epoch}_{step}_best.pth', saved_path=params.saved_path)

                model.train()

                # Early stopping
                if epoch - best_epoch > params.es_patience > 0:
                    print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                    break
    except KeyboardInterrupt:
        save_checkpoint(model=model, name=f'efficientdet-d{params.compound_coef}_{epoch}_{step}.pth', saved_path=params.saved_path)
        writer.close()
    writer.close()


if __name__ == '__main__':
    opt = get_args()
    train(opt)
    