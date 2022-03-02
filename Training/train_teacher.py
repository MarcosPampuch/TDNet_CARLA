import os
import oyaml as yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
from torch.nn.parallel.scatter_gather import gather
from torch.utils import data
from tqdm import tqdm
from encoding.parallel import DataParallelModel, DataParallelCriterion
from ptsemseg.models import get_model
#from ptsemseg.loss import get_loss_function
from ptsemseg.loss.lovasz_loss import OhemCE, smooth_one_hot, cross_entropy_with_probs, DeepLabCE, FocalLoss, calculate_weigths
from ptsemseg.loader import get_loader 
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.optimizers import get_optimizer
from ptsemseg.utils import convert_state_dict,clean_state_dict
import pdb
import sys
sys.path.insert(1,"/home/2018015/mgrass01/TDNet")
from Testing.model.pspnet.pspnet import pspnet


def init_seed(manual_seed, en_cudnn=False):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = en_cudnn
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    random.seed(manual_seed)

def train(cfg, logger, logdir):
    # Setup seeds
    init_seed(11733, en_cudnn=False)

    # Setup Augmentations
    train_augmentations = cfg["training"].get("train_augmentations", None)
    t_data_aug = get_composed_augmentations(train_augmentations)
    val_augmentations = cfg["validating"].get("val_augmentations", None)
    v_data_aug = get_composed_augmentations(val_augmentations)

    # Setup Dataloader


#    data_loader = get_loader(cfg["data"]["dataset"])
#    data_path = cfg["data"]["path"]
    data_loader = get_loader('CARLA')
    data_path = '/gpfs1/dlocal/home/2020010/PARTAGE/CARLA_NEW'

#    t_loader = data_loader(data_path,split=cfg["data"]["train_split"],augmentations=t_data_aug,path_num=path_n)
#    v_loader = data_loader(data_path,split=cfg["data"]["val_split"],augmentations=v_data_aug,path_num=path_n)
    
    t_loader = data_loader(data_path, mode='train',
                          time_step= 1,
                          transform=True,
                          resize= None,
                          type_carla=1)
    
    v_loader = data_loader(data_path, mode='val',
                        time_step= 1,
                        transform=True,
                        resize= None,
                        type_carla=1)

    trainloader = data.DataLoader(t_loader,
                                  batch_size=cfg["training"]["batch_size"],
                                  num_workers=cfg["training"]["n_workers"],
                                  shuffle=True,
                                  drop_last=True  )
    valloader = data.DataLoader(v_loader,
                                batch_size=cfg["validating"]["batch_size"],
                                num_workers=cfg["validating"]["n_workers"] )

    logger.info("Using training seting {}".format(cfg["training"]))
    
    # Setup Metrics
    running_metrics_val = runningScore(t_loader.n_classes)

    # Setup Model and Loss
#    loss_fn = get_loss_function(cfg["training"])
#    teacher = get_model(cfg["teacher"], t_loader.n_classes)
#    model = get_model(cfg["model"],t_loader.n_classes, loss_fn, cfg["training"]["resume"],teacher)
    model = get_model(cfg["model"], t_loader.n_classes)
#    logger.info("Using loss {}".format(loss_fn))

    # Setup optimizer
    optimizer = get_optimizer(cfg["training"], model)

    # Setup Multi-GPU
    model = DataParallelModel(model, device_ids=[0]).cuda()

    #Initialize training param
    cnt_iter = 0
    best_iou = 0.0
    time_meter = averageMeter()

    while cnt_iter <= cfg["training"]["train_iters"]:
        for (f_img, labels) in trainloader:
            cnt_iter += 1
            model.train()
            f_img = f_img[-1].cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
#            f_img = f_img.cuda()
#            labels = labels.cuda()
            
#            target_tensor = smooth_one_hot(labels, classes = 13, smoothing = 0.1, lb_ignore= 250)
            start_ts = time.time()
            outputs = model(f_img,labels)

#            seg_loss = gather(outputs, 0)
#            seg_loss = torch.mean(seg_loss)
            loss =criterion(outputs, labels)
            loss.backward()
            time_meter.update(time.time() - start_ts)

            optimizer.step()

            if (cnt_iter + 1) % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(
                                            cnt_iter + 1,
                                            cfg["training"]["train_iters"],
                                            loss.item(),
                                            time_meter.avg / cfg["training"]["batch_size"], )

                print(print_str)
                logger.info(print_str)
                time_meter.reset()

            if (cnt_iter + 1) % cfg["training"]["val_interval"] == 0 or (cnt_iter + 1) == cfg["training"]["train_iters"]:
                model.eval()
                with torch.no_grad():
                    for i_val, (f_img_val, labels_val) in tqdm(enumerate(valloader)):
                        
                        f_img_val = f_img_val[-1]
                        
                        outputs = model(f_img_val)
#                        outputs = gather(outputs, 0, dim=0)
                        
                        pred = outputs.data.max(1)[1].cpu().numpy()
                        gt = labels_val.data.cpu().numpy()

                        running_metrics_val.update(gt, pred)

                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k, v)
                    logger.info("{}: {}".format(k, v))

                for k, v in class_iou.items():
                    logger.info("{}: {}".format(k, v))

                running_metrics_val.reset()

                if score["Mean IoU : \t"] >= best_iou:
                    best_iou = score["Mean IoU : \t"]
                    state = {
                        "epoch": cnt_iter + 1,
                        "model_state": clean_state_dict(model.module.state_dict(),'teacher'),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(logdir,
                        "{}_{}_best_model.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                    )
                    torch.save(state, save_path)
class_weights = np.array((8.7140, 31.2160, 30.7612, 27.5623, 27.9024, 37.7695,  8.1916,  4.9276,
             6.8403, 33.7536, 17.6963, 46.7649,  3.3284),dtype = np.float32) # caarla_50000
    
class_weights=torch.from_numpy(class_weights).float()
#criterion_val = torch.nn.CrossEntropyLoss(weight=class_weights.cuda(), ignore_index=250, size_average=True)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights.cuda(), ignore_index=250, size_average=True)

#os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    run_id = random.randint(1, 100000)
    logdir = os.path.join("runs", os.path.basename(args.config)[:-4], str(run_id))
    if not os.path.exists(os.path.join("runs", os.path.basename(args.config)[:-4])):
        os.mkdir(os.path.join("runs", os.path.basename(args.config)[:-4]))

    os.mkdir(logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Let the games begin")

    train(cfg, logger, logdir)
