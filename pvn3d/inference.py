import torch
import torch.nn as nn
import torchvision.transforms as transforms

import os
import cv2
import pcl
import argparse
import numpy as np
from PIL import Image
import pickle as pkl

from lib import PVN3D
from common import Config
from lib.utils.basic_utils import Basic_Utils
from datasets.ycb.ycb_dataset import YCB_Dataset
from datasets.linemod.linemod_dataset import LM_Dataset
from lib.utils.sync_batchnorm import convert_model
from lib.utils.pvn3d_eval_utils import cal_frame_poses, cal_frame_poses_lm
from lib.utils.basic_utils import Basic_Utils


config = Config(dataset_name='ycb')
bs_utils = Basic_Utils(config)
DEBUG = False

def get_normal(cld):
    cloud = pcl.PointCloud()
    cld = cld.astype(np.float32)
    cloud.from_array(cld)
    ne = cloud.make_NormalEstimation()
    kdtree = cloud.make_kdtree()
    ne.set_SearchMethod(kdtree)
    ne.set_KSearch(50)
    n = ne.compute()
    n = n.to_array()
    return n

def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system('mkdir -p {}'.format(fd))

def checkpoint_state(model=None, optimizer=None, best_prec=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None
    return {
        "epoch": epoch,
        "it": it,
        "best_prec": best_prec,
        "model_state": model_state,
        "optimizer_state": optim_state,
    }

def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
    filename = "{}.pth.tar".format(filename)

    if os.path.isfile(filename):
        print("==> Loading from checkpoint '{}'".format(filename))
        try:
            checkpoint = torch.load(filename)
        except:
            checkpoint = pkl.load(open(filename, "rb"))
        epoch = checkpoint["epoch"]
        it = checkpoint.get("it", 0.0)
        best_prec = checkpoint["best_prec"]
        if model is not None and checkpoint["model_state"] is not None:
            model.load_state_dict(checkpoint["model_state"])
        if optimizer is not None and checkpoint["optimizer_state"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        print("==> Done")
        return it, epoch, best_prec
    else:
        print("==> Checkpoint '{}' not found".format(filename))
        return None

def preprocess_rgbd(image, depth):

    K = config.intrinsic_matrix['ycb_K1']

    cam_scale = 25
    msk_dp = depth > 1e-6

    dpt = bs_utils.fill_missing(depth, cam_scale, 1)
    msk_dp = dpt > 1e-6

    rgb = np.transpose(image, (2, 0, 1))
    cld, choose = bs_utils.dpt_2_cld(dpt, cam_scale, K)
    normal = get_normal(cld)[:, :3]
    normal[np.isnan(normal)] = 0.0

    rgb_lst = []
    for ic in range(rgb.shape[0]):
        rgb_lst.append(
            rgb[ic].flatten()[choose].astype(np.float32)
        )
    rgb_pt = np.transpose(np.array(rgb_lst), (1, 0)).copy()

    choose = np.array([choose])
    choose_2 = np.array([i for i in range(len(choose[0, :]))])

    if len(choose_2) < 400:
        return None
    if len(choose_2) > config.n_sample_points:
        c_mask = np.zeros(len(choose_2), dtype=int)
        c_mask[:config.n_sample_points] = 1
        np.random.shuffle(c_mask)
        choose_2 = choose_2[c_mask.nonzero()]
    else:
        choose_2 = np.pad(choose_2, (0, config.n_sample_points-len(choose_2)), 'wrap')

    cld_rgb_nrm = np.concatenate((cld, rgb_pt, normal), axis=1)
    cld = cld[choose_2, :]
    cld_rgb_nrm = cld_rgb_nrm[choose_2, :]
    choose = choose[:, choose_2]

    cls_id_lst = np.array(range(1, 22))

    return torch.from_numpy(rgb.astype(np.float32)), \
        torch.from_numpy(cld.astype(np.float32)), \
        torch.from_numpy(cld_rgb_nrm.astype(np.float32)), \
        torch.LongTensor(choose.astype(np.int32)), \
        torch.LongTensor(cls_id_lst.astype(np.int32))

def perform_inference(ckpt, rgb, cld, cld_rgb_nrm, choose, cls_id_lst):
    rgb = rgb.reshape((1, rgb.shape[0], rgb.shape[1], rgb.shape[2]))
    cld = cld.reshape((1, cld.shape[0], cld.shape[1]))
    cld_rgb_nrm = cld_rgb_nrm.reshape((1, cld_rgb_nrm.shape[0], cld_rgb_nrm.shape[1]))
    choose = choose.reshape((1, choose.shape[0], choose.shape[1]))
    cls_id_lst = cls_id_lst.reshape((1, cls_id_lst.shape[0]))
    model = PVN3D(
        num_classes=config.n_objects, pcld_input_channels=6, pcld_use_xyz=True,
        num_points=config.n_sample_points
    ).cuda()
    model = convert_model(model)
    model.cuda()

    cld = cld.cuda()
    rgb = rgb.cuda()
    cld_rgb_nrm = cld_rgb_nrm.cuda()
    choose = choose.cuda()

    # load status from checkpoint
    checkpoint_status = load_checkpoint(model, None, filename=ckpt[:-8])
    model = nn.DataParallel(model)

    model.eval()
    with torch.set_grad_enabled(False):
        pred_kp_of, pred_rgbd_seg, pred_ctr_of = model(
            cld_rgb_nrm, rgb, choose
        )
        _, classes_rgbd = torch.max(pred_rgbd_seg, -1)

        pred_cls_ids, pred_pose_lst = cal_frame_poses(
            cld[0], classes_rgbd[0], pred_ctr_of[0], pred_kp_of[0], True,
            config.n_objects, True
        )

        np_rgb = rgb.cpu().numpy().astype("uint8")[0].transpose(1, 2, 0).copy()
        np_rgb = np_rgb[:, :, ::-1].copy()
        ori_rgb = np_rgb.copy()

        for cls_id in cls_id_lst[0].cpu().numpy():
            idx = np.where(pred_cls_ids == cls_id)[0]
            if len(idx) == 0:
                continue
            pose = pred_pose_lst[idx[0]]
            obj_id = int(cls_id
            )
            mesh_pts = bs_utils.get_pointxyz(obj_id, ds_type='ycb').copy()
            mesh_pts = np.dot(mesh_pts, pose[:, :3].T) + pose[:, 3]
            K = config.intrinsic_matrix["ycb_K1"]
            mesh_p2ds = bs_utils.project_p3d(mesh_pts, 1.0, K)
            color = bs_utils.get_label_color(obj_id, n_obj=22, mode=1)
            np_rgb = bs_utils.draw_p2ds(np_rgb, mesh_p2ds, color=color)
        vis_dir = os.path.join(config.log_eval_dir, "pose_vis")
        ensure_fd(vis_dir)
        f_pth = os.path.join(vis_dir, "out.jpg")
        cv2.imwrite(f_pth, np_rgb)
    
if __name__ == '__main__':
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-wp', '--weights_path', type=str, help='path to the pretrained weights file')
    argparser.add_argument('-img', '--image', type=str, help='path to background images for synthetic data')
    argparser.add_argument('-dep', '--depth', type=str, help='number of synthetic training data samples')

    args = argparser.parse_args()

    img = np.array(Image.open(args.image))
    dep = np.array(Image.open(args.depth))

    rgb, cld, cld_rgb_nrm, choose, cls_id_lst = preprocess_rgbd(img, dep)

    perform_inference(args.weights_path, rgb, cld, cld_rgb_nrm, choose, cls_id_lst)
