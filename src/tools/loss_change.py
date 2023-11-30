"""
----------------------------------------------------------------------------------------------
PointHMR Official Code
Copyright (c) Deep Computer Vision Lab. (DCVL.) All Rights Reserved
Licensed under the MIT license.
----------------------------------------------------------------------------------------------
Modified from MeshGraphormer (https://github.com/microsoft/MeshGraphormer)
Copyright (c) Microsoft Corporation. All Rights Reserved [see https://github.com/microsoft/MeshGraphormer/blob/main/LICENSE for details]
----------------------------------------------------------------------------------------------
"""

import torch
import src.modeling.data.config as cfg
from src.utils.geometric_layers import orthographic_projection
from torch.nn import functional as F

def keypoint_2d_loss(criterion_keypoints, pred_keypoints_2d, gt_keypoints_2d):
    """
    Compute 2D reprojection loss if 2D keypoint annotations are available.
    The confidence (conf) is binary and indicates whether the keypoints exist or not.
    """
    loss = criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d).mean()
    return loss

def keypoint_3d_loss(criterion_keypoints, pred_keypoints_3d, gt_keypoints_3d):
    """
    Compute 3D keypoint loss if 3D keypoint annotations are available.
    """
    return criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d).mean()

def vertices_loss(criterion_vertices, pred_vertices, gt_vertices):
    """
    Compute per-vertex loss if vertex annotations are available.
    """
    return criterion_vertices(pred_vertices,gt_vertices).mean()

def make_gt(verts_camed, has_smpl, MAP, img_size=112):
    verts_camed = ((verts_camed[has_smpl==1] + 1) * 0.5) * img_size
    x = verts_camed[:, :,0].long()
    y = verts_camed[:, :,1].long()

    indx = img_size*y + x
    flag1 = indx<img_size*img_size
    flag2 = -1 < indx
    flag = flag2*flag1

    GT = MAP[indx[flag]].reshape(-1,1,img_size,img_size).to(verts_camed.device)
    # GT = conv_gauss(GT, device=verts_camed.device)
    #
    # GT[GT==0] = -0.1

    return GT, flag

def calc_heatmap_loss(heatmap,gt,has_smpl,flag, criterion_heatmap):
    pred = heatmap[has_smpl==1][flag]
    return dice_loss(pred.unsqueeze(1),gt.flatten(2))
    # return criterion_heatmap(heatmap,gt)

def dice_loss(pred, target, smooth=1e-5):
    # pred = torch.sigmoid(pred)
    # binary cross entropy loss
    bce = F.binary_cross_entropy(pred, target, reduction='mean')*1e3

    # dice coefficient
    intersection = (pred * target).sum(dim=(1, 2))
    union = (pred).sum(dim=(1, 2)) + (target).sum(dim=(1, 2))
    dice = 2.0 * (intersection + smooth) / (union + 2 * smooth)

    # dice loss
    dice_loss = 1.0 - dice
    # total loss
    loss = dice_loss.mean() + bce

    return loss

def calc_losses_3DJoints_Vertices_simple(args,
                       pred_3d_joints,
                       pred_vertices,
                       gt_3d_joints,
                       gt_vertices,
                       criterion_keypoints,
                       criterion_vertices):
    
    #print(f"\n{pred_3d_joints.shape}\n{gt_3d_joints.shape}")
    #print(f"\n{pred_vertices.shape}\n{gt_vertices.shape}")

    gt_3d_joints = gt_3d_joints[:, :, :-1].clone()
    loss_3d_joints = keypoint_3d_loss(criterion_keypoints,pred_3d_joints,gt_3d_joints)
    loss_vertices = vertices_loss(criterion_vertices, pred_vertices, gt_vertices)
    
    loss = args.joints_loss_weight * loss_3d_joints \
           + args.vertices_loss_weight * loss_vertices
    return loss

def MixSTE_loss(args,
                pred_3d_joints,
                pred_vertices,
                gt_3d_joints,
                gt_vertices):
    gt_3d_joints = gt_3d_joints[:, :, :-1].clone()
    w_mpjpe = torch.tensor([2.5, 2.5, 1, 1, 2.5, 2.5, 4, 4, 1.5, 1.5, 4, 4, 1, 1]).cuda()
    wmpjpe_loss = torch.mean(w_mpjpe * torch.norm(pred_3d_joints - gt_3d_joints, dim=-1))

    velocity_loss = mean_velocity_error_train(pred_3d_joints, gt_3d_joints)

    criterion_vertices = torch.nn.L1Loss().cuda(args.device)
    mesh_loss = criterion_vertices(pred_vertices,gt_vertices)

    loss = args.joints_loss_weight * wmpjpe_loss \
           + args.joints_loss_weight * velocity_loss \
           + args.vertices_loss_weight * mesh_loss
    return loss

def OSX_loss(args,
            pred_camera,
            pred_3d_joints,
            pred_vertices_sub2,
            pred_vertices_sub,
            pred_vertices,
            gt_vertices_sub2,
            gt_vertices_sub,
            gt_vertices,
            gt_3d_joints,
            gt_2d_joints,
            has_smpl,
            criterion_keypoints,
            criterion_2d_keypoints,
            criterion_vertices,
            smpl,
            heatmap,
            criterion_heatmap,
            MAP,
            need_hloss = True):
    # obtain 3d joints, which are regressed from the full mesh
    pred_3d_joints_from_smpl = smpl.module.get_h36m_joints(pred_vertices)
    pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:, cfg.H36M_J17_TO_J14, :]

    # obtain 2d joints, which are projected from 3d joints of smpl mesh
    try:
        pred_2d_joints_from_smpl   = orthographic_projection(pred_3d_joints_from_smpl, pred_camera)
    except:
        print(2)
    pred_2d_joints = orthographic_projection(pred_3d_joints, pred_camera)
    # compute 2d joint loss
    loss_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints, gt_2d_joints) + \
                     keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_smpl, gt_2d_joints)


    # compute 3d joint loss  (where the joints are directly output from transformer)
    loss_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints, gt_3d_joints)
    # compute 3d vertex loss
    loss_vertices = (args.vloss_w_sub2 * vertices_loss(criterion_vertices, pred_vertices_sub2, gt_vertices_sub2) + \
                     args.vloss_w_sub * vertices_loss(criterion_vertices, pred_vertices_sub, gt_vertices_sub) + \
                     args.vloss_w_full * vertices_loss(criterion_vertices, pred_vertices, gt_vertices))
    # compute 3d joint loss (where the joints are regressed from full mesh)
    loss_reg_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints_from_smpl, gt_3d_joints)

    loss_3d_joints = loss_3d_joints + loss_reg_3d_joints
    
    # heatmap loss
    heatmap_loss = 0
    if need_hloss:
        gt_2d_vertices = orthographic_projection(gt_vertices_sub2, pred_camera)
        gt, flag = make_gt(gt_2d_vertices,has_smpl, MAP)

        heatmap_loss = calc_heatmap_loss(heatmap,gt,has_smpl,flag, criterion_heatmap)
    
    # loss
    loss = args.joints_loss_weight * loss_3d_joints \
           + args.vertices_loss_weight * loss_vertices \
           + args.vertices_loss_weight * loss_2d_joints + args.heatmap_loss_weight * heatmap_loss
    
    

    return pred_2d_joints_from_smpl, loss_2d_joints, loss_3d_joints, loss_vertices, loss

def mean_velocity_error_train(predicted, target, axis=0):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    # 8帧向前差分
    velocity_predicted = torch.diff(predicted, dim=axis)
    velocity_target = torch.diff(target, dim=axis)

    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=len(target.shape)-1))

# OSXLoss_2d3dVelocity
def calc_loss_AllMesh_2D_3D_3DVelocity_heatmap(args,
                pred_camera,
                pred_3d_joints,
                pred_vertices_sub2,
                pred_vertices_sub,
                pred_vertices,
                gt_vertices_sub2,
                gt_vertices_sub,
                gt_vertices,
                gt_3d_joints,
                gt_2d_joints,
                has_smpl,
                smpl,
                heatmap,
                criterion_heatmap,
                MAP,
                need_hloss = True):

    # obtain 3d joints, which are regressed from the full mesh
    pred_3d_joints_from_smpl = smpl.module.get_h36m_joints(pred_vertices)
    pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:, cfg.H36M_J17_TO_J14, :]

    # obtain 2d joints, which are projected from 3d joints of smpl mesh
    try:
        pred_2d_joints_from_smpl   = orthographic_projection(pred_3d_joints_from_smpl, pred_camera)
    except:
        print(2)
    pred_2d_joints = orthographic_projection(pred_3d_joints, pred_camera)

    # compute 3d joint loss  (where the joints are directly output from transformer)
    criterion_keypoints = torch.nn.L1Loss().cuda(args.device)

    gt_3d_joints = gt_3d_joints[:, :, :-1]
    # compute 3d joint loss
    loss_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints, gt_3d_joints)
    # compute 3d joint loss (where the joints are regressed from full mesh)
    loss_reg_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints_from_smpl, gt_3d_joints)

    # 3d Joints mean_velocity_error_train
    loss_3d_mean_velocity = mean_velocity_error_train(pred_3d_joints,gt_3d_joints)
    
    loss_3d_joints = loss_3d_joints + loss_reg_3d_joints + loss_3d_mean_velocity

    # compute 2d joint loss
    criterion_2d_keypoints = torch.nn.L1Loss().cuda(args.device)
    gt_2d_joints = gt_2d_joints[:, :, :-1]
    loss_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints, gt_2d_joints) + \
                     keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_smpl, gt_2d_joints)
    # print(pred_2d_joints.shape)
    # print(gt_2d_joints.shape)
    loss_2d_mean_velocity = mean_velocity_error_train(pred_2d_joints, gt_2d_joints)
    loss_2d_joints = loss_2d_joints + loss_2d_mean_velocity

    # compute 3d vertex loss
    criterion_vertices = torch.nn.L1Loss().cuda(args.device)
    loss_vertices = args.vloss_w_sub2 * vertices_loss(criterion_vertices, pred_vertices_sub2, gt_vertices_sub2) + \
                    args.vloss_w_sub * vertices_loss(criterion_vertices, pred_vertices_sub, gt_vertices_sub) + \
                    args.vloss_w_full * vertices_loss(criterion_vertices, pred_vertices, gt_vertices)

    # heatmap loss
    heatmap_loss = 0
    if need_hloss:
        gt_2d_vertices = orthographic_projection(gt_vertices_sub2, pred_camera)
        gt, flag = make_gt(gt_2d_vertices,has_smpl, MAP)

        heatmap_loss = calc_heatmap_loss(heatmap,gt,has_smpl,flag, criterion_heatmap)

    # loss
    loss = args.joints_loss_weight * loss_3d_joints \
           + args.vertices_loss_weight * loss_vertices \
           + args.vertices_loss_weight * loss_2d_joints + args.heatmap_loss_weight * heatmap_loss

    return loss
    # return pred_2d_joints_from_smpl, loss_2d_joints, loss_3d_joints, loss_vertices, loss

def mean_velocity_error_train_2(predicted, target, frame_num = 8, joint_num = 14, axis=1):
    assert predicted.shape == target.shape
    bs,n,c = target.shape
    joint_num = n
    batch_size = predicted.shape[0]//frame_num
    j_dim = predicted.shape[2]

    predicted = predicted.reshape(batch_size,frame_num,joint_num, j_dim)
    target = target.reshape(batch_size,frame_num,joint_num, j_dim)

    velocity_predicted = torch.diff(predicted, dim=axis)
    velocity_target = torch.diff(target, dim=axis)

    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=len(target.shape)-1))

def calc_loss_AllMesh_2D_3D_3DVelocity_heatmap_2(args,
                pred_camera,
                pred_3d_joints,
                pred_vertices_sub2,
                pred_vertices_sub,
                pred_vertices,
                gt_vertices_sub2,
                gt_vertices_sub,
                gt_vertices,
                gt_3d_joints,
                gt_2d_joints,
                has_smpl,
                smpl,
                heatmap,
                criterion_heatmap,
                MAP,
                need_hloss = True):

    # obtain 3d joints, which are regressed from the full mesh
    pred_3d_joints_from_smpl = smpl.module.get_h36m_joints(pred_vertices)
    pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:, cfg.H36M_J17_TO_J14, :]

    # obtain 2d joints, which are projected from 3d joints of smpl mesh
    try:
        pred_2d_joints_from_smpl   = orthographic_projection(pred_3d_joints_from_smpl, pred_camera)
    except:
        print(2)
    pred_2d_joints = orthographic_projection(pred_3d_joints, pred_camera)

    # compute 3d joint loss  (where the joints are directly output from transformer)
    criterion_keypoints = torch.nn.L1Loss().cuda(args.device)
    # criterion_keypoints = F.mse_loss

    gt_3d_joints = gt_3d_joints[:, :, :-1]
    # compute 3d joint loss
    loss_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints, gt_3d_joints)  
    loss_reg_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints_from_smpl, gt_3d_joints)
    
    # 3d Joints mean_velocity_error_train
    loss_3d_mean_velocity = mean_velocity_error_train_2(pred_3d_joints,gt_3d_joints)
    
    loss_3d_joints = loss_3d_joints + loss_reg_3d_joints + loss_3d_mean_velocity

    # compute 2d joint loss
    criterion_2d_keypoints = torch.nn.L1Loss().cuda(args.device)
    # criterion_2d_keypoints = F.mse_loss
    gt_2d_joints = gt_2d_joints[:, :, :-1]
    loss_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints, gt_2d_joints) 
    loss_reg_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_smpl, gt_2d_joints)
    # print(pred_2d_joints.shape)
    # print(gt_2d_joints.shape)
    loss_2d_mean_velocity = mean_velocity_error_train_2(pred_2d_joints, gt_2d_joints)
    loss_2d_joints = loss_2d_joints + loss_reg_2d_joints + loss_2d_mean_velocity

    # compute 3d vertex loss
    criterion_vertices = torch.nn.L1Loss().cuda(args.device)
    loss_vertices = args.vloss_w_sub2 * vertices_loss(criterion_vertices, pred_vertices_sub2, gt_vertices_sub2) + \
                    args.vloss_w_sub * vertices_loss(criterion_vertices, pred_vertices_sub, gt_vertices_sub) + \
                    args.vloss_w_full * vertices_loss(criterion_vertices, pred_vertices, gt_vertices)

    # heatmap loss
    heatmap_loss = 0
    if need_hloss:
        gt_2d_vertices = orthographic_projection(gt_vertices_sub2, pred_camera)
        gt, flag = make_gt(gt_2d_vertices,has_smpl, MAP)

        heatmap_loss = calc_heatmap_loss(heatmap,gt,has_smpl,flag, criterion_heatmap)

    # loss
    loss = args.joints_loss_weight * loss_3d_joints \
           + args.vertices_loss_weight * loss_vertices \
           + args.vertices_loss_weight * loss_2d_joints + args.heatmap_loss_weight * heatmap_loss

    return loss
    # return pred_2d_joints_from_smpl, loss_2d_joints, loss_3d_joints, loss_vertices, loss
    
def calc_loss_AllMesh_2D_3DVelocity_no_heatmap(args,
                pred_camera,
                pred_3d_joints,
                pred_vertices_sub2,
                pred_vertices_sub,
                pred_vertices,
                gt_vertices_sub2,
                gt_vertices_sub,
                gt_vertices,
                gt_3d_joints,
                gt_2d_joints,
                smpl):

    # obtain 3d joints, which are regressed from the full mesh
    pred_3d_joints_from_smpl = smpl.module.get_h36m_joints(pred_vertices)
    pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:, cfg.H36M_J17_TO_J14, :]

    # obtain 2d joints, which are projected from 3d joints of smpl mesh
    try:
        pred_2d_joints_from_smpl   = orthographic_projection(pred_3d_joints_from_smpl, pred_camera)
    except:
        print(2)
    pred_2d_joints = orthographic_projection(pred_3d_joints, pred_camera)

    # compute 3d joint loss  (where the joints are directly output from transformer)
    criterion_keypoints = torch.nn.L1Loss().cuda(args.device)
    # criterion_keypoints = F.mse_loss

    gt_3d_joints = gt_3d_joints[:, :, :-1]
    # compute 3d joint loss
    loss_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints, gt_3d_joints)  
    loss_reg_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints_from_smpl, gt_3d_joints)
    
    # 3d Joints mean_velocity_error_train
    loss_3d_mean_velocity = mean_velocity_error_train_2(pred_3d_joints,gt_3d_joints)
    
    loss_3d_joints = loss_3d_joints + loss_reg_3d_joints + loss_3d_mean_velocity

    # compute 2d joint loss
    criterion_2d_keypoints = torch.nn.L1Loss().cuda(args.device)
    # criterion_2d_keypoints = F.mse_loss
    gt_2d_joints = gt_2d_joints[:, :, :-1]
    loss_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints, gt_2d_joints) 
    loss_reg_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_smpl, gt_2d_joints)
    # print(pred_2d_joints.shape)
    # print(gt_2d_joints.shape)
    loss_2d_mean_velocity = mean_velocity_error_train_2(pred_2d_joints, gt_2d_joints)
    loss_2d_joints = loss_2d_joints + loss_reg_2d_joints + loss_2d_mean_velocity

    # compute 3d vertex loss
    criterion_vertices = torch.nn.L1Loss().cuda(args.device)
    loss_vertices = args.vloss_w_sub2 * vertices_loss(criterion_vertices, pred_vertices_sub2, gt_vertices_sub2) + \
                    args.vloss_w_sub * vertices_loss(criterion_vertices, pred_vertices_sub, gt_vertices_sub) + \
                    args.vloss_w_full * vertices_loss(criterion_vertices, pred_vertices, gt_vertices)


    # loss
    loss = args.joints_loss_weight * loss_3d_joints \
           + args.vertices_loss_weight * loss_vertices \
           + args.vertices_loss_weight * loss_2d_joints

    return loss

def calc_loss_AllMesh_2D3DVelocity_no_smpl_no_heatmap(args,
                pred_camera,
                pred_3d_joints,
                pred_vertices_sub2,
                pred_vertices_sub,
                pred_vertices,
                gt_vertices_sub2,
                gt_vertices_sub,
                gt_vertices,
                gt_3d_joints,
                gt_2d_joints):

    # obtain 3d joints, which are regressed from the full mesh
    '''
    pred_3d_joints_from_smpl = smpl.module.get_h36m_joints(pred_vertices)
    pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:, cfg.H36M_J17_TO_J14, :]

    # obtain 2d joints, which are projected from 3d joints of smpl mesh
    try:
        pred_2d_joints_from_smpl   = orthographic_projection(pred_3d_joints_from_smpl, pred_camera)
    except:
        print(2)
    '''
    pred_2d_joints = orthographic_projection(pred_3d_joints, pred_camera)
    

    # compute 3d joint loss  (where the joints are directly output from transformer)
    criterion_keypoints = torch.nn.L1Loss().cuda(args.device)
    # criterion_keypoints = F.mse_loss

    gt_3d_joints = gt_3d_joints[:, :, :-1]
    # compute 3d joint loss
    loss_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints, gt_3d_joints)  
    # loss_reg_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints_from_smpl, gt_3d_joints)
    
    # 3d Joints mean_velocity_error_train
    loss_3d_mean_velocity = mean_velocity_error_train_2(pred_3d_joints,gt_3d_joints)
    
    loss_3d_joints = loss_3d_joints + loss_3d_mean_velocity

    # compute 2d joint loss
    criterion_2d_keypoints = torch.nn.L1Loss().cuda(args.device)
    # criterion_2d_keypoints = F.mse_loss
    gt_2d_joints = gt_2d_joints[:, :, :-1]
    loss_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints, gt_2d_joints) 
    # loss_reg_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_smpl, gt_2d_joints)
    # print(pred_2d_joints.shape)
    # print(gt_2d_joints.shape)
    loss_2d_mean_velocity = mean_velocity_error_train_2(pred_2d_joints, gt_2d_joints)
    loss_2d_joints = loss_2d_joints + loss_2d_mean_velocity

    # compute 3d vertex loss
    criterion_vertices = torch.nn.L1Loss().cuda(args.device)
    loss_vertices = args.vloss_w_sub2 * vertices_loss(criterion_vertices, pred_vertices_sub2, gt_vertices_sub2) + \
                    args.vloss_w_sub * vertices_loss(criterion_vertices, pred_vertices_sub, gt_vertices_sub) + \
                    args.vloss_w_full * vertices_loss(criterion_vertices, pred_vertices, gt_vertices)


    # loss
    loss = args.joints_loss_weight * loss_3d_joints \
           + args.vertices_loss_weight * loss_vertices \
           + args.vertices_loss_weight * loss_2d_joints

    return loss

'''mPVE:  84.78, mPJPE:  73.42, PAmPJPE:  45.45'''
def calc_loss_SingleMesh_2D3DVelocity_no_heatmap(args,
                pred_camera,
                pred_3d_joints,
                pred_vertices,
                gt_vertices,
                gt_3d_joints,
                gt_2d_joints,
                smpl):

    # obtain 3d joints, which are regressed from the full mesh
    pred_3d_joints_from_smpl = smpl.module.get_h36m_joints(pred_vertices)
    pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:, cfg.H36M_J17_TO_J14, :]

    # obtain 2d joints, which are projected from 3d joints of smpl mesh
    try:
        pred_2d_joints_from_smpl   = orthographic_projection(pred_3d_joints_from_smpl, pred_camera)
    except:
        print(2)
    pred_2d_joints = orthographic_projection(pred_3d_joints, pred_camera)

    # compute 3d joint loss  (where the joints are directly output from transformer)
    criterion_keypoints = torch.nn.L1Loss().cuda(args.device)
    # criterion_keypoints = F.mse_loss

    gt_3d_joints = gt_3d_joints[:, :, :-1]
    # compute 3d joint loss
    loss_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints, gt_3d_joints)  
    loss_reg_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints_from_smpl, gt_3d_joints)
    
    # 3d Joints mean_velocity_error_train
    loss_3d_mean_velocity = mean_velocity_error_train_2(pred_3d_joints,gt_3d_joints)
    
    loss_3d_joints = loss_3d_joints + loss_reg_3d_joints + loss_3d_mean_velocity

    # compute 2d joint loss
    criterion_2d_keypoints = torch.nn.L1Loss().cuda(args.device)
    # criterion_2d_keypoints = F.mse_loss
    gt_2d_joints = gt_2d_joints[:, :, :-1]
    loss_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints, gt_2d_joints) 
    loss_reg_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_smpl, gt_2d_joints)
    # print(pred_2d_joints.shape)
    # print(gt_2d_joints.shape)
    loss_2d_mean_velocity = mean_velocity_error_train_2(pred_2d_joints, gt_2d_joints)
    loss_2d_joints = loss_2d_joints + loss_reg_2d_joints + loss_2d_mean_velocity

    # compute 3d vertex loss
    criterion_vertices = torch.nn.L1Loss().cuda(args.device)
    loss_vertices = vertices_loss(criterion_vertices, pred_vertices, gt_vertices)
                    # args.vloss_w_sub2 * vertices_loss(criterion_vertices, pred_vertices_sub2, gt_vertices_sub2) + \
                    # args.vloss_w_sub * vertices_loss(criterion_vertices, pred_vertices_sub, gt_vertices_sub)
                    
    # loss
    loss = args.joints_loss_weight * loss_3d_joints \
           + args.vertices_loss_weight * loss_vertices \
           + args.vertices_loss_weight * loss_2d_joints

    return loss

def calc_loss_SingleMesh_2D3DVelocity_no_smpl_no_heatmap(args,
                pred_camera,
                pred_3d_joints,
                pred_vertices,
                gt_vertices,
                gt_3d_joints,
                gt_2d_joints):

    # obtain 3d joints, which are regressed from the full mesh
    '''
    pred_3d_joints_from_smpl = smpl.module.get_h36m_joints(pred_vertices)
    pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:, cfg.H36M_J17_TO_J14, :]

    # obtain 2d joints, which are projected from 3d joints of smpl mesh
    try:
        pred_2d_joints_from_smpl   = orthographic_projection(pred_3d_joints_from_smpl, pred_camera)
    except:
        print(2)
    '''
    pred_2d_joints = orthographic_projection(pred_3d_joints, pred_camera)
    

    # compute 3d joint loss  (where the joints are directly output from transformer)
    criterion_keypoints = torch.nn.L1Loss().cuda(args.device)
    # criterion_keypoints = F.mse_loss

    gt_3d_joints = gt_3d_joints[:, :, :-1]
    # compute 3d joint loss
    loss_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints, gt_3d_joints)  
    # loss_reg_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints_from_smpl, gt_3d_joints)
    
    # 3d Joints mean_velocity_error_train
    loss_3d_mean_velocity = mean_velocity_error_train_2(pred_3d_joints,gt_3d_joints)
    
    loss_3d_joints = loss_3d_joints + loss_3d_mean_velocity

    # compute 2d joint loss
    criterion_2d_keypoints = torch.nn.L1Loss().cuda(args.device)
    # criterion_2d_keypoints = F.mse_loss
    gt_2d_joints = gt_2d_joints[:, :, :-1]
    loss_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints, gt_2d_joints) 
    # loss_reg_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_smpl, gt_2d_joints)
    # print(pred_2d_joints.shape)
    # print(gt_2d_joints.shape)
    loss_2d_mean_velocity = mean_velocity_error_train_2(pred_2d_joints, gt_2d_joints)
    loss_2d_joints = loss_2d_joints + loss_2d_mean_velocity

    # compute 3d vertex loss
    criterion_vertices = torch.nn.L1Loss().cuda(args.device)
    loss_vertices = vertices_loss(criterion_vertices, pred_vertices, gt_vertices) # * args.vloss_w_full + \
                    # args.vloss_w_sub2 * vertices_loss(criterion_vertices, pred_vertices_sub2, gt_vertices_sub2) + \
                    # args.vloss_w_sub * vertices_loss(criterion_vertices, pred_vertices_sub, gt_vertices_sub) + \
                     


    # loss
    loss = args.joints_loss_weight * loss_3d_joints \
           + args.vertices_loss_weight * loss_vertices \
           + args.vertices_loss_weight * loss_2d_joints

    return loss

def calc_loss_SingleMesh_Mesh2D3DVelocity_no_heatmap(args,
                pred_camera,
                pred_3d_joints,
                pred_vertices,
                gt_vertices,
                gt_3d_joints,
                gt_2d_joints,
                smpl):

    # obtain 3d joints, which are regressed from the full mesh
    pred_3d_joints_from_smpl = smpl.module.get_h36m_joints(pred_vertices)
    pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:, cfg.H36M_J17_TO_J14, :]

    # obtain 2d joints, which are projected from 3d joints of smpl mesh
    try:
        pred_2d_joints_from_smpl   = orthographic_projection(pred_3d_joints_from_smpl, pred_camera)
    except:
        print(2)
    pred_2d_joints = orthographic_projection(pred_3d_joints, pred_camera)

    # compute 3d joint loss  (where the joints are directly output from transformer)
    criterion_keypoints = torch.nn.L1Loss().cuda(args.device)
    # criterion_keypoints = F.mse_loss

    gt_3d_joints = gt_3d_joints[:, :, :-1]
    # compute 3d joint loss
    loss_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints, gt_3d_joints)  
    loss_reg_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints_from_smpl, gt_3d_joints)
    
    # 3d Joints mean_velocity_error_train
    loss_3d_mean_velocity = mean_velocity_error_train_2(pred_3d_joints,gt_3d_joints)
    
    loss_3d_joints = loss_3d_joints + loss_reg_3d_joints + loss_3d_mean_velocity

    # compute 2d joint loss
    criterion_2d_keypoints = torch.nn.L1Loss().cuda(args.device)
    # criterion_2d_keypoints = F.mse_loss
    gt_2d_joints = gt_2d_joints[:, :, :-1]
    loss_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints, gt_2d_joints) 
    loss_reg_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_smpl, gt_2d_joints)
    # print(pred_2d_joints.shape)
    # print(gt_2d_joints.shape)
    loss_2d_mean_velocity = mean_velocity_error_train_2(pred_2d_joints, gt_2d_joints)
    loss_2d_joints = loss_2d_joints + loss_reg_2d_joints + loss_2d_mean_velocity

    # compute 3d vertex loss
    criterion_vertices = torch.nn.L1Loss().cuda(args.device)
    #loss_vertices = args.vloss_w_full * mean_velocity_error_train_2(pred_vertices, gt_vertices, joint_num = pred_vertices.shape[1])
                    # args.vloss_w_sub2 * vertices_loss(criterion_vertices, pred_vertices_sub2, gt_vertices_sub2) + \
                    # args.vloss_w_sub * vertices_loss(criterion_vertices, pred_vertices_sub, gt_vertices_sub)
    loss_vertices = mean_velocity_error_train_2(pred_vertices, gt_vertices, joint_num = pred_vertices.shape[1]) + \
                    vertices_loss(criterion_vertices, pred_vertices, gt_vertices)

    # loss
    loss = args.joints_loss_weight * loss_3d_joints \
           + args.vertices_loss_weight * loss_vertices \
           + args.vertices_loss_weight * loss_2d_joints

    return loss


def calc_loss_SingleMesh_2D3DVelocity_changeWeight_no_heatmap(args,
                pred_camera,
                pred_3d_joints,
                pred_vertices,
                gt_vertices,
                gt_3d_joints,
                gt_2d_joints,
                smpl,
                weight_3d = 1000,
                weight_2d = 100,
                weight_vertices = 100):

    # obtain 3d joints, which are regressed from the full mesh
    pred_3d_joints_from_smpl = smpl.module.get_h36m_joints(pred_vertices)
    pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:, cfg.H36M_J17_TO_J14, :]

    # obtain 2d joints, which are projected from 3d joints of smpl mesh
    try:
        pred_2d_joints_from_smpl   = orthographic_projection(pred_3d_joints_from_smpl, pred_camera)
    except:
        print(2)
    pred_2d_joints = orthographic_projection(pred_3d_joints, pred_camera)

    # compute 3d joint loss  (where the joints are directly output from transformer)
    criterion_keypoints = torch.nn.L1Loss().cuda(args.device)
    # criterion_keypoints = F.mse_loss

    gt_3d_joints = gt_3d_joints[:, :, :-1]
    # compute 3d joint loss
    loss_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints, gt_3d_joints)  
    loss_reg_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints_from_smpl, gt_3d_joints)
    
    # 3d Joints mean_velocity_error_train
    loss_3d_mean_velocity = mean_velocity_error_train_2(pred_3d_joints,gt_3d_joints)
    
    loss_3d_joints = loss_3d_joints + loss_reg_3d_joints + loss_3d_mean_velocity

    # compute 2d joint loss
    criterion_2d_keypoints = torch.nn.L1Loss().cuda(args.device)
    # criterion_2d_keypoints = F.mse_loss
    gt_2d_joints = gt_2d_joints[:, :, :-1]
    loss_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints, gt_2d_joints) 
    loss_reg_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_smpl, gt_2d_joints)
    # print(pred_2d_joints.shape)
    # print(gt_2d_joints.shape)
    loss_2d_mean_velocity = mean_velocity_error_train_2(pred_2d_joints, gt_2d_joints)
    loss_2d_joints = loss_2d_joints + loss_reg_2d_joints + loss_2d_mean_velocity

    # compute 3d vertex loss
    criterion_vertices = torch.nn.L1Loss().cuda(args.device)
    loss_vertices = vertices_loss(criterion_vertices, pred_vertices, gt_vertices)
                    # args.vloss_w_sub2 * vertices_loss(criterion_vertices, pred_vertices_sub2, gt_vertices_sub2) + \
                    # args.vloss_w_sub * vertices_loss(criterion_vertices, pred_vertices_sub, gt_vertices_sub)
                    
    # loss
    
    loss = weight_3d * loss_3d_joints \
           + weight_vertices * loss_vertices \
           + weight_2d * loss_2d_joints

    return loss

def calc_loss_noMesh_2D3DVelocity_changeWeight(args,
                pred_camera,
                pred_3d_joints,
                pred_vertices,
                gt_3d_joints,
                gt_2d_joints,
                smpl,
                weight_3d = 1000,
                weight_2d = 100):
    # obtain 3d joints, which are regressed from the full mesh
    pred_3d_joints_from_smpl = smpl.module.get_h36m_joints(pred_vertices)
    pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:, cfg.H36M_J17_TO_J14, :]

    # obtain 2d joints, which are projected from 3d joints of smpl mesh
    try:
        pred_2d_joints_from_smpl   = orthographic_projection(pred_3d_joints_from_smpl, pred_camera)
    except:
        print(2)
    pred_2d_joints = orthographic_projection(pred_3d_joints, pred_camera)

    # compute 3d joint loss  (where the joints are directly output from transformer)
    criterion_keypoints = torch.nn.L1Loss().cuda(args.device)
    # criterion_keypoints = F.mse_loss

    gt_3d_joints = gt_3d_joints[:, :, :-1]
    # compute 3d joint loss
    loss_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints, gt_3d_joints)  
    loss_reg_3d_joints = keypoint_3d_loss(criterion_keypoints, pred_3d_joints_from_smpl, gt_3d_joints)
    
    # 3d Joints mean_velocity_error_train
    loss_3d_mean_velocity = mean_velocity_error_train_2(pred_3d_joints,gt_3d_joints)
    
    loss_3d_joints = loss_3d_joints + loss_reg_3d_joints + loss_3d_mean_velocity

    # compute 2d joint loss
    criterion_2d_keypoints = torch.nn.L1Loss().cuda(args.device)
    # criterion_2d_keypoints = F.mse_loss
    gt_2d_joints = gt_2d_joints[:, :, :-1]
    loss_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints, gt_2d_joints) 
    loss_reg_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_smpl, gt_2d_joints)
    # print(pred_2d_joints.shape)
    # print(gt_2d_joints.shape)
    loss_2d_mean_velocity = mean_velocity_error_train_2(pred_2d_joints, gt_2d_joints)
    loss_2d_joints = loss_2d_joints + loss_reg_2d_joints + loss_2d_mean_velocity

    # compute 3d vertex loss
    # criterion_vertices = torch.nn.L1Loss().cuda(args.device)
    # loss_vertices = vertices_loss(criterion_vertices, pred_vertices, gt_vertices)
                    # args.vloss_w_sub2 * vertices_loss(criterion_vertices, pred_vertices_sub2, gt_vertices_sub2) + \
                    # args.vloss_w_sub * vertices_loss(criterion_vertices, pred_vertices_sub, gt_vertices_sub)
                    
    # loss
    
    loss = weight_3d * loss_3d_joints \
        + weight_2d * loss_2d_joints
           # + weight_vertices * loss_vertices \
           

    return loss

