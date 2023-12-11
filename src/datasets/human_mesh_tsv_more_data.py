"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""

import cv2
import math
import json
from PIL import Image
import os.path as op
import numpy as np
import code

from src.utils.tsv_file import TSVFile, CompositeTSVFile
from src.utils.tsv_file_ops import load_linelist_file, load_from_yaml_file, find_file_path_in_yaml
from src.utils.image_ops import img_from_base64, crop, flip_img, flip_pose, flip_kp, transform, rot_aa
import torch
import torchvision.transforms as transforms


class MeshTSVDataset(object):
    def __init__(self, img_file, label_file=None, hw_file=None,
                 linelist_file=None, is_train=True, cv2_output=False, scale_factor=1):

        self.img_file = img_file
        self.label_file = label_file
        print(label_file)
        self.hw_file = hw_file
        self.linelist_file = linelist_file
        self.img_tsv = self.get_tsv_file(img_file)
        self.label_tsv = None if label_file is None else self.get_tsv_file(label_file)
        self.hw_tsv = None if hw_file is None else self.get_tsv_file(hw_file)

        if self.is_composite:
            assert op.isfile(self.linelist_file)
            self.line_list = [i for i in range(self.hw_tsv.num_rows())]
        else:
            self.line_list = load_linelist_file(linelist_file)

        self.cv2_output = cv2_output
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.is_train = is_train
        self.is_train = is_train

        if self.is_train:
            self.person_dict = {'courtyard_goodNews_00': 2, 'courtyard_jacket_00': 1, 'outdoors_slalom_00': 1, 'courtyard_giveDirections_00': 2, 'courtyard_arguing_00': 2, 'outdoors_freestyle_00': 1, 'outdoors_climbing_02': 1, 'courtyard_rangeOfMotions_00': 2, 'courtyard_captureSelfies_00': 2, 'courtyard_basketball_00': 2, 'outdoors_slalom_01': 1, 'courtyard_box_00': 1, 'courtyard_laceShoe_00': 1, 'courtyard_dancing_01': 2, 'courtyard_capoeira_00': 2, 'courtyard_shakeHands_00': 2, 'courtyard_warmWelcome_00': 2, 'outdoors_climbing_01': 1, 'courtyard_bodyScannerMotions_00': 1, 'courtyard_backpack_00': 1, 'outdoors_climbing_00': 1, 'courtyard_relaxOnBench_00': 1, 'courtyard_golf_00': 1, 'courtyard_relaxOnBench_01': 1}
        else: 
            self.person_dict = {'downtown_walkBridge_01': 1, 'downtown_rampAndStairs_00': 2, 'downtown_runForBus_01': 2, 'downtown_bus_00': 2, 'downtown_warmWelcome_00': 2, 'downtown_runForBus_00': 2, 'downtown_weeklyMarket_00': 1, 'downtown_walking_00': 2, 'downtown_walkUphill_00': 1, 'downtown_car_00': 2, 'downtown_bar_00': 2, 'downtown_stairs_00': 1, 'downtown_upstairs_00': 1, 'downtown_arguing_00': 2, 'downtown_windowShopping_00': 1, 'flat_guitar_01': 1, 'office_phoneCall_00': 2, 'downtown_crossStreets_00': 2, 'outdoors_fencing_01': 1, 'downtown_sitOnStairs_00': 2, 'downtown_cafe_00': 2, 'flat_packBags_00': 1, 'downtown_enterShop_00': 1, 'downtown_downstairs_00': 1}
        


        self.scale_factor = 0.25 # rescale bounding boxes by a factor of [1-options.scale_factor,1+options.scale_factor]
        self.noise_factor = 0.4
        self.rot_factor = 30 # Random rotation in the range [-rot_factor, rot_factor]
        self.img_res = 224

        self.image_keys = self.prepare_image_keys()
        # self.imgname = self.data['imgname']
        self.need_all = []
        num = 0
        self.step = 8
        name_all = {}
        self.name_key = []
        if self.is_train:
            for i in range(len(self.image_keys)):
                
                name = self.image_keys[i]
                # if "S1_Directions_1.54138969" not in name:
                #     continue
                # if "S1_Discussion_1.55011271" not in name:
                #     continue
                if name.split("/")[0]=="images":
                    name = name.split("/")[1]
                    name = name.split("_")[:-1]
                    name = "_".join(name)
                else:
                    name = name.split("/")[0]
                # if len(name_all.keys())>9:
                #     continue
                if name not in name_all.keys():
                    # print(self.image_keys[i],name)
                    name_all[name] =  [[str(self.image_keys[i])],[i]]
                    self.name_key.append(name[0])
                else:
                    name_all[name][0].append(str(self.image_keys[i]))
                    name_all[name][1].append(i)
            #排序
            for name in name_all.keys():
                list1 = name_all[name][0]
                list2 = name_all[name][1]
                list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2))))
                name_all[name] = [list1, list2]
        else:
            for i in range(len(self.image_keys)):
                
                name = self.image_keys[i]
                # if "S11_Walking_1" not in name:
                #     continue
                
                if name.split("/")[0]=="images":
                    name = name.split("/")[1]
                    name = name.split("_")[:-1]
                    name = "_".join(name)
                else:
                    name = name.split("/")[0]
                if name not in name_all.keys():
                    # print(self.image_keys[i],name)
                    name_all[name] =  [[str(self.image_keys[i])],[i]]
                    self.name_key.append(name)
                else:
                    name_all[name][0].append(str(self.image_keys[i]))
                    name_all[name][1].append(i)
            #排序
            for name in name_all.keys():
                list1 = name_all[name][0]
                list2 = name_all[name][1]
                list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2))))
                name_all[name] = [list1, list2]
        self.name_all = name_all
        self.name_list_in8 = []
        step = self.step
	for key in name_all.keys():
            this_name_all = self.name_all[key]
            name_ =  self.get_img_key(this_name_all[1][0])


            if name_.split("/")[0]=="images":
                name_ = name_.split("/")[1]
                name_ = name_.split(".")[0]
            else:
                name_ = name_.split("/")[0]
            if name_ in self.person_dict.keys():
                person_num = self.person_dict[name_]
            else:
                person_num = 1
            for pn in range(person_num):
                idx_name_all = this_name_all[1][pn::person_num]

                for index in range(0,len(idx_name_all[:-self.step])+1,self.step): 

                    self.name_list_in8.append(idx_name_all[index:index+self.step])
                if len(idx_name_all[-self.step:])%self.step != 0:
                    
                    self.name_list_in8.append(idx_name_all[-self.step:])
        self.joints_definition = ('R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder',
        'L_Elbow','L_Wrist','Neck','Top_of_Head','Pelvis','Thorax','Spine','Jaw','Head','Nose','L_Eye','R_Eye','L_Ear','R_Ear')
        self.pelvis_index = self.joints_definition.index('Pelvis')

    def get_tsv_file(self, tsv_file):
        if tsv_file:
            if self.is_composite:
                return CompositeTSVFile(tsv_file, self.linelist_file,
                        root=self.root)
            tsv_path = find_file_path_in_yaml(tsv_file, self.root)
            return TSVFile(tsv_path)

    def get_valid_tsv(self):
        # sorted by file size
        if self.hw_tsv:
            return self.hw_tsv
        if self.label_tsv:
            return self.label_tsv

    def prepare_image_keys(self):
        tsv = self.get_valid_tsv()
        return [tsv.get_key(i) for i in range(tsv.num_rows())]

    def prepare_image_key_to_index(self):
        tsv = self.get_valid_tsv()
        return {tsv.get_key(i) : i for i in range(tsv.num_rows())}


    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling

        back = 1
        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
            if np.random.uniform() <= 0.5:
                back = -1
	    
            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.noise_factor, 1+self.noise_factor, 3)
	    
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.rot_factor,
                    max(-2*self.rot_factor, np.random.randn()*self.rot_factor))
	    
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.scale_factor,
                    max(1-self.scale_factor, np.random.randn()*self.scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0
	
        return flip, pn, rot, sc,back

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale, 
                      [self.img_res, self.img_res], rot=rot)
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
                                  [self.img_res, self.img_res], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/self.img_res - 1.
        # flip the x coordinates
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1]) 
        # flip the x coordinates
        if f:
            S = flip_kp(S)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose = pose.astype('float32')
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def get_line_no(self, idx):
        return idx if self.line_list is None else self.line_list[idx]

    def get_image(self, idx): 
        line_no = self.get_line_no(idx)
        row = self.img_tsv[line_no]
        # use -1 to support old format with multiple columns.
        cv2_im = img_from_base64(row[-1])
        if self.cv2_output:
            return cv2_im.astype(np.float32, copy=True)
        cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)

        return cv2_im

    def get_annotations(self, idx):
        line_no = self.get_line_no(idx)
        if self.label_tsv is not None:
            row = self.label_tsv[line_no]
            annotations = json.loads(row[1])
            return annotations
        else:
            return []

    def get_target_from_annotations(self, annotations, img_size, idx):
        # This function will be overwritten by each dataset to 
        # decode the labels to specific formats for each task. 
        return annotations


    def get_img_info(self, idx):
        if self.hw_tsv is not None:
            line_no = self.get_line_no(idx)
            row = self.hw_tsv[line_no]
            try:
                # json string format with "height" and "width" being the keys
                return json.loads(row[1])[0]
            except ValueError:
                # list of strings representing height and width in order
                hw_str = row[1].split(' ')
                hw_dict = {"height": int(hw_str[0]), "width": int(hw_str[1])}
                return hw_dict

    def get_img_key(self, idx):
        line_no = self.get_line_no(idx)
        # based on the overhead of reading each row.
        if self.hw_tsv:
            return self.hw_tsv[line_no][0]
        elif self.label_tsv:
            return self.label_tsv[line_no][0]
        else:
            return self.img_tsv[line_no][0]

    def __len__(self):
        # return len(self.name_list_in8)//100
        # return 170
        return len(self.name_list_in8)

        # if self.line_list is None:
        #     return self.img_tsv.num_rows() 
        # else:
        #     return len(self.line_list)

    def __getitem__(self, idx):
        #获取时序片段
        this_name_num = self.name_list_in8[idx]
        #加入随机参数back，在训练时，随机正序或者倒叙索引时序片段
        flip,pn,rot,sc,back = self.augm_params()

        itrm_all = {}

        #num_list[::1]代表正序，num_list[::-1]代表倒叙
        for idx in this_name_num[::back]:
            img = self.get_image(idx)

            meta_data = {}
            meta_data['ori_img_1000'] = torch.Tensor(img)

            img_key = self.get_img_key(idx)
            annotations = self.get_annotations(idx)


            annotations = annotations[0]
            center = annotations['center']
            scale = annotations['scale']
            has_2d_joints = annotations['has_2d_joints']
            has_3d_joints = annotations['has_3d_joints']
            joints_2d = np.asarray(annotations['2d_joints'])
            joints_3d = np.asarray(annotations['3d_joints'])

            if joints_2d.ndim==3:
                joints_2d = joints_2d[0]
            if joints_3d.ndim==3:
                joints_3d = joints_3d[0]

            # Get SMPL parameters, if available
            has_smpl = np.asarray(annotations['has_smpl'])
            pose = np.asarray(annotations['pose'])
            betas = np.asarray(annotations['betas'])

            try:
                gender = annotations['gender']
            except KeyError:
                gender = 'none'

            # Get augmentation parameters
            

            # Process image
            img = self.rgb_processing(img, center, sc*scale, rot, flip, pn)
            img = torch.from_numpy(img).float()
            # Store image before normalization to use it in visualization
            transfromed_img = self.normalize_img(img)


            # normalize 3d pose by aligning the pelvis as the root (at origin)
            root_pelvis = joints_3d[self.pelvis_index,:-1]
            joints_3d[:,:-1] = joints_3d[:,:-1] - root_pelvis[None,:]
            # 3d pose augmentation (random flip + rotation, consistent to image and SMPL)
            joints_3d_transformed = self.j3d_processing(joints_3d.copy(), rot, flip)
            # 2d pose augmentation
            joints_2d_transformed = self.j2d_processing(joints_2d.copy(), center, sc*scale, rot, flip)

            ###################################
            # Masking percantage
            # We observe that 30% works better for human body mesh. Further details are reported in the paper.
            mvm_percent = 0.3
            ###################################
            
            mjm_mask = np.ones((14,1))
            if self.is_train:
                num_joints = 14
                pb = np.random.random_sample()
                masked_num = int(pb * mvm_percent * num_joints) # at most x% of the joints could be masked
                indices = np.random.choice(np.arange(num_joints),replace=False,size=masked_num)
                mjm_mask[indices,:] = 0.0
            mjm_mask = torch.from_numpy(mjm_mask).float()

            mvm_mask = np.ones((431,1))
            if self.is_train:
                num_vertices = 431
                pb = np.random.random_sample()
                masked_num = int(pb * mvm_percent * num_vertices) # at most x% of the vertices could be masked
                indices = np.random.choice(np.arange(num_vertices),replace=False,size=masked_num)
                mvm_mask[indices,:] = 0.0
            mvm_mask = torch.from_numpy(mvm_mask).float()

            # meta_data = {}
            meta_data['ori_img'] = img
            meta_data["transfromed_img"] = transfromed_img
            
            meta_data['pose'] = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
            meta_data['betas'] = torch.from_numpy(betas).float()
            meta_data['joints_3d'] = torch.from_numpy(joints_3d_transformed).float()
            meta_data['has_3d_joints'] = has_3d_joints
            meta_data['has_smpl'] = np.array([has_smpl])

            meta_data['mjm_mask'] = mjm_mask
            meta_data['mvm_mask'] = mvm_mask

            # Get 2D keypoints and apply augmentation transforms
            meta_data['has_2d_joints'] = has_2d_joints
            meta_data['joints_2d'] = torch.from_numpy(joints_2d_transformed).float()
            meta_data['scale'] = float(sc * scale)
            meta_data['center'] = np.asarray(center).astype(np.float32)
            meta_data['gender'] = gender
            for key in meta_data.keys():
                # print(key)
                if key not in itrm_all.keys():
                    itrm_all[key] = [meta_data[key]]
                else:
                    itrm_all[key].append(meta_data[key])
        new_itrm = {}

        new_itrm['transfromed_img'] = torch.stack(itrm_all['transfromed_img'] ,dim=0) # input image
        new_itrm['ori_img'] = torch.stack(itrm_all['ori_img'] ,dim=0) # input image
        new_itrm['ori_img_1000'] = torch.stack(itrm_all['ori_img_1000'] ,dim=0) # input image
        new_itrm['pose'] = torch.stack(itrm_all['pose'],dim=0) # SMPL pose parameters
        new_itrm['betas'] = torch.stack(itrm_all['betas'],dim=0) # SMPL beta parameters
        new_itrm['joints_3d'] = torch.stack(itrm_all['joints_3d'],dim=0) # 3D pose
        new_itrm['joints_2d'] = torch.stack(itrm_all['joints_2d'],dim=0) # 3D pose
        new_itrm['mjm_mask'] = torch.stack(itrm_all['mjm_mask'],dim=0) # 3D pose
        new_itrm['mvm_mask'] = torch.stack(itrm_all['mvm_mask'],dim=0) # 3D pose
        new_itrm['center'] = torch.tensor(itrm_all['center']) # 3D pose
        new_itrm['has_smpl'] = torch.tensor(itrm_all['has_smpl']) # flag that indicates whether SMPL parameters are valid
        new_itrm['has_3d_joints'] = torch.tensor(itrm_all['has_3d_joints']) # flag that indicates whether 3D pose is valid
        new_itrm['has_2d_joints'] = torch.tensor(itrm_all['has_2d_joints']) # flag that indicates whether 3D pose is valid
        new_itrm['scale']  = torch.tensor(itrm_all['scale']) # flag that indicates whether image was flipped during data augmentation
        new_itrm['gender'] = itrm_all['gender']

        # return img_key, transfromed_img, meta_data
        return img_key, new_itrm['transfromed_img'], new_itrm



class MeshTSVYamlDataset(MeshTSVDataset):
    """ TSVDataset taking a Yaml file for easy function call
    """
    def __init__(self, yaml_file, is_train=True, cv2_output=False, scale_factor=1):
        self.cfg = load_from_yaml_file(yaml_file)
        self.is_composite = self.cfg.get('composite', False)
        self.root = op.dirname(yaml_file)
        
        if self.is_composite==False:
            img_file = find_file_path_in_yaml(self.cfg['img'], self.root)
            label_file = find_file_path_in_yaml(self.cfg.get('label', None),
                                                self.root)
            hw_file = find_file_path_in_yaml(self.cfg.get('hw', None), self.root)
            linelist_file = find_file_path_in_yaml(self.cfg.get('linelist', None),
                                                self.root)
        else:
            img_file = self.cfg['img']
            hw_file = self.cfg['hw']
            label_file = self.cfg.get('label', None)
            linelist_file = find_file_path_in_yaml(self.cfg.get('linelist', None),
                                                self.root)

        super(MeshTSVYamlDataset, self).__init__(
            img_file, label_file, hw_file, linelist_file, is_train, cv2_output=cv2_output, scale_factor=scale_factor)
