# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.config import configurable
# from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
# from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
import logging
from typing import Dict, Tuple, List, Optional
from collections import OrderedDict
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone, Backbone
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.utils.events import get_event_storage
from detectron2.structures import ImageList
import pdb


############### Image discriminator ##############
class FCDiscriminator_img(nn.Module):
    def __init__(self, num_classes, ndf1=256, ndf2=128):
        super(FCDiscriminator_img, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ndf1, ndf2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ndf2, ndf2, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(ndf2, 1, kernel_size=3, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x
#################################



############### Image discriminator ##############
class FCDiscriminator_img(nn.Module):
    def __init__(self, num_classes, ndf1=256, ndf2=128):
        super(FCDiscriminator_img, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ndf1, ndf2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ndf2, ndf2, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(ndf2, 1, kernel_size=3, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x
#################################

################ Gradient reverse function
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

#######################

@META_ARCH_REGISTRY.register()
class DAobjTwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        dis_type: str,
        # dis_loss_weight: float = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super(GeneralizedRCNN, self).__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        # @yujheli: you may need to build your discriminator here

        self.dis_type = dis_type
        self.D_img = None
        # self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels['res4']) # Need to know the channel
        
        # self.D_img = None
        self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]) # Need to know the channel
        # self.bceLoss_func = nn.BCEWithLogitsLoss()
    def build_discriminator(self):
        self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]).to(self.device) # Need to know the channel

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "dis_type": cfg.SEMISUPNET.DIS_TYPE,
            # "dis_loss_ratio": cfg.xxx,
        }

    def preprocess_image_train(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        images_t = [x["image_unlabeled"].to(self.device) for x in batched_inputs]
        images_t = [(x - self.pixel_mean) / self.pixel_std for x in images_t]
        images_t = ImageList.from_tensors(images_t, self.backbone.size_divisibility)

        return images, images_t

    def forward(
        self, batched_inputs, branch="supervised", memory=None, given_proposals=None, val_mode=False
    ):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if self.D_img == None:
            self.build_discriminator()
        if (not self.training) and (not val_mode):  # only conduct when testing mode
            return self.inference(batched_inputs)
                 
               
        self.conv = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1).to(self.device)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1,1)).to(self.device)
        
        source_label = 0
        target_label = 1

        if branch == "domain":
            # self.D_img.train()
            # source_label = 0
            # target_label = 1
            # images = self.preprocess_image(batched_inputs)
            images_s, images_t = self.preprocess_image_train(batched_inputs)

            features = self.backbone(images_s.tensor)

            # import pdb
            # pdb.set_trace()
           
            features_s = grad_reverse(features[self.dis_type])
            D_img_out_s = self.D_img(features_s)
            loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))

            features_t = self.backbone(images_t.tensor)
            
            features_t = grad_reverse(features_t[self.dis_type])
            # features_t = grad_reverse(features_t['p2'])
            D_img_out_t = self.D_img(features_t)
            loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))

            # import pdb
            # pdb.set_trace()

            losses = {}
            losses["loss_D_img_s"] = loss_D_img_s
            losses["loss_D_img_t"] = loss_D_img_t
            return losses, [], [], None
        
        
        if branch == "domain_mem":
            # self.D_img.train()
            # source_label = 0
            # target_label = 1
            # images = self.preprocess_image(batched_inputs)
            images_s, images_t = self.preprocess_image_train(batched_inputs)
            features = self.backbone(images_s.tensor)
            
            
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None
            
            
            # pool source features for gt boxes 
            proposal_boxes = [x.gt_boxes for x in gt_instances] # each images of batch have multiple boxes -collect them all
            # VERIFY
            features1 = [features[f] for f in self.roi_heads.in_features] # torch.Size([4, 1024, 38, 57])
            proposal_rois = self.roi_heads.box_pooler(features1, proposal_boxes) # torch.Size([17, 1024, 7, 7])
            # A tensor of shape (M, C, output_size, output_size) where M is the total number of boxes aggregated over all N batch images
            box_features = self.roi_heads.box_head(proposal_rois) # flatten ---> torch.Size([17, 1024])
            

            loss_s = []
            for i in range (len(proposal_rois)):
                ad = proposal_rois[i]
                ads  = torch.unsqueeze(ad, 0)
                features_sa = grad_reverse(ads).to(self.device)
                D_img_out_s = self.D_img(features_sa)
                
                loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))
                loss_s.append(loss_D_img_s)
            
                
            
            
            
            features_t = self.backbone(images_t.tensor)
            features1_t = [features_t[f] for f in self.roi_heads.in_features]
            features1_t = features1_t[0][0]
            
            """
            box_features = F.normalize(box_features, p=2, dim=1, eps=1e-12) 
            #extract similar features from memory 
            #reterive the most similar pair of memory
            x = features11_t
            x = self.conv(x)
            x1 = self.pool(x)
            x = x1.view(-1, 1024)
            
            
            loss_t=[]
            highest_cos_20 = []
            highest_cos_20_score = []
            # extract 20 targets 
            for i in range (len(x)):
                data1 = x[i:i+1]
                data1 = F.normalize(data1, p=2, dim=1, eps=1e-12) 
                ab_=[]
                cos1 = []
               
                for j in range (len(memory)):
                    meu = memory[j]
                    meu = F.normalize(meu, p=2, dim=1, eps=1e-12)
                    cos_sim = (meu * data1).sum(dim=-1)
                    am = torch.unsqueeze(cos_sim, 1)
                    
                    amm = meu*am                    
                    an = torch.sum(amm, 0); an = torch.unsqueeze(an, dim=0)
                    
                    #ant = torch.unsqueeze(an, dim=0)
                    ann = F.normalize(an, p=2, dim=1, eps=1e-12)
                    pdb.set_trace()
                    cos_sim2 = (box_features * abs(ann)).sum(dim=-1)
                    cos1.append(max(cos_sim2).tolist())
                    
                    ab_.append(an[0]) #memory update 
                   
                
                abm = torch.stack(ab_)
                abm = F.normalize(abm, p=2, dim=1, eps=1e-12)
                #pdb.set_trace()
                #cos_sim1 = (data1 * abm).sum(dim=-1)
                #knn = cos_sim1.topk(1, largest=True)
                #topk_pos = abm[knn.indices] # top K positive selected - 1
                val1 = cos1[-1]
                #highest_cos_20.append(topk_pos[0])
                #highest_cos_20_score.append(knn.values[0])
                highest_cos_20_score.append(val1)
               
                
                ams  = torch.unsqueeze(abm.clone(), -1); ams  = torch.unsqueeze(ams.clone(), -1)
                features_t = grad_reverse(ams.clone()).to(self.device)
                D_img_out_t = self.D_img(features_t.clone())
                loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))
                loss_t.append(loss_D_img_t)
                
            """
            #highest_cos_19 = torch.stack(highest_cos_20) 
            #highest_cos_20_score = torch.stack(highest_cos_20_score)
            #pdb.set_trace()
            
            
            sim20 =[]; loss_t=[]
            feat_s = features1[0][0]
            #similary of 38x59 with all memory and then sum it with weights  
            for i in range (1, len(memory)): # amap for each class
                meu = memory[i][0:10]
                meu = F.normalize(meu, p=2, dim=1, eps=1e-12)
                meu = meu.clone()
                reterive_at = torch.zeros(features1_t.shape[-3], features1_t.shape[-2], features1_t.shape[-1])
                reterive_at = features1_t.clone() # cosine similarity map with original feature map 
                
                reterive_at_s = torch.zeros(feat_s.shape[-3], feat_s.shape[-2], feat_s.shape[-1])
                reterive_at_s = feat_s.clone() 
                
                
                cos_sim_12 = []
                
                for j in range (10): # features1_t.shape[-2]
                    for k in range (10): # features1_t.shape[-1]
                        data1 = F.normalize(features1_t[:, j, k].clone(), p=2, dim=0, eps=1e-12)
                        cos_sim = (data1 * meu).sum(dim=-1)
                        
                        cos_sim  = torch.unsqueeze(cos_sim, 1)
                        ff0 = cos_sim.repeat(1, 1024)
                        ff = meu*ff0  # extracted value 
                        #ff = [cos_sim[k]*meu[k] for k in range (len(cos_sim))]
                        ffa = F.normalize(sum(ff), p=2, dim=0, eps=1e-12) 
                        reterive_at[:, j, k] =  features1_t[:, j, k].clone()*(ffa.clone()*data1.clone()).sum(dim=-1)
                        
                        
                        data2 = F.normalize(feat_s[:, j, k].clone(), p=2, dim=0, eps=1e-12)
                        cos_sim = (data2 * meu).sum(dim=-1)
                        
                        cos_sim  = torch.unsqueeze(cos_sim, 1)
                        ff0 = cos_sim.repeat(1, 1024)
                        ff = meu*ff0  # extracted value 
                        #ff = [cos_sim[k]*meu[k] for k in range (len(cos_sim))]
                        ffa = F.normalize(sum(ff), p=2, dim=0, eps=1e-12) 
                        reterive_at_s[:, j, k] =  feat_s[:, j, k].clone()*(ffa.clone()*data2.clone()).sum(dim=-1)
                        
                        
                        aa1 = F.normalize(reterive_at_s[:, j, k], p=2, dim=0, eps=1e-12)
                        aa2 = F.normalize(reterive_at[:, j, k], p=2, dim=0, eps=1e-12)
                        cos_sim = (aa1 * aa2).sum(dim=-1)
                        
                        cos_sim_12.append(cos_sim.tolist())
                        
                           
                #sim20.append(sum(cos_sim_12)/len(cos_sim_12))
                sim20.append(min(cos_sim_12))
                ams  = torch.unsqueeze(reterive_at.clone(), 0)
                features_t = grad_reverse(ams.clone()).to(self.device)
                D_img_out_t = self.D_img(features_t.clone())
                loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))
                loss_t.append(loss_D_img_t)
            
                
            
                

            
            #pdb.set_trace()
            #torch.mean(torch.stack(loss_t))
            #torch.mean(torch.stack(loss_s))

            #pdb.set_trace()
            
            
            losses = {}
            losses["loss_D_img_s1"] = loss_D_img_s
            losses["loss_D_img_t1"] =  loss_D_img_t
            return losses, [], [], None, max(sim20)#highest_cos_20_score
        
        

        # self.D_img.eval()
        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        # TODO: remove the usage of if else here. This needs to be re-organized
        if branch.startswith("supervised"):
            features_s = grad_reverse(features[self.dis_type])
            D_img_out_s = self.D_img(features_s)
            loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))

            
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses["loss_D_img_s"] = loss_D_img_s*0.001
            return losses, [], [], None

        elif branch.startswith("supervised_target"):

            # features_t = grad_reverse(features_t[self.dis_type])
            # D_img_out_t = self.D_img(features_t)
            # loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))

            
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            # losses["loss_D_img_t"] = loss_D_img_t*0.001
            # losses["loss_D_img_s"] = loss_D_img_s*0.001
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            """
            unsupervised weak branch: input image without any ground-truth label; output proposals of rpn and roi-head
            """
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)
            # notice that we do not use any target in ROI head to do inference!
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            # if self.vis_period > 0:
            #     storage = get_event_storage()
            #     if storage.iter % self.vis_period == 0:
            #         self.visualize_training(batched_inputs, proposals_rpn, branch)

            return {}, proposals_rpn, proposals_roih, ROI_predictions
        elif branch == "unsup_data_strong":
            raise NotImplementedError()
        elif branch == "val_loss":
            raise NotImplementedError()
            
        
        
        # New 
        #**************************** MEMORY UPDATE *****************************
        elif branch == "unsup_data_memory":
            """
            memory network branch: input image features and ground-truth label; output roi features of GT and label 
            """
            # pool features for gt boxes 
            proposal_boxes = [x.gt_boxes for x in gt_instances] # each images of batch have multiple boxes -collect them all
            # VERIFY
            features = [features[f] for f in self.roi_heads.in_features] # torch.Size([4, 1024, 38, 57])
            proposal_rois = self.roi_heads.box_pooler(features, proposal_boxes) # torch.Size([17, 1024, 7, 7])
            # A tensor of shape (M, C, output_size, output_size) where M is the total number of boxes aggregated over all N batch images
            box_features = self.roi_heads.box_head(proposal_rois) # flatten ---> torch.Size([17, 1024])
            
            
            box_classes = []
            for x in gt_instances:
                box_classes = box_classes + x.gt_classes.tolist()
            
            if (box_features.shape[0] != len(box_classes)):
                print("***************ALERT!!! ALERT!!!**************")
               
            return {}, box_features, box_classes # for each image image there must be a gt_box
        
        
        ### write memory -->
        
        elif branch == "write_memory":
           
            # pool features for gt boxes 
            proposal_boxes = [x.gt_boxes for x in gt_instances] # each images of batch have multiple boxes -collect them all
            # VERIFY
            features = [features[f] for f in self.roi_heads.in_features] # torch.Size([4, 1024, 38, 57])
            proposal_rois = self.roi_heads.box_pooler(features, proposal_boxes) # torch.Size([17, 1024, 7, 7]) --> pooled features
            # A tensor of shape (M, C, output_size, output_size) where M is the total number of boxes aggregated over all N batch images
            box_features = self.roi_heads.box_head(proposal_rois) # flatten ---> torch.Size([17, 1024])
            
            
            box_classes = []
            for x in gt_instances:
                box_classes = box_classes + x.gt_classes.tolist()
                
                
            # use each of 17 to update all memory elements [100 x 1024]
            Kma = []
            for i in range (len(box_classes)):
                b_class = box_classes[i]
                mem = memory[b_class]
                data = box_features[i:i+1]
                mem = F.normalize(mem, p=2, dim=1, eps=1e-12)
                data = F.normalize(data, p=2, dim=1, eps=1e-12)
                
                cos_sim = (mem * data).sum(dim=-1)
                data_rep = data.repeat(100, 1); am = torch.unsqueeze(cos_sim, 1); am = am.repeat(1, 1024)
                memory[b_class] = mem + data*am # memory update 
                
                Km = torch.norm(mem - data_rep, dim=1, p=2)
                Kma.append(torch.mean(Km))
            
            if len(Kma)!=0:
                la = torch.mean(torch.stack(Kma))  # mean over batch 
            else: # if there is no box out of all batches    
                la = 0.0
            losses = {}
            losses["loss_mem"] = la
            
            
            
            """  
            Lcm = []
            aa = torch.zeros(len(proposal_rois), 10, 1024)
            bb = torch.zeros(len(proposal_rois), 10, 1024)
            for i in range (len(proposal_rois)):# for each box
                b_class = box_classes[i]
                if b_class==0:
                    continue 
                mem = memory[b_class]
                prop = torch.reshape(proposal_rois[i], [proposal_rois.shape[-1]*proposal_rois.shape[-2], proposal_rois.shape[1]])
                prop = F.normalize(prop, p=2, dim=1, eps=1e-12)
                prop = prop.clone()
                
                Lc = []
                
                for m in range (10): # for each memory
                    # compute cosine sim
                    data1 = F.normalize(mem[m].clone(), p=0, dim=0, eps=1e-12)
                    cos_sim = (data1.clone() * prop.clone()).sum(dim=-1) # 49 lenght 
                    ff = [cos_sim[k]*prop[k] for k in range (len(cos_sim))]
                    
                
                    mega = data1 + sum(ff)
                    memory[b_class][m] = mega
                    
                    aa[i, m, :]=data1
                    bb[i, m, :]=data1 #sum(ff)
                    
                    #Lc.append(torch.norm(data1 - sum(ff), dim=0, p=2))
                
                
                #Lcm.append(torch.mean(torch.stack(Lc)))    
            #pdb.set_trace()
            
            Km = torch.norm(aa - bb, dim=2, p=2)[0]
            
                    
            if len(Km)!=0:
                #la = torch.mean(torch.stack(Km))  # mean over batch 
                la = torch.mean(Km) 
            else: # if there is no box out of all batches    
                la = 0.0
            
            losses = {}
            losses["loss_mem"] = la
            """
           
                   
            return losses, memory # for each image image there must be a gt_box
        
        

        

    def visualize_training(self, batched_inputs, proposals, branch=""):
        """
        This function different from the original one:
        - it adds "branch" to the `vis_name`.
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = (
                "Left: GT bounding boxes "
                + branch
                + ";  Right: Predicted proposals "
                + branch
            )
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch



@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None