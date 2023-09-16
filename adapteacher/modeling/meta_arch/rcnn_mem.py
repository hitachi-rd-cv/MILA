import pdb
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
from detectron2.modeling.poolers import ROIPooler
from torch.autograd import Variable,Function
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
import torchvision.transforms.functional as TF

# MILA
import torchvision
import itertools
import cv2
import glob


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
        #pooler: ROIPooler,
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
        #self.pooler = pooler

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
            #"in_features": cfg.MODEL.ROI_HEADS.IN_FEATURES,
            #"pooler_resolution": cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION,
            #"pooler_type": cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE,
            #"pooler_scales":(1.0 / input_shape[in_features[0]].stride, ),
            #"sampling_ratio": cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO,
            #"mask_on": cfg.MODEL.MASK_ON,

            #ret["pooler"] = ROIPooler(output_size="pooler_resolution",
            #scales="pooler_scales", sampling_ratio="sampling_ratio", pooler_type="pooler_type",
            #)


        }




    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            
            
            valid_map = proposal_bbox_inst.scores > thres
            indices = valid_map.nonzero()[:,0]

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.pred_boxes = new_boxes
            new_proposal_inst.pred_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        return new_proposal_inst, indices


    def process_pseudo_label(
        self, proposals_rpn_unsup_k, pred_inds, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        list_indices = []
        

        num_proposal_output = 0.0
        #for proposal_bbox_inst in proposals_rpn_unsup_k:
        for i in range (len(proposals_rpn_unsup_k)): #batch
            pred_inds_inst = pred_inds[i]
            proposal_bbox_inst = proposals_rpn_unsup_k[i]
            
            # dynamic 
            pred_classes = proposal_bbox_inst.pred_classes
            new_th = cur_threshold[pred_classes]
            
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst, indices = self.threshold_bbox(
                    proposal_bbox_inst, thres=new_th, proposal_type=proposal_type
                )
                pred_inds_inst_select =pred_inds_inst[indices] # choose only those indices which have thresold greather
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
            list_indices.append(pred_inds_inst_select)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output, list_indices
   
    def process_pseudo_label_orig(
        self, proposals_rpn_unsup_k, pred_inds, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        list_indices = []

        num_proposal_output = 0.0
        #for proposal_bbox_inst in proposals_rpn_unsup_k:
        for i in range (len(proposals_rpn_unsup_k)):
            pred_inds_inst = pred_inds[i]
            proposal_bbox_inst = proposals_rpn_unsup_k[i]
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst, indices = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
                pred_inds_inst_select =pred_inds_inst[indices] # choose only those indices which have thresold greather
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
            list_indices.append(pred_inds_inst_select)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output, list_indices



    def feat_allign_with_memory(self, tgt_class, tgt_feat, memory_store, no_class, K):

        input1_1 = []# 4x100x1024
        input2_2 = []

        for i in range (len(tgt_feat)): # batches
            tgt_class_u = tgt_class[i] # target classes
            tgt_feat_u =tgt_feat[i] # target features

            input1_11 = [] #100x1024
            input2_22 = []


            for j in range (len(tgt_feat_u)): # 100 times but some times zeroth class

                class_id = tgt_class_u[j].item()

                # ignore if the predicted target class is 0 for VOC dataset only
                if no_class==20:
                    if class_id == 0:
                        continue

                input1 = memory_store[class_id] # all car classes
                input2 = tgt_feat_u[j:j+1] # for each tgt features all memory from same class

                #find best match pair - nearest slot in memory of same class (with cosine similairy)
                #sim = F.cosine_similarity(input1, input2) #or dot product
                ##simL = F.softmax(sim)
                #knn = sim.topk(K, largest=True)
                #topk_mean = torch.mean(data[knn.indices], 0)

                data = input1
                test = input2

                dist = torch.norm(data - test, dim=1, p=None)
                knn = dist.topk(K, largest=False)
                #print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))
                topk_mean = torch.mean(data[knn.indices], 0) # topk nearest features with test data

                #input1_11.append(input1[index])
                input1_11.append(topk_mean)
                input2_22.append(input2[0])

            #print("**************", len(input1_11))
            if len(input1_11)!=0: # len(input1_11) <= len(tgt_class_u)
                input1_1.append(torch.stack(input1_11))  # 4(3,2,1,0) times 100(99,...1) x 1024 Source proposals
                input2_2.append(torch.stack(input2_22))  # 4 times 100 x 1024 target proposals

        #print(input2_2)
        return input1_1, input2_2 # len(input1_1) <= batch size  ----> it might be []


    # MILA return multiple positive and multiple negative 
    def feat_allign_with_memory_trip(self, tgt_class, tgt_feat, memory_store, no_class, K):
        
        # siamese type setting 
        input1_1 = [] # anchor
        input2_2 = [] # pos
        input3_3 = [] # neg
        #input4_4 = [] # pos similairty score
        input4_4 = [] # sim crop
        
        voc_class = torch.arange(0, 20).tolist()
        city_class = torch.arange(0, 8).tolist()
       
        for i in range (len(tgt_feat)): # batches 
            # all boxes within batch and its labels 
            tgt_class_u = tgt_class[i] # target classes [100]
            tgt_feat_u =tgt_feat[i] # target features [100x1024]
            
            input1_11 = [] # store anchor
            input2_22 = [] # store pos
            input3_33 = [] # store neg
            #input4_44 = [] # pos similairty score
            #input4_44 = [] # store matching crop 
            
            
            for j in range (len(tgt_feat_u)): # 100 times but some times zeroth class
                
                # pick one target box feaure and label 
                class_id = tgt_class_u[j].item()
                
                # ignore if the predicted target class is 0 for VOC dataset only
                #if no_class==20:
                #    if class_id == 0:
                #        continue 
                
                # K positive selection from memory  
                data = memory_store[class_id] # e.g, all cars
                test = tgt_feat_u[j:j+1] # for tgt features 
       
                #obj_crops = crops[class_id]
                
                #  distance and cosine sim giving same results 
                #dist = torch.norm(data - test, dim=1, p=None)
                #knn = dist.topk(K, largest=False)
                #topk_pos = data[knn.indices] # top K positive selected 
                
                # compute cosine sim
                data1 = F.normalize(data, p=2, dim=1, eps=1e-12)
                test1 = F.normalize(test, p=2, dim=1, eps=1e-12)
                #cos_sim = torch.mm(test, data.transpose(0, 1))
                cos_sim = (data1 * test1).sum(dim=-1)
                knn = cos_sim.topk(K, largest=True)
                topk_pos = data1[knn.indices] # top K positive selected - 1
                
              
                #sim_crops = obj_crops[knn.indices]
                #input4_44.append(sim_crops)
                
                """
                # N negative selection 
                if no_class==20:
                    class_ids1 = set(voc_class)-set([class_id])
                if no_class==8:
                    class_ids1 = set(city_class)-set([class_id])
                
                ### select s negative samples from rest of the classes n = s x len(class_id), e.g. s = 1
                neg = []
                for k in class_ids1:
                    mem_l = memory_store[k]
                    ind = torch.randint(0, mem_l.shape[0], (1,)) # select a random number in a range 
                    input1_neg = mem_l[ind] # 1 negative  
                    input1_neg = F.normalize(input1_neg, p=2, dim=1, eps=1e-12)
                    neg.append(input1_neg) # store it 
                """
                topk_neg = data1[knn.indices] # chnages for sim10k
                
                
                # anchor, positive, negative samples --- triplet loss all are normlaized 
                
                # combine K (1) positive and N (18) negative samples 
                pos_= topk_pos[0] # [1, 1024]
                #ind1 = torch.randint(0, len(neg), (1,)) # select one neg out of 18 different classes 
                #neg_ = neg[ind1][0] # [1, 1024] # changes for sim10k
                neg_ = topk_neg[0] 
                anchor = test1[0] # [1, 1024] --> [1024]

                # store anchor, pos+neg, gt 
                input1_11.append(anchor) # 100 x  19x1024
                input2_22.append(pos_) # anchor 100 x  19x1024 
                input3_33.append(neg_) #100  x 19
                #input4_44.append(knn.values)
            
            
           
           
            #input4_4.append(input4_44)
            if len(input1_11)!=0: # len(input1_11) <= len(tgt_class_u)
                input1_1.append(torch.stack(input1_11))  # 100 x 19x1024   100 x1024
                input2_2.append(torch.stack(input2_22))  # 100 x 19x1024  100 x1024
                input3_3.append(torch.stack(input3_33))  # 100  x 19 # 100 x1024
                #input4_4.append(torch.stack(input4_44)) # 100 x 1
                
                
        return input1_1, input2_2, input3_3  #, input4_4 # len(input1_1) <= batch size  ----> it might be [], input4_4 is exception 
    
    
    

    def ContrastiveLoss(self, output1, output2, targets):
        self.margin = 1.0
        d = (output1 - output2).pow(2).sum(-1).sqrt() # distance
        loss = torch.mean(0.5 * targets.float() * d.pow(2) + \
                          0.5 * (1 - targets.float()) * F.relu(self.margin - d).pow(2))
        return loss # we also return distance; it is needed to evaluate the model
    
    def TripletLoss(self, anchor, positive, negative): # , sim_score
        self.margin = 1.0
        distance_positive = ((anchor - positive).pow(2).sum(1)) #* sim_score[:,0] # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() 



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

    # New (memory and Pencoder)
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False, memory=None, no_class=None, topk=None, cur_threshold=None
    ):  #Pencoder=None,
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

            losses = {}
            losses["loss_D_img_s"] = loss_D_img_s
            losses["loss_D_img_t"] = loss_D_img_t
            return losses, [], [], None


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
            #pdb.set_trace()

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
            #print("********************pred_instances1*************",proposals_roih[0])

            return {}, proposals_rpn, proposals_roih, ROI_predictions
        elif branch == "unsup_data_strong":
            raise NotImplementedError()
        elif branch == "val_loss":
            raise NotImplementedError()

        # MILA 
        #**************************** MEMORY UPDATE *****************************
        elif branch == "unsup_data_memory":
            """
            memory network branch: input image features and ground-truth label; output roi features of GT and label
            """
            # pool features for gt boxes
            proposal_boxes = [x.gt_boxes for x in gt_instances] # each have multiple boxes collect all
            # VERIFY
            features = [features[f] for f in self.roi_heads.in_features]
            proposal_rois = self.roi_heads.box_pooler(features, proposal_boxes) # torch.Size([17, 1024, 7, 7])
            # A tensor of shape (M, C, output_size, output_size) where M is the total number of boxes aggregated over all N batch images
            box_features = self.roi_heads.box_head(proposal_rois) # flatten ---> torch.Size([17, 1024])
            

            #Filtering OUT source Bboxs to be stored based on the prediction results! 
            predictions = self.roi_heads.box_predictor(box_features) # class and box predictions [8000, 21] & [8000, 20*4]
            probs = F.softmax(predictions[0], dim=-1)
            #probs = predictions[0].sigmoid()
            _, pred_classes = torch.max(probs, 1)
            
            gt_classes = [x.gt_classes.cpu().tolist() for x in gt_instances]
            gt_classes = list(itertools.chain.from_iterable(gt_classes))
            
            
            # save cropped source instances so that they can be used to save to visualize correspond to the target instances
            crop_list = []
            for m in range (len(proposal_boxes)): # batch 
                for n in range (proposal_boxes[m].tensor.shape[0]): # boxes 
                    bounding_box = proposal_boxes[m][n].tensor
                    x, y, w, h = bounding_box[0].tolist()
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    image_tensor = batched_inputs[m]['image']
                    cropped_image = TF.crop(image_tensor, y, x, h-y, w-x)
                    image_np = cropped_image.permute(1, 2, 0).cpu().numpy()
                    crop_list.append(image_np)
                
            

            box_features_f = []
            box_classes_f = []
            box_crop_f = []
            for i in range (len(gt_classes)):
                if gt_classes[i]==pred_classes[i].item():
                   box_features_f.append(box_features[i])
                   box_classes_f.append(gt_classes[i])
                   box_crop_f.append(crop_list[i])
            
            if len(box_classes_f)!=0:
                box_features_f = torch.stack(box_features_f)
           
            
            return {}, box_features_f, box_classes_f, box_crop_f # for each image image there must be a gt_box

        # MILA 
        elif branch == "contra_loss": # for target only
            """
            contra loss branch: input target data and memory; output loss
            """

            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            ) # 4 Instance object each of 2000 proposals

            proposal_boxes = [x.proposal_boxes for x in proposals_rpn] #target proposal boxes -->4 box objects with 2000 proposals in each
            features1 = [features[f] for f in self.roi_heads.in_features] # torch.Size([4, 1024, 40, 38])
            proposal_rois = self.roi_heads.box_pooler(features1, proposal_boxes) # torch.Size([8000, 1024, 7, 7])
            box_features = self.roi_heads.box_head(proposal_rois) # flatten --  -> torch.Size([8000, 1024])
            predictions = self.roi_heads.box_predictor(box_features) # class and box predictions [8000, 21] & [8000, 20*4]

            # apply prediction deltas to proposal boxes
            # filter the boxes by scores ---> choose score carefully
            # non-maximum suppression
            # choose top-k results
            pred_instances, pred_inds = self.roi_heads.box_predictor.inference(predictions, proposals_rpn)



            #basic dynamic thresholding 
            #filter boxes again same as teacher does to generate pseudo labels
            #cur_threshold = 0.03
            th_voc = (torch.tensor([91.42, 90.32, 92.50, 82.12, 78.49, 95.08, 91.05, 95.69, 78.59, 87.30, 78.83, 94.30, 95.15, 93.96, 91.88, 73.97, 90.52, 82.21, 94.55, 85.58])/100).to(self.device) 
            th_city = (torch.tensor([42.05, 50.96, 64.92, 34.62, 54.22, 41.26, 34.95, 47.18])/100).to(self.device)
            
            if no_class==20:
                th = th_voc
            else:
                th = th_city
            
            # rescale values 
            v_min, v_max = th.min(), th.max()
            new_min, new_max = 0.6, 0.8
            th_p = (th - v_min)/(v_max - v_min)*(new_max - new_min) + new_min


            pred_instances_p, _, pred_inds_p = self.process_pseudo_label(
                pred_instances, pred_inds, th, "roih", "thresholding"
            )
            
            #pred_instances_p, _, pred_inds_p = self.process_pseudo_label_orig(
            #    pred_instances, pred_inds, cur_threshold, "roih", "thresholding"
            #)
            
            
            # pool features for gt boxes
            #proposal_boxes = [x.gt_boxes for x in gt_instances] # each have multiple boxes collect all
            
            """
            #crop boxes over target for filtered bounding boxes 
            crop_list = []
            for m in range (len(pred_instances_p)): # batch 
                crop_list1 = []
                for n in range (pred_instances_p[m].pred_boxes.tensor.shape[0]): # boxes 
                    bounding_box = pred_instances_p[m][n].pred_boxes.tensor
                    
                    x, y, w, h = bounding_box[0].tolist()
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    image_tensor = batched_inputs[m]['image']
                    cropped_image = TF.crop(image_tensor, y, x, h-y, w-x)
                    image_np = cropped_image.permute(1, 2, 0).cpu().numpy()
                    crop_list1.append(image_np)
                crop_list.append(crop_list1)
            """
            

            # for each crop find feaure to be alligned with memory
            # save target crop and crop correposnd to reteried feature
            # use gt of target and check reterived instances 
            
            #[8000, 1024] ---> [4, 2000, 1024] # pred_inds_p is in range 0 - 2000
            list1 = torch.ones(len(proposal_boxes))*2000 # batch size
            for n in range (len(list1)):
                list1[n] = proposals_rpn[n].objectness_logits.shape[0]  # rpn boxes in each batch               
            list1 = list1.type(torch.int64)
            list1 = list1.tolist()            
            box_features = torch.split(box_features, list1) #split features in lists of specified size --- OK
            
           
            # features of the proposed target boxes ---- OK
            tgt_feat = [box_features[i][pred_inds_p[i]] for i in range (len(pred_inds_p))] # torch.Size([100, 1024]) for N batches
            tgt_class = [f.pred_classes for f in pred_instances_p] # torch.Size([100,]) for N batches
            
            
            # find alligned features to target proposals from source memory
            #input1_1, input2_2 = self.feat_allign_with_memory(tgt_class, tgt_feat, memory, no_class, topk)
            input1_1, input2_2, input3_3 = self.feat_allign_with_memory_trip(tgt_class, tgt_feat, memory, no_class, topk) # anchor input4_4
             
            # loss
            ContraL = []# --- OK
            for i in range (len(input1_1)): # for each batch
                if len(input1_1[i])!=0:
                    ContraL.append(self.TripletLoss(input1_1[i], input2_2[i], input3_3[i])) # input4_4[i]

            if len(ContraL)!=0:
                ContraL = torch.mean(torch.stack(ContraL))  # mean over batch
            else: # if there is no box out of all batches
                ContraL = 0.0

            losses = {}
            losses["loss_D_img_contra"] = ContraL

            return losses, [], [], None
        #****************************************************************************


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
