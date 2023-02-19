# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn.functional as F
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
from ..ssod_utils import QFLv2
from ..losses import GIoULoss

__all__ = ['PicoDet']


@register
class PicoDet(BaseArch):
    """
    Generalized Focal Loss network, see https://arxiv.org/abs/2006.04388

    Args:
        backbone (object): backbone instance
        neck (object): 'FPN' instance
        head (object): 'PicoHead' instance
    """

    __category__ = 'architecture'

    def __init__(self, backbone, neck, head='PicoHead'):
        super(PicoDet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.export_post_process = True
        self.export_nms = True
        self.is_teacher = False

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])

        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        kwargs = {'input_shape': neck.out_shape}
        head = create(cfg['head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "head": head,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        fpn_feats = self.neck(body_feats)

        self.is_teacher = self.inputs.get('is_teacher', False)
        if self.training or self.is_teacher:
            loss = self.head(fpn_feats, targets=self.inputs)
            return loss
        else:
            if not self.export_post_process:  # default False
                head_outs = self.head(fpn_feats, export_post_process=True)
                return {'picodet': head_outs}
            else:
                head_outs = self.head(fpn_feats)
                bboxes, bbox_num = self.head.post_process(
                    head_outs,
                    self.inputs['scale_factor'],
                    export_nms=self.export_nms)
                if self.export_nms:  # default True
                    return {'bbox': bboxes, 'bbox_num': bbox_num}
                else:
                    return {'bbox': bboxes, 'scores': bbox_num}

    def get_loss(self, ):
        return self._forward()

    def get_pred(self):
        return self._forward()

    def get_loss_keys(self):
        return ['loss_vfl', 'loss_bbox', 'loss_dfl']

    def get_distill_loss(self, head_outs, teacher_head_outs, ratio=0.01):
        # student_probs: already sigmoid
        student_probs, student_deltas, student_dfl = head_outs
        teacher_probs, teacher_deltas, teacher_dfl = teacher_head_outs
        nc = student_probs.shape[-1]
        student_probs = student_probs.reshape([-1, nc])
        teacher_probs = teacher_probs.reshape([-1, nc])
        student_deltas = student_deltas.reshape([-1, 4])
        teacher_deltas = teacher_deltas.reshape([-1, 4])
        student_dfl = student_dfl.reshape([-1, 4, 8])
        teacher_dfl = teacher_dfl.reshape([-1, 4, 8])

        with paddle.no_grad():
            # Region Selection
            count_num = int(teacher_probs.shape[0] * ratio)
            #teacher_probs = F.sigmoid(teacher_probs) # already sigmoid
            max_vals = paddle.max(teacher_probs, 1)
            sorted_vals, sorted_inds = paddle.topk(max_vals,
                                                   teacher_probs.shape[0])
            mask = paddle.zeros_like(max_vals)
            mask[sorted_inds[:count_num]] = 1.
            fg_num = sorted_vals[:count_num].sum()
            b_mask = mask > 0.

        loss_logits = QFLv2(
            student_probs, teacher_probs, weight=mask, reduction="sum") / fg_num

        inputs = paddle.concat(
            (-student_deltas[b_mask][..., :2], student_deltas[b_mask][..., 2:]),
            axis=-1)
        targets = paddle.concat(
            (-teacher_deltas[b_mask][..., :2], teacher_deltas[b_mask][..., 2:]),
            axis=-1)
        iou_loss = GIoULoss(reduction='mean')
        loss_deltas = iou_loss(inputs, targets)

        #loss_dfl = paddle.to_tensor([0]) # todo
        student_dfl_pred = student_dfl[b_mask].reshape([-1, 8])
        teacher_dfl_tar = teacher_dfl[b_mask].reshape([-1, 8])

        loss_dfl = self.distribution_focal_loss(student_dfl_pred,
                                                teacher_dfl_tar)

        return {
            "distill_loss_vfl": loss_logits,
            "distill_loss_bbox": loss_deltas,
            "distill_loss_dfl": loss_dfl,
            "fg_sum": fg_num,
        }

    def _df_loss(self, pred_dist, target):
        target_left = paddle.cast(target, 'int64')
        target_right = target_left + 1
        weight_left = target_right.astype('float32') - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist, target_left, reduction='none') * weight_left
        loss_right = F.cross_entropy(
            pred_dist, target_right, reduction='none') * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def distribution_focal_loss(self,
                                pred_corners,
                                target_corners,
                                weight_targets=None):
        target_corners_label = F.softmax(target_corners, axis=-1)
        loss_dfl = F.cross_entropy(
            pred_corners,
            target_corners_label,
            soft_label=True,
            reduction='none')
        loss_dfl = loss_dfl.sum(1)
        if weight_targets is not None:
            loss_dfl = loss_dfl * (weight_targets.expand([-1, 4]).reshape([-1]))
            loss_dfl = loss_dfl.sum(-1) / weight_targets.sum()
        else:
            loss_dfl = loss_dfl.mean(-1)
        loss_dfl = loss_dfl / 4.  # 4 direction
        return loss_dfl
