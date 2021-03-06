# -*- coding: utf-8 -*-
"""
# @file name  : detect.py
# @author     : chenzhanpeng https://github.com/chenzpstar
# @date       : 2022-01-11
# @brief      : FCOS检测后处理类
"""

import torch
import torch.nn as nn

from models.config import FCOSConfig
from models.utils import box_nms, decode_preds, reshape_feats


class FCOSDetect(nn.Module):
    def __init__(self, cfg=None):
        super(FCOSDetect, self).__init__()
        if cfg is None:
            self.cfg = FCOSConfig
        else:
            self.cfg = cfg

        self.use_ctr = self.cfg.use_ctr
        self.strides = self.cfg.strides
        self.score_thr = self.cfg.score_thr
        self.nms_iou_thr = self.cfg.nms_iou_thr
        self.max_boxes_num = self.cfg.max_boxes_num
        self.nms_mode = self.cfg.nms_mode

    def forward(self, preds):
        cls_logits, reg_preds, ctr_logits = preds

        cls_logits = reshape_feats(cls_logits)  # bchw -> b(hw)c
        ctr_logits = reshape_feats(ctr_logits)  # bchw -> b(hw)c
        cls_preds = cls_logits.sigmoid()
        ctr_preds = ctr_logits.sigmoid()

        cls_scores, pred_labels = cls_preds.max(dim=-1)  # b(hw)c -> b(hw)
        if self.use_ctr:
            cls_scores = (cls_scores * ctr_preds.squeeze(dim=-1)).sqrt()
        pred_labels += 1

        pred_boxes = decode_preds(reg_preds, self.strides)  # bchw -> b(hw)c

        return self._post_process((cls_scores, pred_labels, pred_boxes))

    def _post_process(self, preds):
        cls_scores, pred_labels, pred_boxes = preds
        batch_size = cls_scores.shape[0]  # b(hw)

        max_num = min(self.max_boxes_num, cls_scores.shape[-1])
        topk_idx = torch.topk(cls_scores, max_num, dim=-1)[1]
        assert topk_idx.shape == (batch_size, max_num)

        nms_cls_scores = []
        nms_pred_labels = []
        nms_pred_boxes = []

        for i in range(batch_size):
            # 1.挑选topk
            topk_cls_scores = cls_scores[i][topk_idx[i]]
            topk_pred_labels = pred_labels[i][topk_idx[i]]
            topk_pred_boxes = pred_boxes[i][topk_idx[i]]

            # 2.过滤低分
            score_mask = topk_cls_scores > self.score_thr
            filter_cls_scores = topk_cls_scores[score_mask]
            filter_pred_labels = topk_pred_labels[score_mask]
            filter_pred_boxes = topk_pred_boxes[score_mask]

            # 3.计算nms
            nms_idx = self._batch_nms(
                filter_cls_scores,
                filter_pred_labels,
                filter_pred_boxes,
                self.nms_iou_thr,
                self.nms_mode,
            )
            nms_cls_scores.append(filter_cls_scores[nms_idx])
            nms_pred_labels.append(filter_pred_labels[nms_idx])
            nms_pred_boxes.append(filter_pred_boxes[nms_idx])

        return nms_cls_scores, nms_pred_labels, nms_pred_boxes

    def _batch_nms(self, cls_scores, labels, boxes, thr, mode="iou"):
        if boxes.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=boxes.device)
        assert boxes.shape[-1] == 4

        coord_max = boxes.max()
        offsets = labels.to(boxes) * (coord_max + 1)
        nms_boxes = boxes + offsets.unsqueeze(dim=-1)

        return box_nms(cls_scores, nms_boxes, thr, mode)


if __name__ == "__main__":

    import torch
    torch.manual_seed(0)

    model = FCOSDetect()

    preds = (
        [torch.rand(2, 3, 2, 2)] * 5,
        [torch.rand(2, 4, 2, 2)] * 5,
        [torch.rand(2, 1, 2, 2)] * 5,
    )
    out = model(preds)
    [print(batch_out.shape) for result_out in out for batch_out in result_out]
