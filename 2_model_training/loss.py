import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert, box_iou, complete_box_iou_loss

class CustomLoss(nn.Module):
    """Custom loss function for object detection combining CIOU loss, objectness loss,
    and class prediction loss."""
    def __init__(self, num_anchors=1):
        super().__init__()
        self.na = num_anchors

    def forward(self, pred, target):
        bs = target.shape[0]
        total_loss = 0
        for img_no in range(bs):
            target_tensor = target[img_no]
            pred_tensor = pred[img_no]

            # Extract components from the prediction tensor
            pred_o = pred_tensor[..., 4]
            pred_c = pred_tensor[..., 5:]

            # Extract components from the target tensor
            target_o = target_tensor[..., 4]
            target_c = target_tensor[..., 5:]

            # Apply mask to retain only the cases where target_p = 1
            mask = target_o == 1

            target_xywh = target_tensor[..., :4]
            target_xywh = target_xywh[mask]

            pred_xywh = pred_tensor[..., :4]

            pred_xywh_anchors = [pred_xywh[anc_id ,...][mask] for anc_id in range(self.na)]
            box_ious = [torch.diagonal(box_iou(box_convert(target_xywh,
                                                           'cxcywh',
                                                           'xyxy'),
                                               box_convert(pred_xywh_anchors[anc_id],
                                                           'cxcywh',
                                                           'xyxy') ))
                        for anc_id in range(self.na)]

            stacked_tensor = torch.stack(box_ious)
            _ ,argmaxes = torch.max(stacked_tensor, dim=0)

            pred_xywh = [pred_xywh_anchors[argmax][i]  for i, argmax in enumerate(argmaxes)]
            pred_xywh = torch.stack(pred_xywh)

            # Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format
            box1_xyxy = box_convert(pred_xywh, in_fmt='cxcywh', out_fmt='xyxy')
            box2_xyxy = box_convert(target_xywh, in_fmt='cxcywh', out_fmt='xyxy')

            # Calculate the Complete IoU (CIOU) loss using torchvision.ops
            ciou_loss = complete_box_iou_loss(box1_xyxy, box2_xyxy, reduction='mean')

            # # Binary Cross-Entropy Loss for objectness score where target_p = 1
            pred_o_anchors = [pred_o[anc_id ,...][mask] for anc_id in range(self.na)]
            pred_o = [pred_o_anchors[argmax][i]  for i, argmax in enumerate(argmaxes)]
            pred_o = torch.stack(pred_o)
            obj_loss = F.binary_cross_entropy(pred_o, target_o[mask])

            # # Binary Cross-Entropy Loss for class predictions where target_p = 1
            pred_c_anchors = [pred_c[anc_id ,...][mask] for anc_id in range(self.na)]
            pred_c = [pred_c_anchors[argmax][i]  for i, argmax in enumerate(argmaxes)]
            pred_c = torch.stack(pred_c)
            class_loss = F.binary_cross_entropy(pred_c, target_c[mask])

            # Total loss as a combination of all individual losses
            total_loss += ciou_loss + obj_loss + class_loss
        total_loss /= bs
        return total_loss
