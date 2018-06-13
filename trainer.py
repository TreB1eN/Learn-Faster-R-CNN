from collections import namedtuple
import time
from torch.nn import functional as F
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator

from torch import nn
import torch as t
from torch.autograd import Variable
from utils import array_tool as at
from utils.vis_tool import Visualizer

from utils.config import opt
# from torchnet.meter import ConfusionMeter, AverageValueMeter

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])


class FasterRCNNTrainer(nn.Module):
    """wrapper for conveniently training. return losses

    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for \
        Region Proposal Network (RPN).
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_loc_loss`: The localization loss for the head module.
    * :obj:`roi_cls_loss`: The classification loss for the head module.
    * :obj:`total_loss`: The sum of 4 loss above.

    Args:
        faster_rcnn (model.FasterRCNN):
            A Faster R-CNN model that is going to be trained.
    """

    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        # target creator create gt_bbox gt_label etc as training targets. 
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()
        # visdom wrapper
        self.vis = Visualizer(env=opt.env)

        # indicators for training status
#         self.rpn_cm = ConfusionMeter(2)
#         self.roi_cm = ConfusionMeter(21)
#         self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

    def forward(self, imgs, bboxes, labels, scale):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~torch.autograd.Variable): A variable with a batch of images.
            bboxes (~torch.autograd.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            labels (~torch.autograd..Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            scale (float): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            namedtuple of 5 losses
        """
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(features, img_size, scale)
        """
        rpn_locs.shape, rpn_scores.shape, rois.shape, roi_indices.shape, anchor.shape = 
        (torch.Size([1, 18648, 4]),
         torch.Size([1, 18648, 2]),
         (1714, 4),
         (1714,),
         (18648, 4))
         rpn网络做的事情是：
         对于每张图片，利用它的feature map， 
        计算 (H/16)× (W/16)×9（大概20000）个anchor属于前景或背景的概率(rpn_scores)，
        以及对应的网络预测的需要修正的位置参数(rpn_locs)。
        然后，对于每张图片，根据前面算出来的前景的概率（rpn_fg_scores），
        选取概率较大的12000个anchor，
        利用回归的位置参数(rpn_locs)，修正这12000个anchor的位置，得到RoIs
        利用非极大值（(Non-maximum suppression, NMS）抑制，选出概率最大的2000个RoIs
            
        注意：在inference的时候，为了提高处理速度，12000和2000分别变为6000和300.          
        """
        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois
        """
        bbox.shape,label.shape,rpn_score.shape,rpn_loc.shape,roi.shape = 
        (torch.Size([2, 4]),
         torch.Size([2]),
         torch.Size([16650, 2]),
         torch.Size([16650, 4]),
         (2000, 4))
        """

        # Sample RoIs and forward
        # it's fine to break the computation graph of rois, 
        # consider them as constant input
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox), # at = array_tools,tensor to numpy 用不着了，在pytorch0.4里
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        """
        sample_roi.shape, gt_roi_loc.shape, gt_roi_label.shape = 
        ((128, 4), (128, 4), (128,))
        proposal_target_creator的作用是：
        RPN会产生大约2000个RoIs，这2000个RoIs不是都拿去训练，
        而是利用ProposalTargetCreator 选择128个RoIs用以训练。选择的规则如下：
        RoIs和gt_bboxes 的IoU大于0.5的，选择一些（比如32个）
        选择 RoIs和gt_bboxes的IoU小于0.5，同时大于等于0（或者0.1）的选择一些（比如 128-32=96个）作为负样本
        为了便于训练，对选择出的128个RoIs，还对他们的gt_roi_loc 进行标准化处理（减去均值除以标准差）
        最终输出128个roi框及其分别对应的需要修正的[ty,tx,th,tw]和label
        """
        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = t.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)
        """
        x.shape, rois.shape, roi_indices.shape = (torch.Size([1, 512, 37, 56]), (128, 4), torch.Size([128]))
        ROIHEAD做的事情是根据前面得到的128个roi框，
        去feature上分别做roi pool,
        得到[128,512,7,7]的最终信息
        相当于每一个roi框，不管他有多大，
        统统roi pool到[512,7,7]
        再然后就是几个linear layer,
        从512*7*7 = 25088 得到 21维的class score 和 84维的roi_cls_locs
        最终输出是[128, 84]，[128, 21]
        """
        # ------------------ RPN losses -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            img_size)
        """
        所以总的来说，AnchorTargetCreator做的事情是：
        根据每一个预先设定的anchor和这张图片的gt_bbox去计算iou,
        再用求得的iou来给每一个anchor打标签，
        1是正样本，0是负样本，-1表示不关心，不参与后续计算
        打标签是通过
        正负样本之和应该是self.n_sample，比例是self.pos_ratio
        打标签的依据是：
        1. iou < 0.3的都算负样本
        2. 对每一个gt_object，标记和它iou最高的的anchor为正样本
            可能同时有多个anchor同时iou最高（相等）
        3. 剩下的anchor里面，iou大于0.7的也算正样本
        4. 还要平衡一下正负样本的数量和比例
        
        它不但打标签，还会计算每一个anchor和它最匹配的gt_bbox的loc,
        用于后续的bbox回归loss计算
        最后，返回的是loc和label # ((16650,), (16650, 4))
        """
        gt_rpn_label = at.tovariable(gt_rpn_label).long()
        gt_rpn_loc = at.tovariable(gt_rpn_loc)
        # gt_rpn_loc.shape, gt_rpn_label.shape ： ((18648, 4), (18648,))
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma) # loss value

        # NOTE: default value of ignore_index is -100 ...
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        """
        rpn_score.shape,gt_rpn_label.shape : (torch.Size([15318, 2]), torch.Size([15318]))
        
         ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Default: -100
        """
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
        # _gt_rpn_label.shape,_rpn_score.shape : (torch.Size([256]), (256, 2))
        
#         self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0] # 128
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4) 
        #torch.Size([128, 84]) to torch.Size([128, 21, 4])
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), \
                              at.totensor(gt_roi_label).long()]
        # 21个class的loc,取对应的gt制定的那个，即gt_roi_label
        # torch.Size([128, 4]) 
        gt_roi_label = at.tovariable(gt_roi_label).long()
        gt_roi_loc = at.tovariable(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

#         self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.
        
        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        t.save(save_dict, save_path)
        self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    flag = Variable(flag)
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    """
    论文里没见有这个sigma参数啊，
    sigma = 1时就说论文里原来的smooth_l1_loss
    当sigma > 1时，相当于变相的把关注的区间变窄了
    关注的权重放大了
    """
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda() # 上来先全设成0 torch.Size([18648, 4])
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    # 把 in_weight gt_label >0 地方置1，这样就实现了过滤哪些不参与计算的loc和cls
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, Variable(in_weight), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= (gt_label >= 0).sum().to(t.float)  # ignore gt_label==-1 for rpn_loss
    return loc_loss
