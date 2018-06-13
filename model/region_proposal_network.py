import numpy as np
from torch.nn import functional as F
import torch as t
from torch import nn

from model.utils.bbox_tools import generate_anchor_base
from model.utils.creator_tool import ProposalCreator


class RegionProposalNetwork(nn.Module):
    """Region Proposal Network introduced in Faster R-CNN.

    This is Region Proposal Network introduced in Faster R-CNN [#]_.
    This takes features extracted from images and propose
    class agnostic bounding boxes around "objects".

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        in_channels (int): The channel size of input.
        mid_channels (int): The channel size of the intermediate tensor.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.
        feat_stride (int): Stride size after extracting features from an
            image.
        initialW (callable): Initial weight value. If :obj:`None` then this
            function uses Gaussian distribution scaled by 0.1 to
            initialize weight.
            May also be a callable that takes an array and edits its values.
        proposal_creator_params (dict): Key valued paramters for
            :class:`model.utils.creator_tools.ProposalCreator`.

    .. seealso::
        :class:`~model.utils.creator_tools.ProposalCreator`

    """

    def __init__(
            self,
            in_channels=512,
            mid_channels=512,
            ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32],
            feat_stride=16,
            proposal_creator_params=dict(),
    ):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base(
            anchor_scales=anchor_scales, ratios=ratios)
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0]
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        # Conv2d(512, 18, kernel_size=(1, 1), stride=(1, 1))
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        # Conv2d(512, 36, kernel_size=(1, 1), stride=(1, 1))
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        """Forward Region Proposal Network.

        Here are notations.

        * :math:`N` is batch size.
        * :math:`C` channel size of the input.
        * :math:`H` and :math:`W` are height and witdh of the input feature.
        * :math:`A` is number of anchors assigned to each pixel.

        Args:
            x (~torch.autograd.Variable): The Features extracted from images.
                Its shape is :math:`(N, C, H, W)`.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
            scale (float): The amount of scaling done to the input images after
                reading them from files.

        Returns:
            (~torch.autograd.Variable, ~torch.autograd.Variable, array, array, array):

            This is a tuple of five following values.

            * **rpn_locs**: Predicted bounding box offsets and scales for \
                anchors. Its shape is :math:`(N, H W A, 4)`.
            * **rpn_scores**:  Predicted foreground scores for \
                anchors. Its shape is :math:`(N, H W A, 2)`.
            * **rois**: A bounding box array containing coordinates of \
                proposal boxes.  This is a concatenation of bounding box \
                arrays from multiple images in the batch. \
                Its shape is :math:`(R', 4)`. Given :math:`R_i` predicted \
                bounding boxes from the :math:`i` th image, \
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            * **roi_indices**: An array containing indices of images to \
                which RoIs correspond to. Its shape is :math:`(R',)`.
            * **anchor**: Coordinates of enumerated shifted anchors. \
                Its shape is :math:`(H W A, 4)`.

        """
        n, _, hh, ww = x.shape
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base), self.feat_stride, hh, ww)

        n_anchor = anchor.shape[0] // (hh * ww)  #9,总共anchors数目
        h = F.relu(self.conv1(x))
        # h.shape : torch.Size([1, 512, hh, ww])
        rpn_locs = self.loc(h)
        # rpn_loss.shape : torch.Size([1, 36, hh, ww])
        # 1*1 的conv收channel数
        
        # UNNOTE: check whether need contiguous
        # A: Yes
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        # reshape to [1,hh*ww*9,4]
        rpn_scores = self.score(h)
        # rpn_scores.shape : torch.Size([1, 18, 37, 50])
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        # reshape to [1,hh*ww*9,2]
        rpn_fg_scores = \
            rpn_scores.view(n, hh, ww, n_anchor, 2)[:, :, :, :, 1].contiguous()
        # rpn_fg_scores.shape = [1,hh*ww*9,1] 取最后一位
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        # reshape to [1,hh*ww*9]
        rpn_scores = rpn_scores.view(n, -1, 2)
        # rpn_scores.shape : torch.Size([1, hh*ww*9, 2])
        """
        对于每张图片，利用它的feature map， 
        计算 (H/16)× (W/16)×9（大概20000）个anchor属于前景的概率(rpn_fg_scores)，
        以及对应的网络预测的需要修正的位置参数(rpn_locs)。
        """
        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(), #梯度在这里就断了
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor,
                img_size,
                scale=scale)
            """
            proposal_layer做的事情是：
            对于每张图片，根据前面算出来的前景的概率（rpn_fg_scores），
            选取概率较大的12000个anchor，
            利用回归的位置参数(rpn_locs)，修正这12000个anchor的位置，得到RoIs
            利用非极大值（(Non-maximum suppression, NMS）抑制，选出概率最大的2000个RoIs
            
            注意：在inference的时候，为了提高处理速度，12000和2000分别变为6000和300.
            注意：这部分的操作不需要进行反向传播，因此可以利用numpy/tensor实现。
            """
            batch_index = i * np.ones((len(roi), ), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)
            # 上面这三行是为batch_size > 1准备的，但这个实现还不支持batch_size > 1

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    # it seems that it can't be boosed using GPU
    import numpy as xp
    # feat_stride = 16
    shift_y = xp.arange(0, height * feat_stride, feat_stride)
    shift_x = xp.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    # shift_x.shape,shift_y.shape 都等于 [height,width]
    # 相当于构建了一个 [height,width] 的网格，每一个网格的大小是16 * 16
    shift = xp.stack(
        (shift_y.ravel(), shift_x.ravel(), shift_y.ravel(), shift_x.ravel()),
        axis=1)
    # shift.shape = (height*width, 4)
    # shift 相当于这些网格左上角和（0,0）点的偏移量，所以叫shift

    A = anchor_base.shape[0]  # 9,anchor数量
    K = shift.shape[0]  # height*width,所有网格数
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    # [1,9,4] + [K,1,4] = [K,9,4]
    # K个网格，每个网格匹配上9种anchor_size
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    # [K*9,4], K*9 个predefined anchors
    return anchor


def _enumerate_shifted_anchor_torch(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    # !TODO: add support for torch.CudaTensor
    # xp = cuda.get_array_module(anchor_base)
    import torch as t
    shift_y = t.arange(0, height * feat_stride, feat_stride)
    shift_x = t.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)
    shift = xp.stack(
        (shift_y.ravel(), shift_x.ravel(), shift_y.ravel(), shift_x.ravel()),
        axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
            mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()