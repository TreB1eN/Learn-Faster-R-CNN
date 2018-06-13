import torch as t
from torch import nn
from torchvision.models import vgg16
from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from model.roi_module import RoIPooling2D
from utils import array_tool as at
from utils.config import opt


def decom_vgg16():
    # the 30th layer of features is relu of conv5_3
    if opt.caffe_pretrain:
        model = vgg16(pretrained=False)
        if not opt.load_path:
            model.load_state_dict(t.load(opt.caffe_pretrain_path))
    else:
        model = vgg16(not opt.load_path)

    features = list(model.features)[:30]
    classifier = model.classifier

    classifier = list(classifier)
    del classifier[6]
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier


class FasterRCNNVGG16(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 n_fg_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):
                 
        extractor, classifier = decom_vgg16()

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
        )


class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        """
        Sequential(
          (0): Linear(in_features=25088, out_features=4096, bias=True)
          (1): ReLU(inplace)
          (2): Linear(in_features=4096, out_features=4096, bias=True)
          (3): ReLU(inplace)
        )
        """
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class # 21 for voc
        self.roi_size = roi_size # 7
        self.spatial_scale = spatial_scale # 1/16
        self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        """
        x.shape, rois.shape, roi_indices.shape = 
        (torch.Size([1, 512, 37, 56]), (128, 4), torch.Size([128]))
        """
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1) 
        # indices_and_rois.shape : torch.Size([128, 5])
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = t.autograd.Variable(xy_indices_and_rois.contiguous())

        pool = self.roi(x, indices_and_rois)
        """
        pool.shape : torch.Size([128, 512, 7, 7])
        
        RoI Pooling 是一种特殊的Pooling操作，
        给定一张图片的Feature map (512×H/16×W/16) ，
        和128个候选区域的座标（128×4），
        RoI Pooling将这些区域统一下采样到 （512×7×7），
        就得到了128×512×7×7的向量。
        可以看成是一个batch-size=128，
        通道数为512，7×7的feature map
        """
        pool = pool.view(pool.size(0), -1)
        #  pool.shape : torch.Size([128, 25088])
        fc7 = self.classifier(pool)
        """
        Sequential(
          (0): Linear(in_features=25088, out_features=4096, bias=True)
          (1): ReLU(inplace)
          (2): Linear(in_features=4096, out_features=4096, bias=True)
          (3): ReLU(inplace)
        )
        注意这里把原来vgg的classifier拉过来做为初始化
        """
        roi_cls_locs = self.cls_loc(fc7)
        # Linear(in_features=4096, out_features=84, bias=True)
        roi_scores = self.score(fc7)
        # Linear(in_features=4096, out_features=21, bias=True)
        return roi_cls_locs, roi_scores
        # (torch.Size([128, 84]), torch.Size([128, 21]))
        """
        对于分类问题,直接利用交叉熵损失. 
        而对于位置的回归损失,一样采用Smooth_L1Loss, 
        只不过只对正样本计算损失.
        而且是只对正样本中的这个类别4个参数计算损失。
        举例来说:
        一个RoI在经过FC 84后会输出一个84维的loc 向量. 
        如果这个RoI是负样本,则这84维向量不参与计算 L1_Loss
        如果这个RoI是正样本,属于label K,
        那么它的第 K×4, K×4+1 ，K×4+2， K×4+3 这4个数参与计算损失，
        其余的不参与计算损失。
        """
        """
        ROIHEAD做的事情是根据前面得到的128个roi框，
        去feature上分别做roi pool,
        得到[128,512,7,7]的最终信息
        相当于每一个roi框，不管他有多大，
        统统roi pool到[512,7,7]
        再然后就是几个linear layer,
        从512*7*7 = 25088 得到 21维的class score 和 84维的roi_cls_locs
        最终输出是[128, 84]，[128, 21]
        """


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
