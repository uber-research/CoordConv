class RPNParams(object):
    '''Holds params for RPN layer'''
    def __init__(self, anchors, rpn_hidden_dim, zero_box_conv, weight_init_std, anchor_scale):
        self.anchors = anchors
        self.rpn_hidden_dim = rpn_hidden_dim
        self.zero_box_conv = zero_box_conv
        self.weight_init_std = weight_init_std
        self.anchor_scale = anchor_scale

    @property
    def num_anchors(self):
        return self.anchors.shape[0]


class BoxSamplerParams(object):
    '''Holds params for BoxSample layer'''
    def __init__(self, hi_thresh, lo_thresh, sample_size):
        self.hi_thresh = hi_thresh
        self.lo_thresh = lo_thresh
        self.sample_size = sample_size
    
    #@property
    #def i_min(self):
    #    return 0. if self.remove_outbounds_boxes else None

    #@property
    #def j_min(self):
    #    return 0. if self.remove_outbounds_boxes else None
    #
    #@property
    #def i_max(self):
    #    return self.image_height-1.0 if self.remove_outbounds_boxes else None

    #@property
    #def j_max(self):
    #    return self.image_width-1.0 if self.remove_outbounds_boxes else None


class STNParams(object):
    '''Holds params for SpatialTransformer layer'''
    def __init__(self, out_h, out_w, dilate_ratio=1):
        self.out_h = out_h
        self.out_w = out_w
        self.dilate_ratio = dilate_ratio

class NMSParams(object):
    '''Holds params for BoxNMS layer (layer to replace in BoxSamplerPosNeg in test mode)'''
    def __init__(self, nms_thresh, max_proposals):
        self.nms_thresh = nms_thresh
        self.max_proposals = max_proposals

class RNNParams(object):
    '''Holds params for Recurrent layer implementation'''
    def __init__(self, vocab_size, emb_size, h_size, seq_length):
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.h_size = h_size
        self.seq_length = seq_length

class RecogParams(object):
    '''Holds params for Recognition base net implementation'''
    def __init__(self, h_size):
        self.h_size = h_size
