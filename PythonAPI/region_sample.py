import numpy as np
import json
from pycocotools._mask import _frString, decode

#from pycocotools.mask import
class RegionSample():
    """a class for targeted sampling of regions from MS-COCO RLE files
    """
    def __init__(self, enc, label=None):
        """Inputs:
        enc   -- list of RLE encoded input dictionaries, or a single encoding
                 {'counts':..., 'size':[...]}
        label -- index of the encoding to use for sampling
        """
        if isinstance(enc, str):
            with open(enc, 'r') as fh:
                enc = json.load(fh)
            if label is not None:
                enc = enc[label]

        assert isinstance(enc, dict)
        self.enc = enc
        cocomask = _frString([enc])
        self.area = cocomask.area(0)
        maskarray = cocomask[0]#.copy()
        height = enc['size'][0]
        self.height = height
        self.sorted_inds_pos = np.cumsum(np.hstack([[0], maskarray[1::2]])).astype('uint32')
        self.sorted_inds_neg =  np.cumsum( np.hstack([ maskarray[::2]])).astype('uint32')
        self.sorted_inds_tot_flat =  np.cumsum(maskarray).astype('uint32')
        self.max_pos = (self.sorted_inds_pos)[-1]

    def __call__(self):
        return self.pos_ind_to_xy(self.sample_pos_ind())

    def decode(self):
        return decode([self.enc])[:,:,0]

    def pos_ind_to_xy(self, pos_ind):
        if isinstance(pos_ind, int):
            assert pos_ind <= self.max_pos
        else:
            assert all(pos_ind <= self.max_pos)
            pos_ind = pos_ind.reshape(-1,1)
        segm = np.argmin(self.sorted_inds_pos <= pos_ind, axis=0) -1
        offset = pos_ind - self.sorted_inds_pos[segm]
        point_flat = self.sorted_inds_tot_flat[2*segm] + offset
        y = point_flat % self.height
        x =  point_flat // self.height
        return x,y

    def sample_pos_ind(self, n=None):
        return np.random.randint(self.max_pos, size=n )
