from mmdet3d.models.detectors.centerpoint import CenterPoint
from mmdet3d.models.builder import HEADS, DETECTORS
import torch.nn.functional as F
import torch


@DETECTORS.register_module()
class CenterPointFixed(CenterPoint):
    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        if self.train_cfg is not None and \
           self.train_cfg['pts']['out_size_factor'] == 4:
            x[0] = F.interpolate(x[0], scale_factor=2, mode='bilinear')
        return x