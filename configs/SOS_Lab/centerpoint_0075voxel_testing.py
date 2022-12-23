_base_ = ['../centerpoint/centerpoint_0075voxel_second_secfpn_4x8_cyclic_20e_nus.py',
          './SOSLab.py',]
"""
pcr = [0.0, 0.0, -2.0, 18.0, 18.0, 10.0]
ss =[41, 1440, 1440]
vs = [
    (pcr[3] - pcr[0]) / ss[1],
    (pcr[4] - pcr[1]) / ss[2],
    (pcr[5] - pcr[2]) / (ss[0] - 1),
]

model = dict(
    type='CenterPointFixed',
    pts_voxel_layer=dict(voxel_size=vs, point_cloud_range=pcr),
    pts_voxel_encoder=dict(num_features=4),
    pts_middle_encoder=dict(
        in_channels=4),
    pts_bbox_head=dict(
        type='CenterHeadWithVel',
        tasks=[
            dict(num_class=1, class_names=['Pedestrian']),
        ],
        bbox_coder=dict(
            post_center_range=pcr,
            voxel_size=vs[:2],
            code_size=7),
    ),
)

train_cfg=dict(
    pts=dict(
        voxel_size=vs,
        code_weights=[1., 1., 1., 1., 1., 1., 1., 1.],
        post_center_limit_range=pcr,
        point_cloud_range=pcr)),

test_cfg=dict(
    pts=dict(
        voxel_size=vs[:2],
        post_center_limit_range=pcr,
        point_cloud_range=pcr)),
"""