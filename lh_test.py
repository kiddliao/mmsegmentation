import mmcv
gt_seg_map = mmcv.imread(
    'data/road_dataset/ann_L_dir/val/978495_sat.png', flag='unchanged', backend='pillow')
print(gt_seg_map.shape)
'data/road_dataset/ann_dir/val/978495_sat.png'
