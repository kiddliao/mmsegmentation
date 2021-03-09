import mmcv
from PIL import Image
from collections import Counter

file_client_args = {'backend': 'disk'}
file_client = mmcv.FileClient(**file_client_args)
color_type = 'unchanged'
imdecode_backend = 'pillow'

path1 = '/home/lh/DATASETS/road_dataset/ann_dir/train/104_sat.png'
path2 = '/home/lh/DATASETS/VOC2012_AUG/VOCdevkit/VOC2012/SegmentationClass/2007_000032.png'
path3 = '/home/lh/DATASETS/VOC2012_AUG/VOCdevkit/VOC2012/SegmentationClassAug/2008_000002.png'
path4 = '/home/lh/DATASETS/VOC2012_AUG/VOCdevkit/VOC2012/SegmentationClassRaw/2007_000032.png'

def get_img(path):
    img_bytes = file_client.get(path)
    img = mmcv.imfrombytes(img_bytes, flag=color_type, backend=imdecode_backend)
    print(img.shape)

    val_map = dict()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            val_map[int(img[i][j])] = val_map.setdefault(int(img[i][j]),0) + 1
    print(val_map)

# get_img(path1)
get_img(path2)
get_img(path3)
get_img(path4)

# def pillow_mode(path):
#     img = Image.open(path)
#     print(img.mode)

# # pillow_mode(path1)
# # pillow_mode(path2)
# # pillow_mode(path3)
# # pillow_mode(path4)
# img = Image.open(path1)
# img.save('1.png')
# img_P = img.convert('P')
# img_P.save('1_p.png')
# path5 = '1_p.png'


# get_img(path5)
