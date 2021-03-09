from mmseg.apis import inference_segmentor, init_segmentor
import mmcv


config_file = 'lh_exp/road_deeplabv3plus/lh_deeplabv3plus.py'
checkpoint_file = '/home/lh/CODE/mmsegmentation/lh_exp/road_deeplabv3plus/latest.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
# or img = mmcv.imread(img), which will only load it once
img = '/home/lh/CODE/mmsegmentation/data/road_dataset/img_dir/train/104_sat.jpg'
result = inference_segmentor(model, img)

dict = {}
for i in range(len(result[0])):
    for j in range(len(result[0][i])):
        dict[result[0][i][j]] = dict.setdefault(result[0][i][j], 0) + 1
print(dict)
# visualize the results in a new window
# model.show_result(img, result, show=True)
# or save the visualization results to image files
model.show_result(
    img, result, out_file='/home/lh/CODE/mmsegmentation/result.jpg')

# test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#    result = inference_segmentor(model, frame)
#    model.show_result(frame, result, wait_time=1)
