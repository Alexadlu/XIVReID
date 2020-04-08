# Experiment all tricks with center loss : 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on-triplet_centerloss0_0005    /data1/lidg/reid_dataset/IV-ReID/split   /home/lidg/reid-strong-baseline/data
# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# with center loss
python3 tools/train.py --config_file='configs/softmax_triplet_with_center.yml' MODEL.DEVICE_ID "('2,3')" DATASETS.NAMES "('regdb')" DATASETS.ROOT_DIR "('/data1/lidg/reid_dataset/IV-ReID/RegDB')" OUTPUT_DIR "('/home/lidg/reid-strong-baseline/logs/regdb/rgb-infrared-gray-cmt-cc')"