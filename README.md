# Deepfake Detection Baselines
Deepfake detection baselines with CNN backbones (Resnet, Xception and Efficientnet)

## Pre-processing
``` python
# extract face images from videos
python ext_ff.py
python ext_celeb.py
 
# split datasets
python get_txt.py
```

## Training
``` python
python train.py
# manage training processing with wandb
# python wandb_train.py
```

``` shell
# batch training
bash run_train.sh
```

## Testing
``` shell
# batch testing
bash run_test.sh
```

## Visualization
``` python
# visualize CAM-based results
python visualization.py
```