methods=("Deepfakes" "Face2Face" "FaceSwap" "NeuralTextures" "FaceShifter")
quality=("c23" "c40")

#!/bin/bash

# resnet18
for q in ${quality[*]}
do
for m in ${methods[*]}
do
  python train.py --gpu_id 1 --method $m --qp $q --img_size 224 --backbone resnet18
done
done

# xception
for q in ${quality[*]}
do
for m in ${methods[*]}
do
  python train.py --gpu_id 0 --method $m --qp $q --img_size 299 --backbone xception
done
done

# efn-b4
for q in ${quality[*]}
do
for m in ${methods[*]}
do
  python train.py --gpu_id 0 --method $m --qp $q --img_size 380 --backbone efn-b4
done
done