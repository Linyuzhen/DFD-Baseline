methods=("Deepfakes" "Face2Face" "FaceSwap" "NeuralTextures" "FaceShifter")
quality=("c23" "c40")

# resnet18
for q in ${quality[*]}
do
for m1 in ${methods[*]}
do
for m2 in ${methods[*]}
do
  python test.py --gpu_id 1 --train_method $m1 --test_method $m2 --qp $q --img_size 224 --backbone resnet18 --save_test_path './rs18_results'
done
done
done

# xception
for q in ${quality[*]}
do
for m1 in ${methods[*]}
do
for m2 in ${methods[*]}
do
  python test.py --gpu_id 0 --train_method $m1 --test_method $m2 --qp $q --img_size 299 --backbone xception --save_test_path './xpt_results'
done
done
done

# efn-b4
for q in ${quality[*]}
do
for m1 in ${methods[*]}
do
for m2 in ${methods[*]}
do
  python test.py --gpu_id 0 --train_method $m1 --test_method $m2 --qp $q --img_size 380 --backbone efn-b4 --save_test_path './enb4_results'
done
done
done