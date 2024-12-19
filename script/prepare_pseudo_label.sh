# Root directory containing datasets
root="data"
scale=1
ls $root
# Loop through all subdirectories in the root directory
## EDIT: changed from "root"/pi* to "root"/*
for dataset_name in "$root"/*; do
  # Check if it's a directory (not a file)
  if [ -d "$dataset_name" ]; then

    # 1. DEVA anything mask (logic within the loop)
    cd Tracking-Anything-with-DEVA/

    if [ "$scale" = "1" ]; then
        img_path="../${dataset_name}/images"
    else
        img_path="../${dataset_name}/images_${scale}"
    fi
    echo "im $img_path"

    # colored mask for visualization check
    # python demo/demo_automatic.py \
    #   --chunk_size 4 \
    #   --img_path "$img_path" \
    #   --amp \
    #   --temporal_setting semionline \
    #   --size 480 \
    #   --output "./example/output_gaussian_dataset/${dataset_name}" \
    #   --suppress_small_objects  \
    #   --SAM_PRED_IOU_THRESHOLD 0.7 \


    # mv ./example/output_gaussian_dataset/${dataset_name}/Annotations ./example/output_gaussian_dataset/${dataset_name}/Annotations_color

    # gray mask for training
    python demo/demo_automatic.py \
      --chunk_size 4 \
      --img_path "$img_path" \
      --amp \
      --temporal_setting semionline \
      --size 480 \
      --use_short_id  \
      --output "./example/output_gaussian_dataset/${dataset_name}" \
      --suppress_small_objects  \
      --SAM_PRED_IOU_THRESHOLD 0.7 \
 
    # 2. copy gray mask to the correponding $root path
    mv ./example/output_gaussian_dataset/${dataset_name}/Annotations ../${dataset_name}/object_mask
    cd ..
  fi
done
