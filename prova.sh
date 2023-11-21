#!/bin/bash

ROOT=/Open-VCLIP
CKPT=$ROOT/ckpt/
OUT_DIR=$CKPT/testing

LOAD_CKPT_FILE=$CKPT/downloaded/openvclip-b16/swa_2_22.pth
PATCHING_RATIO=0.5

# DATA.PATH_TO_DATA_DIR $ROOT/zs_label_db/ucf101_full \ # option: ucf101full / ucf101_split1 / ucf101_split2 / ucf101_split3 / 
# DATA.PATH_PREFIX      # need to replace with the path of the dataset
# TEST.CUSTOM_LOAD_FILE # path of checkpoint to be loaded
# TEST.PATCHING_RATIO   # relates to the patching ratio: [old_w * ratio + new_w * (1 - ratio)]
# TEST.CLIP_ORI_PATH    # need to replace with the path of CLIP weights
# MODEL.TEMPORAL_MODELING_TYPE # selection of temporal modeling module

cd $ROOT
#python -W ignore -u $ROOT/run_net.py \
python -W ignore -u $ROOT/run_devel.py \
    --cfg $ROOT/configs/Kinetics/TemporalCLIP_vitb16_8x16_STAdapter.yaml \
    --opts DATA.PATH_TO_DATA_DIR $ROOT/zs_label_db/ucf101_full \
    DATA.PATH_PREFIX /dev/shm/ucf/UCF-101 \
    DATA.PATH_LABEL_SEPARATOR , \
    DATA.INDEX_LABEL_MAPPING_FILE $ROOT/zs_label_db/ucf101-index2cls.json \
    TRAIN.ENABLE False \
    OUTPUT_DIR $OUT_DIR \
    TEST.BATCH_SIZE 480 \
    NUM_GPUS 1 \
    DATA.DECODING_BACKEND "pyav" \
    MODEL.NUM_CLASSES 101 \
    MODEL.TEMPORAL_MODELING_TYPE 'expand_temporal_view'
   # TEST.CUSTOM_LOAD True \
   # TEST.CUSTOM_LOAD_FILE $ROOT/ckpt/downloaded/openvclip-b16/swa_2_22.pth \
   # TEST.CHECKPOINT_FILE_PATH $LOAD_CKPT_FILE \
   # TEST.SAVE_RESULTS_PATH temp.pyth \
   # TEST.NUM_ENSEMBLE_VIEWS 3 \
   # TEST.NUM_SPATIAL_CROPS 1 \
   # TEST.PATCHING_MODEL True \
   # TEST.PATCHING_RATIO $PATCHING_RATIO \
   # TEST.CLIP_ORI_PATH /root/.cache/clip/ViT-B-16.pt \
