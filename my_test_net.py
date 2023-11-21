#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
from pytorchvideo.layers.distributed import get_local_rank

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter
from slowfast.utils.env import pathmgr

logger = logging.get_logger(__name__)


@torch.no_grad()

def test(cfg):

    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    try:
        du.init_distributed_training(cfg)
    except:
        du.init_distributed_training(cfg.NUM_GPUS, cfg.SHARD_ID)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    if len(cfg.TEST.NUM_TEMPORAL_CLIPS) == 0:
        cfg.TEST.NUM_TEMPORAL_CLIPS = [cfg.TEST.NUM_ENSEMBLE_VIEWS]

    test_meters = []
    for num_view in cfg.TEST.NUM_TEMPORAL_CLIPS:

        cfg.TEST.NUM_ENSEMBLE_VIEWS = num_view

        # Print config.
        logger.info("Test with config:")
        logger.info(cfg)
        # Build the video model and print model statistics.
        model = build_model(cfg)
         
        if not cfg.TEST.CUSTOM_LOAD:
            cu.load_test_checkpoint(cfg, model)

        # custom load checkpoint here
        if cfg.TEST.CUSTOM_LOAD:
            custom_load_file = cfg.TEST.CUSTOM_LOAD_FILE
            assert pathmgr.exists(
                    custom_load_file
            ), "Checkpoint '{}' not found".format(custom_load_file)
            logger.info("Loading custom network weights from {}.".format(custom_load_file)) 
            print('Sono qui')
            checkpoint = torch.load(custom_load_file, map_location='cpu')
            checkpoint_model = checkpoint['model_state']
            state_dict = model.state_dict()
             
            if cfg.TEST.PATCHING_MODEL and cfg.TEST.CLIP_ORI_PATH:
                logger.info("patching model")
                patching_ratio = cfg.TEST.PATCHING_RATIO
                try:
                    clip_ori_state = torch.jit.load(cfg.TEST.CLIP_ORI_PATH, map_location='cpu').state_dict()
                    # pop some unnessesary keys
                    _ = [clip_ori_state.pop(i) for i in ['input_resolution', 'context_length', 'vocab_size']]
                    raw_clip_flag = True
                except:
                    clip_ori_state = torch.load(cfg.TEST.CLIP_ORI_PATH, map_location='cpu')['model_state']
                    raw_clip_flag = False
                
                logger.info("model contains %d keys for patching"%len(checkpoint_model)) 
                logger.info("original clip model contains %d keys"%len(clip_ori_state))
                
                if cfg.MODEL.NUM_EXPERTS > 0:
                    for key in list(clip_ori_state.keys()):
                        if 'mlp' in key and key.startswith('visual'):
                            layer_id = int(key.split('.mlp')[0].split('.')[-1])
                            if layer_id not in cfg.MODEL.EXPERT_INSERT_LAYERS:
                                continue
                            for expert_id in range(cfg.MODEL.NUM_EXPERTS):
                                if 'c_fc' in key or 'gelu' in key:
                                    new_key = key.replace('mlp', 'experts_head.%d'%expert_id)
                                else:
                                    new_key = key.replace('mlp', 'experts_tail.%d'%expert_id)
                                clip_ori_state[new_key] = clip_ori_state[key]
                    
                    logger.info("expanded original clip model contains %d keys"%len(clip_ori_state))
                
                missing_params_name = None
                if len(clip_ori_state) == len(checkpoint_model):
                    logger.info("no extra params added")
                else:
                    if raw_clip_flag:
                        logger.info("Missing Params for patching:")
                        logger.info(list(set(checkpoint_model.keys())-set(['model.'+i for i in clip_ori_state.keys()])))
                        missing_params_name = list(set(checkpoint_model.keys())-set(['model.'+i for i in clip_ori_state.keys()]))
                    else:
                        missing_params_name = list(set(checkpoint_model.keys())-set([i for i in clip_ori_state.keys()]))
                    
                # add model prefix
                patching_checkpoint_model = {}
                for key in clip_ori_state:
                    if raw_clip_flag:
                        patching_checkpoint_model['model.'+key] = clip_ori_state[key] * cfg.TEST.PATCHING_RATIO + checkpoint_model['model.'+key] * (1 - cfg.TEST.PATCHING_RATIO)
                    else:
                        if key not in checkpoint_model:
                            continue
                        
                        patching_checkpoint_model[key] = clip_ori_state[key] * cfg.TEST.PATCHING_RATIO + checkpoint_model[key] * (1 - cfg.TEST.PATCHING_RATIO)
                
                if missing_params_name != None:
                    for key in missing_params_name:
                        patching_checkpoint_model[key] = checkpoint_model[key]

                checkpoint_model = patching_checkpoint_model

            if 'module' in list(state_dict.keys())[0]:
                new_checkpoint_model = {} 
                for key, value in checkpoint_model.items():
                    new_checkpoint_model['module.' + key] = value
                checkpoint_model = new_checkpoint_model
            
            for key in checkpoint_model.keys(): 
                if key not in state_dict.keys():
                    logger.info("missing parameters")
                    logger.info(key)
            
            model.load_state_dict(checkpoint_model, strict=False)

        flops, params = 0.0, 0.0
        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            model.eval()
            flops, params = misc.log_model_info(
                model, cfg, use_train_input=False
            )

        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            misc.log_model_info(model, cfg, use_train_input=False)
        model.init_knn_labels(train_loader)
        

        # # Perform multi-view test on the entire dataset.
        test_meter = perform_test(test_loader, model, test_meter, cfg, writer)

    ## PERFORM
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()
    
    if cfg.MODEL.RECORD_ROUTING:
        rout_list = []

    for cur_iter, (inputs, labels, video_idx, time, meta) in enumerate(
        test_loader
    ):
       
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()

            # Perform the forward pass.
            if cfg.MODEL.RECORD_ROUTING:
                preds, routing_state = model(inputs)
                # routing_state shape [layer_num, patch_num, bz * clip_len, 2)
                rshape = routing_state.shape
                routing_state = routing_state.reshape(rshape[0], rshape[1], inputs[0].shape[0], -1, 2).permute(2, 0, 1, 3, 4)
                if get_local_rank() == 0:
                    if cur_iter % 10 == 0:
                        print(routing_state[:,:,:,:,0].mean(-1).mean(0).detach().cpu().squeeze().numpy())
            
            elif cfg.MODEL.KEEP_RAW_MODEL and cfg.MODEL.ENSEMBLE_PRED:
                preds, raw_preds = model(inputs)
                preds = cfg.MODEL.ENSEMBLE_RAWMODEL_RATIO * raw_preds + (1 - cfg.MODEL.ENSEMBLE_RAWMODEL_RATIO) * preds

            else:
                preds = model(inputs)
        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            preds, labels, video_idx = du.all_gather([preds, labels, video_idx])
            """
            if cfg.MODEL.RECORD_ROUTING:
                routing_state = du.all_gather([routing_state])[0]
                routing_state = routing_state.cpu()
                rout_list.append(routing_state)
            """
            # if cfg.MODEL.RECORD_ROUTING and cur_iter >= 10:
            #     break

        if cfg.NUM_GPUS:
            preds = preds.cpu()
            labels = labels.cpu()
            video_idx = video_idx.cpu()




    return result_string + " \n " + result_string_views
