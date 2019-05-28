import subprocess as sbp
import shlex
import os


cfg = "/input/01_architectures/MRCNN/MRCNN/weakly_supervised/s2_2c/finetuning/resnet_101_2class_conf_recursive.yaml"
output_dir = "/output/01_Results/MRCNN/MRCNN/weakly_supervised/s2_2c/finetuning/w_c_4_5/"

initial_gt_VIS = "/dataIn/Train/VIS/SMD_VIS_skip_2_train_seg.json"
initial_gt_NIR = "/dataIn/Train/NIR/SMD_NIR_skip_2_train_seg.json"
original_gt_seg = "/dataIn/Train/VIS/SMD_VIS_skip_2_train_seg_original.json"
original_gt_rect = "/dataIn/Train/VIS/SMD_VIS_skip_2_train_seg_rect_original.json"
original_gt_nir = "/dataIn/Train/NIR/SMD_NIR_skip_2_train_seg_rect_original.json"

# just to be sure those two variables are intialized
subset = "VIS"
initial_gt = initial_gt_VIS

#####
# initial test
#####
#config =[
#"segm_init_LR_untouched",
#"segm_init_mask_LR_dominant",
#"rect_init_LR_untouched",
#"rect_init_mask_LR_dominant"
#]

#original_gt = [
#original_gt_seg, 
#original_gt_seg, 
#original_gt_rect, 
#original_gt_rect
#]

#loss_weights = [(1.0, 1.0, 1.0), 
#(0.5, 0.5, 1.0), 
#(1.0, 1.0, 1.0), 
#(0.5, 0.5, 1.0)
#]

#####
# ablation study
#####
config =[
#"segm_init_mask_LS_dominant_3_4",
#"segm_init_mask_LS_dominant_1_4",
#"segm_init_dect_LS_dominant",
"segm_init_mask_LS_dominant_NIR"
]

original_gt = [
#original_gt_seg, 
#original_gt_seg, 
#original_gt_seg,
original_gt_nir
]

loss_weights = [
#(0.75, 0.75, 1.0),
#(0.25, 0.25, 1.0),
#(1.0, 1.0, 0.5),
(0.5, 0.5, 1.0)
]

#####
# overfitting test
#####
#config =[
#"segm_init_mask_LR_dominant"
#]

#original_gt = [
#original_gt_seg
#]

#loss_weights = [
#(0.5, 0.5, 1.0)
#]

#train_weights = "/output/01_Results/MRCNN/MRCNN/weakly_supervised/s2_2c/finetuning/w_c_4_5/segm_init_mask_LR_dominant/training_round_14_1/train/smd_VIS_train_2_seg/generalized_rcnn/model_final.pkl"

def lr_cfg(loss_weights):
    return "FAST_RCNN.WEIGHT_LOSS_CLASSIFICATION {:.1f} FAST_RCNN.WEIGHT_LOSS_REGRESSION {:.1f} MRCNN.WEIGHT_LOSS_MASK {:.1f}".format(loss_weights[0], loss_weights[1], loss_weights[2])

# offset of configs as first config is already trained!
offset = 0
for idx, rec_train_cfg in enumerate(config):
    train_weights = "/input/01_architectures/MRCNN/weights/mask_reinit_last/ResNet101.pkl"
    
    if not os.path.exists(os.path.join(output_dir, rec_train_cfg)):
        os.makedirs(os.path.join(output_dir, rec_train_cfg))
    
    if "NIR" in rec_train_cfg:
        cfg = "/input/01_architectures/MRCNN/MRCNN/weakly_supervised/s2_2c/finetuning/resnet_101_2class_conf_recursive_NIR.yaml"
        subset = "NIR"
        initial_gt = initial_gt_NIR
    else:
        cfg = "/input/01_architectures/MRCNN/MRCNN/weakly_supervised/s2_2c/finetuning/resnet_101_2class_conf_recursive.yaml"
        subset = "VIS"
        initial_gt = initial_gt_VIS
    
    # set original GT as initial GT
    cmd = "cp {:s} {:s}".format(original_gt[idx + offset], initial_gt)
    sbp.call(shlex.split(cmd))
    #cmd = "cp {:s} {:s}".format("/output/01_Results/MRCNN/MRCNN/weakly_supervised/s2_2c/finetuning/w_c_4_5/segm_init_mask_LR_dominant/training_round_14_1/weakly_gt_14_1.json", initial_gt)
    #sbp.call(shlex.split(cmd))
    
    for i in range(15):
    #for i in range(15, 30):
        for j in range(2):
            # set annot_target
            annot_target = os.path.join(output_dir, rec_train_cfg,"training_round_{:d}_{:d}/weakly_gt_{:d}_{:d}.json".format(i,j,i,j))
            
            # train with "i-1" weights and annotations
            cmd = "python tools/train_net.py --cfg {:s} --multi-gpu-testing TRAIN.WEIGHTS {:s} OUTPUT_DIR {:s} TRAIN.AUTO_RESUME False {:s}".format(cfg, train_weights, os.path.join(output_dir, rec_train_cfg, "training_round_{:d}_{:d}/".format(i,j)), lr_cfg(loss_weights[idx + offset]))
            sbp.call(shlex.split(cmd))
            
            # save detections on training set to "i-1" annotations
            detections = os.path.join(output_dir, rec_train_cfg, "training_round_{:d}_{:d}/test/smd_{:s}_train_2_seg/generalized_rcnn/detections.pkl".format(i,j,subset))
            log_path = os.path.join(output_dir, rec_train_cfg, "training_round_{:d}_{:d}".format(i,j))
            cmd = "python /input/save_annotations.py {:s} {:s} {:s} {:s}".format(detections, original_gt[idx + offset], annot_target, log_path)        
            sbp.call(shlex.split(cmd))
            
            # copy new GT to source folder
            cmd = "cp {:s} {:s}".format(annot_target, initial_gt)        
            sbp.call(shlex.split(cmd))
        
        # set configurations for next round
        train_weights = os.path.join(output_dir, rec_train_cfg, "training_round_{:d}_{:d}/".format(i,j), "train/smd_{:s}_train_2_seg/generalized_rcnn/model_final.pkl".format(subset))

# reset initial GT to original segmentation GT
cmd = "cp {:s} {:s}".format(original_gt[0], initial_gt)
sbp.call(shlex.split(cmd)) 
