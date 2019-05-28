"""functions for evaluating smd dataset"""

from os.path import dirname
import sys
sys.path.append('/EvaluationCode/source/')
import evaluation
import plot as eval_plot
import cv2
import json
import os
import numpy as np
import cPickle as pkl
import detectron.utils.boxes as box_utils
from detectron.core.config import cfg


def evaluate_boxes(dataset, all_boxes, output_dir, num_classes=11):
    """
    :param dataset: dataset in coco format
    :param all_boxes: detection boxes
    :param output_dir: dir to save eval.json
    :return: res_dict = {thrs_0: {'TP': x0, 'FP': y0, 'FN': z0}, ... , thrs_N: {'TP': xN, 'FP': yN, 'FN': zN}}
    """
    
    res = []
    
    gt = get_gt(dataset)
    
    imgs = dataset.COCO.imgs
    
    num_imgs = len(imgs)
    
    thrs = [0.5, 0.3]     
         
    ev_json = EvalSave(thrs)
    ev_json._set_dataset_name(dataset.name)
        
    res_dict = {}
    [res_dict.update({val: {'TP': 0, 'FP': 0, 'FN': 0}}) for val in thrs]
    
    l_limit = 2500
    u_limit = 50 * 10**3
        
    # create confusion matrix only for 10c datasets
    calc_cfn_matrix = True if "10c" in dataset.name else False
    
    for t in thrs:  
        # intialize caches for different tp, fp, fn and m_iou values
        tp_sum = [0] * int((1/ev_json._conf_stride + 1))
        fp_sum = [0] * int((1/ev_json._conf_stride + 1))
        fn_sum = [0] * int((1/ev_json._conf_stride + 1))        
        m_iou_sum = [0] * int((1/ev_json._conf_stride + 1))

        tp_sum_low = [0] * int((1/ev_json._conf_stride + 1))
        fp_sum_low = [0] * int((1/ev_json._conf_stride + 1))
        fn_sum_low = [0] * int((1/ev_json._conf_stride + 1))        
        m_iou_sum_low = [0] * int((1/ev_json._conf_stride + 1))
        
        tp_sum_mid = [0] * int((1/ev_json._conf_stride + 1))
        fp_sum_mid = [0] * int((1/ev_json._conf_stride + 1))
        fn_sum_mid = [0] * int((1/ev_json._conf_stride + 1))        
        m_iou_sum_mid = [0] * int((1/ev_json._conf_stride + 1))

        tp_sum_high = [0] * int((1/ev_json._conf_stride + 1))
        fp_sum_high = [0] * int((1/ev_json._conf_stride + 1))
        fn_sum_high = [0] * int((1/ev_json._conf_stride + 1))        
        m_iou_sum_high = [0] * int((1/ev_json._conf_stride + 1))
        
        conf_mat = [np.zeros((num_classes, num_classes)).astype(np.uint32),
                     np.zeros((num_classes, num_classes)).astype(np.uint32)]
        
        classification_error = [np.array([0]), np.array([0])]
        
        for im in range(num_imgs):
            sys.stdout.write("\rimage {:d} of {:d} - thrs = {:0.1f}".format(im+1, num_imgs, t))
            # sort out empty boxes
            n_e_boxes = []
            for idx, b in enumerate(all_boxes[1:]):
                if b[im].any():                    
                    for i in range(len(b[im])):       
                        n_e_boxes.append([b[im][i].tolist(), idx+1])
            
            boxes = get_image_boxes(dataset, all_boxes, im)
            
            for box in n_e_boxes:
                box[0][2] = box[0][2] - box[0][0]
                box[0][3] = box[0][3] - box[0][1]
                                    
            r_end =  int((1 + ev_json._conf_stride) / ev_json._conf_stride)
            conf = [c*ev_json._conf_stride for c in range(int(1/ev_json._conf_stride) + 1)]
                        
            # as coco forces you to provide a label, empty images are detected by bounding boxes containing only a label
            # validation case: empty frames were left emtpy and didn't get a label
            if not gt[im]:
                gt_ev = []                
            else:
                # test case: there's no bounding box but a label to tell the network that there's no object
                if len(gt[im][0]) == 1:
                    gt_ev = []
                else:
                    gt_ev = gt[im]
                    
                
            # prepare for size dependant evaluation
            gt_low = []
            gt_mid = []
            gt_high = []
            for g in gt_ev:
                if g[2]*g[3] <= l_limit:
                    gt_low.append(g)
                elif g[2]*g[3] > u_limit:
                    gt_high.append(g)
                else:
                    gt_mid.append(g)
            
            # TODO(eomoos): change cache arrays type to numpy array so you can easily add cache to target array!
            # eval complete dataset
            if calc_cfn_matrix:
                tp_c, fp_c, fn_c, m_iou, max_conf = evaluation.eval_with_conf(n_e_boxes, gt_ev, t, conf, calc_conf_mat=calc_cfn_matrix, conf_mat=conf_mat, correct_pred=classification_error)
            else:
                tp_c, fp_c, fn_c, m_iou, max_conf = evaluation.eval_with_conf(n_e_boxes, gt_ev, t, conf)
            # eval only with tiny objects - A <= 2500
            tp_c_low, fp_c_low, fn_c_low, m_iou_low, _ = evaluation.eval_with_conf(n_e_boxes, gt_low, t, conf)
            # eval only with medium size objects - 2500 < A < 50,000
            tp_c_mid, fp_c_mid, fn_c_mid, m_iou_mid, _ = evaluation.eval_with_conf(n_e_boxes, gt_mid, t, conf)
            # eval only with huge objects - A >= 50000
            tp_c_high, fp_c_high, fn_c_high, m_iou_high, _ = evaluation.eval_with_conf(n_e_boxes, gt_high, t, conf)
            
            # calculate sums of different size tp(conf), fp(conf), fn(conf), m_iou(conf)
            tp_sum = [tp_sum[i] + tp_c[i] for i in range(len(tp_c))]
            fp_sum = [fp_sum[i] + fp_c[i] for i in range(len(fp_c))]
            fn_sum = [fn_sum[i] + fn_c[i] for i in range(len(fn_c))]
            m_iou_sum = [m_iou_sum[i] + m_iou[i] for i in range(len(m_iou))]

            tp_sum_low = [tp_sum_low[i] + tp_c_low[i] for i in range(len(tp_c_low))]
            fn_sum_low = [fn_sum_low[i] + fn_c_low[i] for i in range(len(fn_c_low))]
            m_iou_sum_low = [m_iou_sum_low[i] + m_iou_low[i] for i in range(len(m_iou_low))]
            
            tp_sum_mid = [tp_sum_mid[i] + tp_c_mid[i] for i in range(len(tp_c_mid))]
            fn_sum_mid = [fn_sum_mid[i] + fn_c_mid[i] for i in range(len(fn_c_mid))]
            m_iou_sum_mid = [m_iou_sum_mid[i] + m_iou_mid[i] for i in range(len(m_iou_mid))]
            
            tp_sum_high = [tp_sum_high[i] + tp_c_high[i] for i in range(len(tp_c_high))]
            fn_sum_high = [fn_sum_high[i] + fn_c_high[i] for i in range(len(fn_c_high))]
            m_iou_sum_high = [m_iou_sum_high[i] + m_iou_high[i] for i in range(len(m_iou_high))]
            
        sys.stdout.write("\n")                
        add_to_res_dict(res_dict, tp_sum[0], fp_sum[0], fn_sum[0], t)
        
        
        # calc mean iou's for different evaluation scales             
        for i in range(len(tp_sum)):        
            if tp_sum[i] > 0:
                ev_json._eval['m_iou'][t].append(m_iou_sum[i]/tp_sum[i]) 
            else:
                ev_json._eval['m_iou'][t].append(0)
        
        for i in range(len(tp_sum_low)):        
            if tp_sum_low[i] > 0:
                ev_json._eval['m_iou_low'][t].append(m_iou_sum_low[i]/tp_sum_low[i]) 
            else:
                ev_json._eval['m_iou_low'][t].append(0)
        
        for i in range(len(tp_sum_mid)):        
            if tp_sum_mid[i] > 0:
                ev_json._eval['m_iou_mid'][t].append(m_iou_sum_mid[i]/tp_sum_mid[i]) 
            else:
                ev_json._eval['m_iou_mid'][t].append(0)
                
        for i in range(len(tp_sum_high)):        
            if tp_sum_high[i] > 0:
                ev_json._eval['m_iou_high'][t].append(m_iou_sum_high[i]/tp_sum_high[i]) 
            else:
                ev_json._eval['m_iou_high'][t].append(0)
        
        for i in range(len(tp_sum)):            
            ev_json._eval['prec'][t].append(evaluation.precision(tp_sum[i], fp_sum[i]))
            ev_json._eval['rec'][t].append(evaluation.recall(tp_sum[i], fn_sum[i]))
            
            ev_json._eval['rec_low'][t].append(evaluation.recall(tp_sum_low[i], fn_sum_low[i]))

            ev_json._eval['rec_mid'][t].append(evaluation.recall(tp_sum_mid[i], fn_sum_mid[i]))

            ev_json._eval['rec_high'][t].append(evaluation.recall(tp_sum_high[i], fn_sum_high[i]))
            
        flat_conf_mat = conf_mat[0].flatten()
        num_tp_dets = sum(flat_conf_mat.tolist())
        cls_acc = evaluation.classification_accuracy(classification_error[0][0], num_tp_dets)
        print(cls_acc)
        
        ev_json._eval['conf_mat'][t].append([flat_conf_mat.tolist(), num_classes])
        ev_json._eval['classification_error'][t].append(
            [cls_acc])
        
        flat_conf_mat = conf_mat[1].flatten()
        num_tp_dets = sum(flat_conf_mat.tolist())
        cls_acc = evaluation.classification_accuracy(classification_error[0][0], num_tp_dets)
        print(cls_acc)
        
        ev_json._eval['conf_mat'][t].append([flat_conf_mat.tolist(), num_classes])
        ev_json._eval['classification_error'][t].append(
            [cls_acc])
            
    ev_json._max_conf = max_conf
    
    ev_json._save_as_json(output_dir)
    
    # print(np.array(ev_json._eval['conf_mat'][0.3][0][0]).reshape((num_classes, num_classes)))
    
    return res_dict


def get_gt(dataset):
    gt = []
    for i in range(len(dataset.COCO.imgs)):
        gt.append([])
        
    for index, _ in enumerate(dataset.COCO.anns):  
        im_id = dataset.COCO.anns[index]['image_id']
        bb = dataset.COCO.anns[index]['bbox']
        label = dataset.COCO.cats[dataset.COCO.anns[index]['category_id']]['name']
        bb.append(label)
        gt[im_id].append(bb)
        
    return gt


def get_image_boxes(dataset, all_boxes, im):
    boxes = []
    for i in range(1, len(all_boxes)):
        im_boxes = all_boxes[i][im].tolist()
        for box in im_boxes:
            label = i
            boxes.append([box, label])
        
    return boxes


def add_to_res_dict(res_dict, tp, fp, fn, thrs):
    res_dict[thrs]['TP'] = res_dict[thrs]['TP'] + tp
    res_dict[thrs]['FP'] = res_dict[thrs]['FP'] + fp
    res_dict[thrs]['FN'] = res_dict[thrs]['FN'] + fn
            
    return 0


def sum_2d_dicts(dict1, dict2):
    """
    :param dict1: {thrs_0: {'TP': x0, 'FP': y0, 'FN': z0}, ... , thrs_N: {'TP': xN, 'FP': yN, 'FN': zN}}
    :param dict2: {thrs_0: {'TP': x0, 'FP': y0, 'FN': z0}, ... , thrs_N: {'TP': xN, 'FP': yN, 'FN': zN}}
    :return: dict2 = dict1 + dict2
    """
    
    for d_one in dict1:
        for d_two in dict1[d_one]:
            dict2[d_one][d_two] = dict2[d_one][d_two] + dict1[d_one][d_two]
    
    return dict2


def draw_histos(vis=False, filepath='/output/evalSave/'):
    print(filepath)
    if not os.path.exists(filepath):
        print('path created')
        os.makedirs(filepath)
    
    with open(os.path.join(filepath, 'eval.json'), 'r') as j:
        data = json.load(j)
        
    # build precision-recall curves
    prec_03, rec_03 = eval_plot.sort_prec_rec(data['evaluation']['prec']['0.3'], data['evaluation']['rec']['0.3'])
    eval_plot.plot_prec_rec(rec_03, prec_03, 'Thrs: 0.3', color=(1, 0, 1))
        
    prec_05, rec_05 = eval_plot.sort_prec_rec(data['evaluation']['prec']['0.5'], data['evaluation']['rec']['0.5'])
    eval_plot.plot_prec_rec(rec_05, prec_05, 'Thrs: 0.5', color=(0, 1, 0))
    
    # build smoothed precision-recall curves
    # smooth PR-curve as described in http://cs229.stanford.edu/section/evaluation_metrics.pdf#
    smoothed_prec_03 = [max(prec_03[idx:]) for idx, _ in enumerate(prec_03)]
    smoothed_prec_05 = [max(prec_05[idx:]) for idx, _ in enumerate(prec_05)]
    
    eval_plot.plot_prec_rec(rec_03, smoothed_prec_03, 'Thrs_smoothed: 0.3', color=(1, 0, 1), linestyle="--")
    eval_plot.plot_prec_rec(rec_05, smoothed_prec_05, 'Thrs_smoothed: 0.5', color=(0, 1, 0), linestyle="--")
    
    #auc_03 = np.trapz(smoothed_prec_03, rec_03)
    #auc_05 = np.trapz(smoothed_prec_05, rec_05)

    auc_03 = np.trapz(prec_03, rec_03)
    auc_05 = np.trapz(prec_05, rec_05)
    
    title = "conf. stride: {:0.2f}, max. conf.: {:0.4f}, AUC_03: {:0.2f}, AUC_05: {:0.2f}".\
                                                format(data['confidence stride'], data['max_conf'], auc_03, auc_05)

    eval_plot.config(data['name'], title=title)
 
    
    eval_plot.savefig(os.path.join(filepath, (data['name'] + '.svg')))
    if vis:
        eval_plot.show()
    else:
        eval_plot.clearfig()

    # build mean iou - recall curves        
    postfix = ['', '_low', '_mid', '_high']
    
    for p in postfix:        
        eval_plot.plot_prec_rec(data['evaluation']['rec' + p]['0.3'], data['evaluation']['m_iou' + p]['0.3'], 'halt au', color=(1, 0, 1))
                
        eval_plot.plot_prec_rec(data['evaluation']['rec' + p]['0.5'], data['evaluation']['m_iou' + p]['0.5'], 'halt au', color=(0, 1, 0))
        
        title = "conf. stride: {:0.2f}, max. conf.: {:0.4f},\nAUC_03: {:0.2f}, AUC_05: {:0.2f}".\
                                                    format(data['confidence stride'], data['max_conf'], auc_03, auc_05)
               
        eval_plot.config(data['name'], title=title)
        
        eval_plot.savefig(os.path.join(filepath, 'm_iou' + p + '.svg'))
        if vis:
            eval_plot.show()
        else:
            eval_plot.clearfig()
    
    classes = ["Background", "Ferry", "Buoy", "Vessel/ship", "Speed boat", "Boat", "Kayak", "Sail boat", "Swimming person", "Flying bird/plane", "Other"]        
    
    num_classes = data['evaluation']['conf_mat']["0.3"][0][1]
    cm = np.array(data['evaluation']['conf_mat']["0.3"][0][0]).reshape((num_classes, num_classes))
    name = os.path.join(filepath, "confusion_matrix_03_05.svg")
    eval_plot.plot_confusion_matrix(cm, classes, name)
    
    cm = np.array(data['evaluation']['conf_mat']["0.3"][1][0]).reshape((num_classes, num_classes))
    name = os.path.join(filepath, "confusion_matrix_03_075.svg")
    eval_plot.plot_confusion_matrix(cm, classes, name)
    
    cm = np.array(data['evaluation']['conf_mat']["0.5"][0][0]).reshape((num_classes, num_classes))
    name = os.path.join(filepath, "confusion_matrix_05_05.svg")
    eval_plot.plot_confusion_matrix(cm, classes, name)
    
    cm = np.array(data['evaluation']['conf_mat']["0.5"][1][0]).reshape((num_classes, num_classes))
    name = os.path.join(filepath, "confusion_matrix_05_075.svg")
    eval_plot.plot_confusion_matrix(cm, classes, name)

def print_dataset(dataset):
    
    print('\nname\n' + str(dataset.name) + '\n')
    print('image_directory\n' + str(dataset.image_directory) + '\n')
    print('image_prefix\n' + str(dataset.image_prefix) + '\n')
    print('COCO.dataset\n' + str(dataset.COCO.dataset) + '\n')
    print('COCO.anns\n' + str(dataset.COCO.anns) + '\n')
    print('COCO.cats\n' + str(dataset.COCO.cats) + '\n')
    print('COCO.imgs\n' + str(dataset.COCO.imgs) + '\n')
    print('COCO.imgToAnns\n' + str(dataset.COCO.imgToAnns) + '\n')
    print('COCO.catToImgs\n' + str(dataset.COCO.catToImgs) + '\n')
    print('debug_timer\n' + str('skip it Stan!') + '\n')
    print('cat_to_id_map\n' + str(dataset.category_to_id_map) + '\n')
    print('classes\n' + str(dataset.classes) + '\n')
    print('num_classes\n' + str(dataset.num_classes) + '\n')
    print('json_category_id_to_contiguous_id\n' + str(dataset.json_category_id_to_contiguous_id) + '\n')
    print('contiguous_category_id_to_json_id\n' + str(dataset.contiguous_category_id_to_json_id) + '\n')
    
    return 0


def load_proposal_file(proposoal_file, output_dir):
    with open(os.path.join(output_dir, proposoal_file), "r") as f:
        rois = pkl.load(f)

    return rois


def prep_proposal_file(proposal_file, output_dir, min_proposal_size=2, top_k=-1):
    rois = load_proposal_file(proposal_file, output_dir)
    boxes = rois['boxes']
    scores = rois['scores']
    boxes_out = []
    for img,img_boxes in enumerate(boxes):
        img_scores = scores[img]
        img_boxes = box_utils.clip_boxes_to_image(img_boxes, 1080, 1920)  # TODO: remove this dirty hack!
        keep = box_utils.unique_boxes(img_boxes)
        img_boxes =img_boxes[keep, :]
        img_scores = img_scores[keep]
        keep = box_utils.filter_small_boxes(img_boxes, min_proposal_size)
        img_boxes =img_boxes[keep, :]
        img_scores = img_scores[keep]
        
        if top_k > 0:           
            img_boxes =img_boxes[:top_k]
            img_scores = img_scores[:top_k]
        
        box_cache = []
        for sc, entry in enumerate(img_boxes):
            entry = np.append(entry, img_scores[sc])
            box_cache.append(entry)
        
        boxes_out.append(np.array(box_cache))
    
    return boxes_out


def plot_rec_prop_curve(proposals, scores, dataset, path, n=15):
    # get gt from dataset
    gt = get_gt(dataset)

    rec_03 = []
    rec_05 = []

    for num in range(1, n + 1):
        tp_03 = 0; tp_05 = 0; fn_03 = 0; fn_05 = 0

        too_less_props = 0

        for idx, boxes in enumerate([proposals[0]]):
            if len(boxes) >= num:
                tp_03_c, tp_05_c, fn_03_c, fn_05_c = evaluation.n_prop_vs_rec(boxes, gt[idx], n=num)

            else:
                tp_03_c, tp_05_c, fn_03_c, fn_05_c = evaluation.n_prop_vs_rec(boxes, gt[idx], n=len(boxes))
                too_less_props = too_less_props + 1

            tp_03 = tp_03 + tp_03_c
            tp_05 = tp_05 + tp_05_c
            fn_03 = fn_03 + fn_03_c
            fn_05 = fn_05 + fn_05_c
        
        ## for debugging
        #print(num, too_less_props)
        #print(tp_03, tp_05, fn_03, fn_05)
        
        rec_03.append(evaluation.recall(tp_03, fn_03))
        rec_05.append(evaluation.recall(tp_05, fn_05))

    eval_plot.plot_prop_rec(rec_03, rec_05, path)

    return 0


class EvalSave:
    _dataset_name = ''
    _max_conf = "N.A."
    _conf_stride = 0
    _eval = {'prec': {}, 'rec': {}, 'prec_low': {}, 'rec_low': {}, 'prec_mid': {}, 'rec_mid': {}, 'prec_high': {}, 'rec_high': {},
             'm_iou': {}, 'm_iou_low': {}, 'm_iou_mid': {},'m_iou_high': {}, 'conf_mat': {}, 'classification_error': {}}
    
    def __init__(self, thrs, conf_stride=0.01):
        for val in thrs:
            self._eval['prec'].update({val: []})
            self._eval['rec'].update({val: []})
            self._eval['prec_low'].update({val: []})
            self._eval['rec_low'].update({val: []})
            self._eval['prec_mid'].update({val: []})
            self._eval['rec_mid'].update({val: []})
            self._eval['prec_high'].update({val: []})
            self._eval['rec_high'].update({val: []})            
            self._eval['m_iou'].update({val: []})
            self._eval['m_iou_low'].update({val: []})
            self._eval['m_iou_mid'].update({val: []})
            self._eval['m_iou_high'].update({val: []})
            self._eval['conf_mat'].update({val: []})
            self._eval['classification_error'].update({val: []})
            self._conf_stride = conf_stride
            self._max_conf = "N.A."
        self._set_dataset_name()
        
    def _set_dataset_name(self, name='unknown name'):
        self._dataset_name = name
        
        return 0
        
    def _update_eval(prec, rec, prec_low, rec_low, prec_mid, rec_mid, prec_high, rec_high, conf, max_conf="N.A."):
        self._eval['prec'][conf] = prec
        self._eval['rec'][conf] = rec
        self._eval['prec_low'][conf] = prec_low
        self._eval['rec_low'][conf] = rec_low
        self._eval['prec_mid'][conf] = prec_mid
        self._eval['rec_mid'][conf] = rec_mid
        self._eval['prec_high'][conf] = prec_high
        self._eval['rec_high'][conf] = rec_high
        self._max_conf = max_conf
        
        
        return 0
    
    def _save_as_json(self, filepath='/output/evalSave/'):
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        with open(os.path.join(filepath,'eval.json'), 'w') as j:
            data = json.dumps({'name': self._dataset_name, 'confidence stride': self._conf_stride, 'evaluation': self._eval, "max_conf": self._max_conf})
            j.write(data)
    
        return 0
        
        
    
