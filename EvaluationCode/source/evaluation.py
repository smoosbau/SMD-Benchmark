
#################################################################################################
# This script will be used to provide all necessary evaluation procedures during my master's thesis
#################################################################################################
import numpy as np
import ast
import cPickle as pkl
import sys


_smd_id_by_cat = {"Ferry": 1,
                  "Buoy": 2,
                  "Vessel/ship": 3,
                  "Speed boat": 4,
                  "Boat": 5,
                  "Kayak": 6,
                  "Sail boat": 7,
                  "Swimming person": 8,
                  "Flying bird/plane": 9,
                  "Other": 10
                  }


def iou(bboxA, bboxB):
    # type: (list, list) -> float
    """
    return Intersection over Union of two bounding boxes
    :param bboxA: [x_upper_left, y_upper_left, width, height]
    :param bboxB: [x_upper_left, y_upper_left, width, height]
    :rtype: float
    """
    
    # calculate area of bboxA and bboxB
    A = bboxA[2] * bboxA[3]
    B = bboxB[2] * bboxB[3]

    # calculate left and right boundaries of both boxes
    xA_0 = bboxA[0]
    xA_1 = bboxA[0] + bboxA[2]

    xB_0 = bboxB[0]
    xB_1 = bboxB[0] + bboxB[2]

    # calculate width of intersection
    w_i = min(xA_1, xB_1) - max(xA_0, xB_0)

    # if intersection has none or negative width return 0 for IoU
    if w_i <= 0:
        return 0, A, B

    # calculate lower and upper boundaries of both boxes
    yA_0 = bboxA[1]
    yA_1 = bboxA[1] + bboxA[3]

    yB_0 = bboxB[1]
    yB_1 = bboxB[1] + bboxB[3]

    # calculate height of intersection
    h_i = min(yA_1, yB_1) - max(yA_0, yB_0)

    # if intersection has none or negative height return 0 for IoU
    if h_i <= 0:
        return 0, A, B

    # calculate intersection area
    intersec_A_B = w_i * h_i

    # return IoU
    return float(intersec_A_B) / (A + B - intersec_A_B), A, B


def dect_is(bb_in, gt, thrs, conf_mat=np.array([]), correct_pred=0):

    """
    :param bb: [[[x_upper_left, y_upper_left, width, height], label],...]
    :param gt: [[x_upper_left, y_upper_left, width, height, label],...]
    :param thrs: threshold for IoU true positive
    :return: [[IoU_bb0_gt0,IoU_bb0_gt1,...],...,[...,IoU_bbn_gtn]]
    """

    if not bb_in and not gt:
        # print("A")
        return 0, 0, 0, 0
    elif not bb_in:
        # print("B")
        return 0, 0, len(gt), 0
    elif not gt:
        # print("C")
        return 0, len(bb_in), 0, 0

    calc_conf_mat = True if len(conf_mat) > 0 else False

    bb = []
    for i in range(len(bb_in)):
        bb.append(bb_in[i])

    index = 0
    tp = 0
    inter_old = 0

    sum_iou = 0

    for grtr in range(len(gt)):
        for det in range(len(bb)):
            inter, A, B = iou(bb[det][0], gt[grtr])
            if (inter > thrs) and (inter > inter_old):
                inter_old = inter
                index = det

        if inter_old > 0:
            tp += 1
            if calc_conf_mat:
                pred = bb[index][1]
                true = _smd_id_by_cat[gt[grtr][-1]]
                conf_mat[pred, true] += 1
                # save num of correct predictions
                # correct_pred += 1 if pred == true else 0
                if pred == true:
                    correct_pred += 1
                    print(correct_pred)

            bb.remove(bb[index])
            sum_iou += inter_old

        inter_old = 0
        index = 0

    if len(bb) > 0:
        fp = len(bb)
    else:
        fp = 0

    if (len(gt) - tp) > 0:
        fn = len(gt) - tp
    else:
        fn = 0

    return tp, fp, fn, sum_iou


def precision(tp, fp):
    if tp == 0:
        return 0
    else:
        return float(tp)/(tp + fp)


def recall(tp, fn):
    if tp == 0:
        return 0
    else:
        return float(tp)/(tp + fn)


def f_measure(prec, rec):
    if (prec == 0) or (rec == 0):
        return 0
    else:
        return 2 * prec * rec / (prec + rec)


def classification_accuracy(correct_preds, total_preds):
    if not total_preds == 0:
        return float(correct_preds)/total_preds
    else:
        return np.nan


def classification_error(correct_preds, total_preds):
    return 1 - classification_accuracy(correct_preds, total_preds)


def eval_with_conf(boxes_conf, gt, thrs, conf, calc_conf_mat=False, conf_mat=np.array([[], []]), correct_pred=[0, 0]):
    tp = []
    fp = []
    fn = []
    m_iou = []
    max_conf = [box[0][-1] for box in boxes_conf]

    if not max_conf:
        max_conf = 0
    else:
        max_conf = max(max_conf)

    for idx, c in enumerate(conf):
        # copy boxes for further manipulations
        boxes = [boxes_conf[j] for j in range(len(boxes_conf))]

        # loop through boxes
        killed = 0
        for i in range(len(boxes)):
            if boxes[i-killed][0][-1] < c:
                boxes.remove(boxes[i-killed])
                killed += 1

        if calc_conf_mat and (c == 0.5 or c == 0.75):
            idx = 0 if c == 0.5 else 1
            sol = dect_is(boxes, gt, thrs, conf_mat=conf_mat[idx], correct_pred=correct_pred[idx])
        else:
            sol = dect_is(boxes, gt, thrs)

        tp.append(sol[0])
        fp.append(sol[1])
        fn.append(sol[2])
        m_iou.append(sol[3])

    return tp, fp, fn, m_iou, max_conf


def determine_p1(prec_rec):
    if (prec_rec[0].count(0) == 1) and (prec_rec[1].count(0) == 1):
        for i in range(len(prec_rec[0])):
            if (prec_rec[0][i] == 0) and (prec_rec[1][i] == 0):
                # remove point (0/0)
                del prec_rec[0][i]
                del prec_rec[1][i]

        prec_min = prec_rec[0].index(min(prec_rec[0]))

        prec_rec[0].append(0)
        prec_rec[1].append(prec_rec[1][prec_min])

    return


def sort_n_props_by_score(proposals, n):
    boxes = proposals['boxes']
    scores = proposals['scores']

    boxes_flat = np.reshape(boxes, (-1, 4))
    scores_flat = np.reshape(scores, (-1))
    # convert boxes from [x1 y1 x2 y2] to [x y w h]
    conv_boxes = []
    for box in boxes_flat:
        conv_boxes.append([box[0], box[1], box[2] - box[0], box[3] - box[1]])

    # sort all proposals if n is less zero
    if n < 0:
        n = len(scores_flat)

    elif n > len(scores_flat):
        n = len(scores_flat)

    max_inds = np.argsort(scores_flat)
    max_inds = max_inds[:n]

    output = []
    for i in max_inds:
        cache = [conv_boxes[i], scores_flat[i]]
        output.append(cache)

    return output


def n_prop_vs_rec(sorted_props, gt, n=100):
    """
    sort n proposals by their score.
    returns #proposals vs. recall -> [[a], [b], ..., [n]]
    a = recall for 1 proposal
    b = recall for 2 proposals
        .
        .
        .
    n = recall for (n+1) proposals

    :param sorted_props: [[prop_0], [prop_1], ... , [prop_N]]
    :param gt: ground truth
    :param n: num of props to sort for evaluation (default = 100)
    :return tp(thrs=0.3), tp(thrs=0.5), fn(thrs=0.3), fn(thrs=0.5)
     """
    sorted_props = [[[prop[0], prop[1], prop[2] - prop[0], prop[3] - prop[1]]] for prop in sorted_props[:n]]
    # sys.stdout.write("\rn = {:d} len(props) = {:d}".format(n, len(sorted_props)))

    # for large n's this should definitely be paralleled
    # print(sorted_props[:n])
    tp_03, _, fn_03, _ = dect_is(sorted_props, gt, 0.3)
    tp_05, _, fn_05, _ = dect_is(sorted_props, gt, 0.5)

    return tp_03, tp_05, fn_03, fn_05
