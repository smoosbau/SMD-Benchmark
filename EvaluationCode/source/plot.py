import itertools
import pylab as pl
import os
import matplotlib.pyplot as plt
import numpy as np


def split_eval_string(tp, fp, fn, prec, rec):

    try:
        prec = float((prec.split(':')[-1]).split('\n')[0])
        rec = float((rec.split(':')[-1]).split('\n')[0])
    except ValueError:
        return 0, 0, 0, -1, -1

    tp = int((tp.split(':')[-1]).split('\n')[0])
    fp = int((fp.split(':')[-1]).split('\n')[0])
    fn = int((fn.split(':')[-1]).split('\n')[0])

    return tp, fp, fn, prec, rec


def read_eval(path):

    with open(path, 'r') as f:

        tp = f.next()
        fp = f.next()
        fn = f.next()
        prec = f.next()
        rec = f.next()

    return split_eval_string(tp, fp, fn, prec, rec)


def read_eval_folder(path):

    files = os.listdir(path)
    file_paths = []
    for f in files:
        if f.endswith('.txt'):
            file_paths.append(path + f)

    tp_l = []
    fp_l = []
    fn_l = []
    prec_l = []
    rec_l = []
    for f in file_paths:
        tp, fp, fn, prec, rec = read_eval(f)
        tp_l.append(tp)
        fp_l.append(fp)
        fn_l.append(fn)
        prec_l.append(prec)
        rec_l.append(rec)

    return tp_l, fp_l, fn_l, prec_l, rec_l


def sort_prec_rec(prec, rec):

    prec = [val for val in prec if val > 0]
    rec = [val for val in rec if val > 0]

    rank = np.argsort(rec)
    rec = np.sort(rec)
    prec = [prec[i] for i in rank]

    return prec, rec


def plot_prec_rec(rec, prec, label, linestyle='-', color=(1, 1, 1)):
    line, = plt.plot(rec, prec, linestyle=linestyle, color=color, label=label)
    plt.axis([-0.05, 1.05, -0.05, 1.05])
    plt.grid(True, which='both')
    plt.xlabel('recall', fontdict=None, labelpad=None)
    plt.ylabel('precision', fontdict=None, labelpad=None)

    return line


def prec_av(prec, rec):
    tup = []
    for i in range(len(prec)):
        tup.append((prec[i], rec[i]))

    tup = sorted(tup)

    prec_av = []
    rec_av = []
    rec_old = tup[0][0]
    prec_sum = tup[0][1]
    num = 1
    for i in range(1, len(tup)):
        if tup[i][0] == rec_old:
            prec_sum += tup[i][1]
            num += 1
        else:
            prec_av.append(prec_sum/num)
            rec_av.append(rec_old)
            rec_old = tup[i][0]
            prec_sum = tup[i][1]
            num = 1

    return prec_av, rec_av


def config(suptitle, ticks=[round(float(i) / 20, 2) for i in range(21)], title=None):
    fig = plt.gcf()
    fig.set_size_inches(17.68, 12.8)
    plt.legend(loc='upper right', frameon=False)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.title(title)
    plt.suptitle(suptitle, fontsize=16)

    return None


def plot_loss_epoch(loss, epoch):
    line, = plt.plot(epoch, loss)
    plt.axis([0, max(epoch)+1, 0, max(loss)+1])
    plt.grid(True, which='both')
    plt.xlabel('epoch', fontdict=None, labelpad=None)
    plt.ylabel('loss', fontdict=None, labelpad=None)

    return line


def show():
    plt.show()
    return


def savefig(title):
    plt.savefig(title, transparent=True, dpi=600)
    return


def clearfig():
    plt.clf()
    return


def plot_prop_rec(rec_03, rec_05, path="."):
    num_props = range(1, len(rec_03) + 1)

    plt.subplot(211)
    plt.plot(num_props, rec_03, "m-")

    plt.subplot(212)
    plt.plot(num_props, rec_05, "g-")

    # plt.show()
    plt.savefig(os.path.join(path, "rec_prop.svg"))

    return 0


def plot_confusion_matrix(cm, classes, name, normalize=False, title='Confusion matrix', cmap=plt.cm.BuGn):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normed = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm_normed, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() # / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    #plt.tight_layout()
    #plt.show()
    plt.gcf().subplots_adjust(bottom=0.5)
    plt.savefig(name)
    plt.clf()

