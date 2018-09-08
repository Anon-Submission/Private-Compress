'''
Private Knowledge Distillation
'''

import tensorflow as tf
import numpy as np
import os, sys
import time
import argparse
from sklearn.cluster import MiniBatchKMeans
from sklearn import preprocessing

from keras.utils import *
from keras import backend as K

from teacher_model import TeacherCifar
from student_model import StudentCifar


def parse_args():
    parser = argparse.ArgumentParser(description='private fitnet model')
    parser.add_argument('--sel_method', dest='sel_method', default='margin',
                        help='method for query sample selection, random/gibbs/margin/diverse/kcenter', type=str)
    parser.add_argument('--sel_rate', dest='sel_rate', default=.5,
                        help='query samples selection rate', type=float)
    parser.add_argument('--lr', dest='lr', default=1e-3, help='learning rate', type=float)
    parser.add_argument('--self_epoch', dest='self_epoch', default=8, help='epoch for self learning', type=int)
    parser.add_argument('--hint_epoch', dest='hint_epoch', default=50, help='epoch for hint learning', type=int)
    parser.add_argument('--kd_epoch', dest='kd_epoch', default=8, help='epoch for kd learning', type=int)
    parser.add_argument('--dropout', dest='dropout', default=0.5, help="dropout ratio", type=float)
    parser.add_argument('--mask', dest='mask', default=0.9, help='mask some training data from the student', type=float)
    parser.add_argument('--noisy_sigma', dest='sigma', default=1., help="noisy sigma", type=float)
    parser.add_argument('--tau',dest='tau', default=3., help="temperature of softmax", type=float)
    parser.add_argument('--add_noise', dest='add_noise', default='Y',
                        help="whether to add noise to protect privacy", type=str)
    parser.add_argument('--batchsize', dest='batchsize', default=512, type=int)
    parser.add_argument('--iterations', dest='iterations', default=4, type=int)
    parser.add_argument('--gpu', dest='gpu', default='0', type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args, parser

global args, parser
args, parser = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# define the log file that receives your log info
name = 'train_fit_cifar_'+args.sel_method+args.gpu+'.log'
log_file = open(name, "w")
# redirect print output to log file
sys.stdout = log_file

dim=32
n_classes=10
# placeholders
global x,y,l2norm_bound,lr

x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='inputs')
y = tf.placeholder(tf.int32, shape=(None, 10), name='labels')
l2norm_bound = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)


def conv(_input, W, b, strides=1, padding='SAME'):
    output = tf.nn.conv2d(_input, W, strides=[1, strides, strides, 1], padding=padding)
    output = tf.nn.bias_add(output, b)
    return output


# Reading cifar dataset
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def read_cifar10():
    # Directory of cifar-10
    path = '～/cifar-10-batches-py'
    batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    train_data = np.uint8(np.zeros((50000,32,32,3)))
    train_label = np.int16(np.zeros((50000,1)))

    N = 10000
    for i in range(len(batches)):
        b = batches[i]
        temp = unpickle(os.path.join(path, b))
        for j in range(N):
            key = bytes("data", 'utf-8')
            train_data[N * i + j][:, :, 2] = np.reshape(temp[key][j][2048:], (32, 32))
            train_data[N * i + j][:, :, 1] = np.reshape(temp[key][j][1024:2048], (32, 32))
            train_data[N * i + j][:, :, 0] = np.reshape(temp[key][j][:1024], (32, 32))
            key = bytes("labels", 'utf-8')
            train_label[N * i + j] = temp[key][j]
    train_label = np_utils.to_categorical(np.array(train_label), 10)

    # mask part of the training data randomly
    np.random.seed(0)
    num = np.array(range(train_data.shape[0]))
    sel = np.random.choice(num, size=int(args.mask * train_data.shape[0]), replace=False)
    train_data = train_data[sel]
    train_label = train_label[sel]

    batches = ['test_batch']
    test_data = np.float32(np.zeros((10000, 32, 32, 3)))
    test_label = np.zeros((10000, 1))

    N = 10000
    for i in range(len(batches)):
        b = batches[i]
        temp = unpickle(os.path.join(path, b))
        for j in range(N):
            key = bytes("data", 'utf-8')
            test_data[N * i + j][:, :, 2] = np.reshape(temp[key][j][2048:], (32, 32))
            test_data[N * i + j][:, :, 1] = np.reshape(temp[key][j][1024:2048], (32, 32))
            test_data[N * i + j][:, :, 0] = np.reshape(temp[key][j][:1024], (32, 32))
            key = bytes("labels", 'utf-8')
            test_label[N * i + j] = temp[key][j]
    test_label = np_utils.to_categorical(np.array(test_label), 10)

    mean = np.mean(train_data, axis=0)
    train_data = train_data - mean
    test_data = test_data - mean

    return train_data, train_label, test_data, test_label


def BatchClipByL2norm(t, upper_bound, name=None):
  with tf.name_scope(values=[t, upper_bound], name=name,
                     default_name="batch_clip_by_l2norm") as name:
    saved_shape = tf.shape(t)
    batch_size = tf.slice(saved_shape, [0], [1])
    t2 = tf.reshape(t, tf.concat(axis=0, values=[batch_size, [-1]]))
    upper_bound_inv = tf.fill(tf.slice(saved_shape, [0], [1]),
                              1.0/upper_bound)
    # Add a small number to avoid divide by 0
    l2norm_inv = tf.reciprocal(tf.reduce_sum(t2, [1]) + 0.000001)
    scale = tf.minimum(l2norm_inv, upper_bound_inv) * upper_bound
    clipped_t = tf.matmul(tf.diag(scale), t2)
    clipped_t = tf.reshape(clipped_t, saved_shape, name=name)
  return clipped_t


def AddGaussianNoise(t, sigma, name=None):
  with tf.name_scope(values=[t, sigma], name=name,
                     default_name="add_gaussian_noise") as name:
    noisy_t = t + tf.random_normal(tf.shape(t), stddev=sigma)
  return noisy_t


def train():
    def pred():
        for _offset in range(0, train_data.shape[0], 512):
            _end = _offset + 512
            _X_batch = train_data[_offset:_end]
            _prob = sess.run(prediction, feed_dict={x: _X_batch, K.learning_phase(): 0})
            if _offset > 0:
                results = np.concatenate((results, _prob), axis=0)
            else:
                results = _prob

        return results

    def get_features():
        for _offset in range(0, train_data.shape[0], 512):
            _end = _offset + 512
            _X_batch = train_data[_offset:_end]
            _prob = sess.run(student_prob, feed_dict={x: _X_batch, K.learning_phase(): 0})
            if _offset > 0:
                results = np.concatenate((results, _prob), axis=0)
            else:
                results = _prob

        return results

    def clustering_data():
        features = get_features()
        standard_scaler = preprocessing.StandardScaler()
        features = standard_scaler.fit_transform(features)
        cluster_model = MiniBatchKMeans(n_clusters=10)
        cluster_model.fit(features)
        unique, counts = np.unique(cluster_model.labels_, return_counts=True)
        cluster_prob = counts / sum(counts)
        cluster_labels = cluster_model.labels_

        return cluster_labels, cluster_prob

    def pairwise_cross_entropy(x, y):
        k = np.multiply(x, np.log(np.divide(x, y)))
        dist = np.sum(k, axis=1)
        dist = np.reshape(dist, [x.shape[0], y.shape[0]])
        return dist

    def update_distances(cluster_centers, features, min_distances):
        if cluster_centers:
            # Update min_distances for all examples given new cluster center.
            center_features = features[cluster_centers]
            dist = pairwise_cross_entropy(features, center_features)

            if min_distances is None:
                min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                dist = np.min(dist, axis=1).reshape((-1, 1))
                min_distances = np.minimum(min_distances, dist)

        return min_distances

    def random_al():
        idx = np.array(range(train_data.shape[0]))
        active_samples = np.random.choice(idx, size=int(args.sel_rate * train_data.shape[0]), replace=False)

        return active_samples

    def gibbs_al():
        N = int(args.sel_rate * train_data.shape[0])
        p = pred()
        p = np.square(p)
        p = 1 - np.sum(p, axis=1)
        rank_ind = np.argsort(p)
        active_samples = rank_ind[-N:]

        return active_samples

    def margin_al():
        N = int(args.sel_rate * train_data.shape[0])
        distances = pred()
        sort_distances = np.sort(distances, 1)[:, -2:]
        min_margin = sort_distances[:, 1] - sort_distances[:, 0]
        rank_ind = np.argsort(min_margin)
        active_samples = rank_ind[0:N]

        return active_samples

    def diverse_al():
        N = int(args.sel_rate * train_data.shape[0])
        distances = pred()
        sort_distances = np.sort(distances, 1)[:, -2:]
        min_margin = sort_distances[:, 1] - sort_distances[:, 0]
        rank_ind = np.argsort(min_margin)

        cluster_labels, cluster_prob = clustering_data()
        new_batch_cluster_counts = [0 for _ in range(10)]
        new_batch = []
        for item in rank_ind:
            if len(new_batch) == N:
                break
            label = cluster_labels[item]
            if new_batch_cluster_counts[label] / N < cluster_prob[label]:
                new_batch.append(item)
                new_batch_cluster_counts[label] += 1
        n_slot_remaining = N - len(new_batch)
        batch_filler = [ind for ind in rank_ind if ind not in new_batch]
        new_batch.extend(batch_filler[0:n_slot_remaining])
        new_batch = np.array(new_batch)

        return new_batch

    def kcenter_al():
        N = int(args.sel_rate * train_data.shape[0])
        new_batch = []
        features = get_features()
        min_distances = None

        for _ in range(N):
            if new_batch == []:
                # Initialize centers with a randomly selected datapoint
                ind = np.random.choice(np.arange(train_data.shape[0]))
            else:
                ind = np.argmax(min_distances)
            assert ind not in new_batch

            min_distances = update_distances([ind], features, min_distances)
            new_batch.append(ind)

        new_batch = np.array(new_batch)
        return new_batch

    # options for active learning
    options = {'random': random_al,
               'margin': margin_al,
               'diverse': diverse_al,
               'gibbs': gibbs_al,
               'kcenter': kcenter_al}

    best = 0

    teacher_model = TeacherCifar('teacher_convlarge_cifar.npy')
    public_teacher_model = TeacherCifar('teacher_convlarge_public.npy')
    student_model = StudentCifar()

    teacher = teacher_model.get_logits()
    t_inter = teacher_model.get_hint()

    public_teacher = public_teacher_model.get_logits()
    public_inter = public_teacher_model.get_hint()

    student = student_model.get_logits()
    s_inter = student_model.get_guide()

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=student, labels=y)
    tf_self_loss = tf.reduce_mean(cross_entropy, name='self_loss')

    shape = tf.cast(tf.shape(x)[0], tf.float32)

    adp_weights = tf.Variable(tf.truncated_normal([1, 1, 64, 128], dtype=tf.float32, stddev=1e-1), name='adp_conv_w')
    adp_biases = tf.Variable(tf.constant(1.0, shape=[128], dtype=tf.float32), name='adp_conv_b')
    adapted = conv(s_inter, adp_weights, adp_biases)
    hint_loss = tf.nn.l2_loss(t_inter-adapted) / shape

    # calculate the avg value of public hint loss in this batch
    public_hint_loss = tf.reduce_sum((public_inter - adapted) ** 2, axis=[1,2,3])
    public_avg_hint_loss = tf.reduce_mean(public_hint_loss)

    # calculate the avg value of ||t_inter-adapted||^2 in this batch
    original_hint_loss = tf.reduce_sum((t_inter - adapted) ** 2, axis=[1, 2, 3])

    if args.add_noise == 'Y':
        bound_hint_loss = BatchClipByL2norm(original_hint_loss, l2norm_bound)
        sanitized_hint_loss = AddGaussianNoise(bound_hint_loss, args.sigma * l2norm_bound)
        tf_hint_loss = tf.reduce_mean(sanitized_hint_loss)/2
    else:
        tf_hint_loss = hint_loss

    prediction = tf.nn.softmax(student, name='predicted')
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1), name='correct_pred')
    accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    teacher_tau = tf.scalar_mul(1.0 / args.tau, teacher)
    teacher_tau = tf.nn.softmax(teacher_tau)

    public_tau = tf.scalar_mul(1.0 / args.tau, public_teacher)
    public_tau = tf.nn.softmax(public_tau)

    student_tau = tf.scalar_mul(1.0 / args.tau, student)
    student_prob = tf.nn.softmax(student_tau)
    original_kd_loss = tf.nn.softmax_cross_entropy_with_logits(logits=student_tau, labels=teacher_tau)
    public_kd_loss = tf.nn.softmax_cross_entropy_with_logits(logits=student_tau, labels=public_tau)
    kd_loss = tf.reduce_mean(original_kd_loss, name='kd_loss')
    public_kd_loss = tf.reduce_mean(public_kd_loss)

    if args.add_noise == 'Y':
        bound_kd_loss = BatchClipByL2norm(original_kd_loss, l2norm_bound)
        sanitized_kd_loss = AddGaussianNoise(bound_kd_loss, args.sigma * l2norm_bound)
        tf_kd_loss = tf.reduce_mean(sanitized_kd_loss)
    else:
        tf_kd_loss = kd_loss

    optimizer_hint = tf.train.AdamOptimizer(learning_rate=lr).minimize(tf_hint_loss)
    optimizer_kd = tf.train.AdamOptimizer(learning_rate=lr).minimize(tf_kd_loss)
    optimizer_self = tf.train.AdamOptimizer(learning_rate=lr).minimize(tf_self_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        train_data, train_label, test_data, test_label=read_cifar10()
        index = np.array(range(train_data.shape[0]))    # index randomly ordered

        pre_time = time.time()
        decay_step = int(args.iterations*0.7)

        hint_epoch = args.hint_epoch

        for i in range(hint_epoch):
            print('------------>Hint Learning Epoch:    ', i + 1)
            idx_sel = np.random.choice(index, size=int(args.sel_rate * index.shape[0]), replace=False)

            batch_round = 0
            batch_l2loss = 0.
            batch_noisy_l2loss = 0.

            for offset in range(0, idx_sel.shape[0], args.batchsize):
                end = offset + args.batchsize
                batch_x, batch_y = train_data[idx_sel[offset:end]], train_label[idx_sel[offset:end]]

                bound = sess.run(public_avg_hint_loss,
                                 feed_dict={x: batch_x, y: batch_y, K.learning_phase(): 1})

                _, l2loss, noisy_l2loss = sess.run([optimizer_hint, hint_loss, tf_hint_loss],
                                                   feed_dict={x: batch_x, y: batch_y, l2norm_bound: bound,
                                                              lr: args.lr, K.learning_phase(): 1})

                batch_round += 1
                batch_l2loss += (l2loss * len(batch_x))
                batch_noisy_l2loss += (noisy_l2loss * len(batch_x))
                if batch_round % 40 == 0 or end >= len(idx_sel):
                    print("Batch: {}/{}...".format(i + 1, batch_round),
                          "Training batch l2loss: {:.4f}".format(l2loss),
                          "Training batch noisy l2loss: {:.4f}".format(noisy_l2loss))
                    sys.stdout.flush()

            now_time = time.time()
            avg_time = (now_time - pre_time)
            pre_time = now_time
            print("")
            print("Epoch: {}/{}...".format(i + 1, hint_epoch),
                  "Training l2loss: {:.4f}".format(batch_l2loss / len(train_data)),
                  "Training noisy l2loss: {:.4f}".format(batch_noisy_l2loss / len(train_data)))
            print("Time: %.3f seconds" % (avg_time))

        # warm start
        for i in range(3):
            print('------------>Self Learning Epoch:    ', i + 1)
            batch_round = 0
            batch_entropy = 0.
            batch_acc = 0.
            batchsize = 128

            for offset in range(0, train_data.shape[0], batchsize):
                end = offset + batchsize
                batch_x, batch_y = train_data[index[offset:end]], train_label[index[offset:end]]

                _, self_loss, acc = sess.run([optimizer_self, tf_self_loss, accuracy_op],
                                             feed_dict={x: batch_x, y: batch_y, lr: 1e-3,
                                                        K.learning_phase(): 1})

                batch_round += 1
                batch_entropy += (self_loss * len(batch_x))
                batch_acc += (acc * len(batch_x))
                if batch_round % 100 == 0 or end >= len(train_data):
                    print("Batch: {}/{}...".format(i + 1, batch_round),
                          "Training entropy loss: {:.4f}".format(self_loss),
                          "Training batch acc: {:.4f}".format(acc))
                    sys.stdout.flush()

            now_time = time.time()
            avg_time = (now_time - pre_time)
            pre_time = now_time
            print("")
            print("Epoch: {}/{}...".format(i + 1, 6),
                  "Training entropy: {:.4f}".format(batch_entropy / len(train_data)),
                  "Training accuracy: {:.4f}".format(batch_acc / len(train_data)))
            print("Time: %.3f seconds" % (avg_time))

            total_acc = 0.
            for offset in range(0, test_data.shape[0], args.batchsize):
                end = offset + args.batchsize
                X_batch = test_data[offset:end]
                y_batch = test_label[offset:end]
                acc = sess.run(accuracy_op, feed_dict={x: X_batch, y: y_batch, K.learning_phase(): 0})
                total_acc += (acc * X_batch.shape[0])
            total_acc = total_acc / test_data.shape[0]
            print("Epoch", i + 1)
            print("Test Accuracy =", total_acc)
            print("")
            sys.stdout.flush()

        for ite in range(args.iterations):
            print('======Iteration {}======'.format(ite))

            if ite <= decay_step:
                self_epoch = max(args.self_epoch - ite * 2, 1)
                kd_epoch = args.kd_epoch
                learning_rate = args.lr
            else:
                self_epoch = max(args.self_epoch - ite * 2, 1)
                kd_epoch = args.kd_epoch
                learning_rate = args.lr / 10

            for i in range(kd_epoch):
                print('------------>KD Learning Epoch:    ', i + 1)
                idx_sel = options[args.sel_method]()

                batch_round = 0
                batch_l2loss = 0.
                batch_noisy_l2loss = 0.
                batch_acc = 0.

                for offset in range(0, idx_sel.shape[0], args.batchsize):
                    end = offset + args.batchsize
                    batch_x, batch_y = train_data[idx_sel[offset:end]], train_label[idx_sel[offset:end]]

                    # kd_loss is the avg value of KD loss in this batch
                    bound = sess.run(public_kd_loss, feed_dict={x: batch_x, y: batch_y, K.learning_phase(): 1})

                    _, l2loss, noisy_l2loss, acc = sess.run([optimizer_kd, kd_loss, tf_kd_loss, accuracy_op],
                                                            feed_dict={x: batch_x, y: batch_y, l2norm_bound: bound,
                                                                       lr: learning_rate, K.learning_phase(): 1})

                    batch_round += 1
                    batch_l2loss += (l2loss * len(batch_x))
                    batch_noisy_l2loss += (noisy_l2loss * len(batch_x))
                    batch_acc += (acc * len(batch_x))
                    if batch_round % 40 == 0 or end >= len(idx_sel):
                        print("Batch: {}/{}...".format(i+1, batch_round),
                              "Training batch l2loss: {:.4f}".format(l2loss),
                              "Training batch noisy l2loss: {:.4f}".format(noisy_l2loss),
                              "Training batch acc: {:.4f}".format(acc))
                        sys.stdout.flush()

                now_time = time.time()
                avg_time = (now_time - pre_time)
                pre_time = now_time
                print("")
                print("Epoch: {}/{}...".format(i+1, kd_epoch),
                      "Training l2loss: {:.4f}".format(batch_l2loss / len(idx_sel)),
                      "Training noisy l2loss: {:.4f}".format(batch_noisy_l2loss / len(idx_sel)),
                      "Training accuracy: {:.4f}".format(batch_acc / len(idx_sel)))
                print("Time: %.3f seconds" % (avg_time))

                total_acc = 0.
                for offset in range(0, test_data.shape[0], args.batchsize):
                    end = offset + args.batchsize
                    X_batch = test_data[offset:end]
                    y_batch = test_label[offset:end]
                    acc = sess.run(accuracy_op, feed_dict={x: X_batch, y: y_batch, K.learning_phase(): 0})
                    total_acc += (acc * X_batch.shape[0])
                total_acc = total_acc / test_data.shape[0]
                print("Epoch", i + 1)
                print("Test Accuracy =", total_acc)
                print("")
                sys.stdout.flush()

                if best < total_acc:
                    best = total_acc
                    print("KD Epoch ", i + 1, " is currently best. Saving the model ...")
                    sys.stdout.flush()

            for i in range(self_epoch):
                print('------------>Self Learning Epoch:    ', i + 1)
                batch_round = 0
                batch_entropy = 0.
                batch_acc = 0.
                batchsize = 128

                for offset in range(0, train_data.shape[0], batchsize):
                    end = offset + batchsize
                    batch_x, batch_y = train_data[index[offset:end]], train_label[index[offset:end]]

                    _, self_loss, acc = sess.run([optimizer_self, tf_self_loss, accuracy_op],
                                                 feed_dict={x: batch_x, y: batch_y, lr: learning_rate,
                                                            K.learning_phase(): 1})

                    batch_round += 1
                    batch_entropy += (self_loss * len(batch_x))
                    batch_acc += (acc * len(batch_x))
                    if batch_round % 100 == 0 or end >= len(train_data):
                        print("Batch: {}/{}...".format(i + 1, batch_round),
                              "Training entropy loss: {:.4f}".format(self_loss),
                              "Training batch acc: {:.4f}".format(acc))
                        sys.stdout.flush()

                now_time = time.time()
                avg_time = (now_time - pre_time)
                pre_time = now_time
                print("")
                print("Epoch: {}/{}...".format(i + 1, self_epoch),
                      "Training entropy: {:.4f}".format(batch_entropy / len(train_data)),
                      "Training accuracy: {:.4f}".format(batch_acc / len(train_data)))
                print("Time: %.3f seconds" % (avg_time))

                total_acc = 0.
                for offset in range(0, test_data.shape[0], args.batchsize):
                    end = offset + args.batchsize
                    X_batch = test_data[offset:end]
                    y_batch = test_label[offset:end]
                    acc = sess.run(accuracy_op, feed_dict={x: X_batch, y: y_batch, K.learning_phase(): 0})
                    total_acc += (acc * X_batch.shape[0])
                total_acc = total_acc / test_data.shape[0]
                print("Epoch", i + 1)
                print("Test Accuracy =", total_acc)
                print("")
                sys.stdout.flush()

                if best < total_acc:
                    best = total_acc
                    print("Self Epoch ", i + 1, " is currently best. Saving the model ...")
                    sys.stdout.flush()


if __name__ == '__main__':
    print("Reading args....")
    print(args)

    train()
    log_file.close()
