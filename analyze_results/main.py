# Modified by Raul Mur-Artal
# Automatically compute the optimal scale factor for monocular VO/SLAM.

# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Requirements:
# sudo apt-get install python-argparse

"""
This script computes the absolute trajectory error from the ground truth
trajectory and the estimated trajectory.
"""
import os
import sys
import numpy
import argparse

import numpy as np

import associate
import matplotlib

import datetime
import matplotlib.pyplot as plt


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)

    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)
    """

    numpy.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)

    W = numpy.zeros((3, 3))
    for column in range(model.shape[1]):
        W += numpy.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = numpy.linalg.linalg.svd(W.transpose())
    S = numpy.matrix(numpy.identity(3))
    if (numpy.linalg.det(U) * numpy.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U * S * Vh

    rotmodel = rot * model_zerocentered
    dots = 0.0
    norms = 0.0

    for column in range(data_zerocentered.shape[1]):
        dots += numpy.dot(data_zerocentered[:, column].transpose(), rotmodel[:, column])
        normi = numpy.linalg.norm(model_zerocentered[:, column])
        norms += normi * normi

    s = float(dots / norms)

    transGT = data.mean(1) - s * rot * model.mean(1)
    trans = data.mean(1) - rot * model.mean(1)

    model_alignedGT = s * rot * model_zerocentered + transGT
    model_aligned = rot * model_zerocentered + trans

    alignment_errorGT = model_alignedGT - data_zerocentered
    alignment_error = model_aligned - data_zerocentered

    trans_errorGT = numpy.sqrt(numpy.sum(numpy.multiply(alignment_errorGT, alignment_errorGT), 0)).A[0]
    trans_error = numpy.sqrt(numpy.sum(numpy.multiply(alignment_error, alignment_error), 0)).A[0]

    return rot, transGT, trans_errorGT, trans, trans_error, s


def plot_traj(ax, stamps, traj, style, color, label):
    """
    Plot a trajectory using matplotlib.

    Input:
    ax -- the plot
    stamps -- time stamps (1xn)
    traj -- trajectory (3xn)
    style -- line style
    color -- line color
    label -- plot legend

    """
    stamps.sort()
    interval = numpy.median([s - t for s, t in zip(stamps[1:], stamps[:-1])])
    x = []
    y = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i] - last < 2 * interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
        elif len(x) > 0:
            ax.plot(x, y, style, color=color, label=label)
            label = ""
            x = []
            y = []
        last = stamps[i]
    if len(x) > 0:
        ax.plot(x, y, style, color=color, label=label)


if __name__ == "__main__":
    ground_truth = sys.argv[1]
    test_file = sys.argv[2]
    orbslam_file = sys.argv[3]
    test_file_dir = os.path.dirname(test_file) + "/"
    ground_truth_dic = associate.read_file_list(ground_truth, False)
    test_dic = associate.read_file_list(test_file, False)
    orbslam_dic = associate.read_file_list(orbslam_file, False)

    matches = associate.associate(ground_truth_dic, test_dic, 0, 10 ** 9 / 30)
    matches_slam = associate.associate(ground_truth_dic, orbslam_dic, 0, 10 ** 9 / 30)
    if len(matches) < 2:
        sys.exit(
            "Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! Did you choose the correct sequence?")
    scale = 1
    ground_xyz = numpy.matrix(
        [[float(value) for value in ground_truth_dic[a][0:3]] for a, b in matches_slam]).transpose()
    test_xyz = numpy.matrix(
        [[float(value) * float(scale) for value in test_dic[b][0:3]] for a, b in matches]).transpose()
    slam_xyz = numpy.matrix(
        [[float(value) * float(scale) for value in orbslam_dic[b][0:3]] for a, b in matches_slam]).transpose()
    slam_rot, slam_transGT, slam_trans_errorGT, slam_trans, slam_trans_error, slam_scale = align(slam_xyz, ground_xyz)
    ground_xyz = numpy.matrix([[float(value) for value in ground_truth_dic[a][0:3]] for a, b in matches]).transpose()

    dictionary_items = test_dic.items()
    sorted_test_list = sorted(dictionary_items)
    sorted_slam_list = sorted(orbslam_dic.items())
    test_xyz_full = numpy.matrix(
        [[float(value) * float(scale) for value in sorted_test_list[i][1][0:3]] for i in
         range(len(sorted_test_list))]).transpose()
    slam_xyz_full = numpy.matrix(
        [[float(value) * float(scale) for value in sorted_slam_list[i][1][0:3]] for i in
         range(len(sorted_slam_list))]).transpose()
    rot, transGT, trans_errorGT, trans, trans_error, scale = align(test_xyz, ground_xyz)

    test_xyz_aligned = scale * rot * test_xyz + trans
    test_xyz_notscaled = rot * test_xyz + trans
    test_xyz_notscaled_full = rot * test_xyz_full + trans
    slam_xyz_aligned = slam_scale * slam_rot * slam_xyz + slam_trans
    slam_xyz_notscaled = slam_rot * slam_xyz + slam_trans
    slam_xyz_notscaled_full = slam_rot * slam_xyz_full + slam_trans
    ground_truth_stamps = list(ground_truth_dic.keys())
    ground_truth_stamps.sort()
    ground_truth_xyz_full = numpy.matrix(
        [[float(value) for value in ground_truth_dic[b][0:3]] for b in ground_truth_stamps]).transpose()
    ground_truth_xyz_full -= ground_truth_xyz_full.mean()
    test_stamps = list(test_dic.keys())
    test_stamps.sort()
    slam_stamps = list(orbslam_dic.keys())
    slam_stamps.sort()
    test_xyz_full_aligned = scale * rot * test_xyz_full + trans
    test_xyz_full_aligned -= test_xyz_full_aligned.mean(1)
    slam_xyz_full_aligned = slam_scale * slam_rot * slam_xyz_full + slam_trans
    slam_xyz_full_aligned -= slam_xyz_full_aligned.mean(1)
    ground_truth_xyz_full -= ground_truth_xyz_full.mean(1)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    errors_file = open(test_file_dir + "errors_" + current_time + ".txt", "w")
    stdout = sys.stdout
    sys.stdout = errors_file
    print("****************test***************")
    print("compared_pose_pairs %d pairs" % (len(trans_error)))

    print("absolute_translational_error.rmse %f m" % numpy.sqrt(
        numpy.dot(trans_error, trans_error) / len(trans_error)))
    print("absolute_translational_error.mean %f m" % numpy.mean(trans_error))
    print("absolute_translational_error.median %f m" % numpy.median(trans_error))
    print("absolute_translational_error.std %f m" % numpy.std(trans_error))
    print("absolute_translational_error.min %f m" % numpy.min(trans_error))
    print("absolute_translational_error.max %f m" % numpy.max(trans_error))
    print("absolute_translational_errorGT.rmse %f m" % numpy.sqrt(
        numpy.dot(trans_errorGT, trans_errorGT) / len(trans_errorGT)))
    print("max idx: %i" % numpy.argmax(trans_error))
    # print "%f" % len(trans_error)
    print("compared_pose_pairs %d pairs" % (len(trans_error)))
    print("****************slam***************")
    print("compared_pose_pairs %d pairs" % (len(slam_trans_error)))

    print("absolute_translational_error.rmse %f m" % numpy.sqrt(
        numpy.dot(slam_trans_error, slam_trans_error) / len(slam_trans_error)))
    print("absolute_translational_error.mean %f m" % numpy.mean(slam_trans_error))
    print("absolute_translational_error.median %f m" % numpy.median(slam_trans_error))
    print("absolute_translational_error.std %f m" % numpy.std(slam_trans_error))
    print("absolute_translational_error.min %f m" % numpy.min(slam_trans_error))
    print("absolute_translational_error.max %f m" % numpy.max(slam_trans_error))
    print("absolute_translational_errorGT.rmse %f m" % numpy.sqrt(
        numpy.dot(slam_trans_errorGT, slam_trans_errorGT) / len(slam_trans_errorGT)))
    print("max idx: %i" % numpy.argmax(slam_trans_error))
    # print "%f" % len(trans_error)
    print("compared_pose_pairs %d pairs" % (len(slam_trans_error)))
    print("****************diff***************")
    print("compared_pose_pairs %d pairs" % (len(trans_error) - len(slam_trans_error)))

    print("absolute_translational_error.rmse %f m" % (numpy.sqrt(
        numpy.dot(trans_error, trans_error) / len(trans_error)) - numpy.sqrt(
        numpy.dot(slam_trans_error, slam_trans_error) / len(slam_trans_error))))
    print("absolute_translational_error.mean %f m" % (numpy.mean(trans_error) - numpy.mean(slam_trans_error)))
    print("absolute_translational_error.median %f m" % (numpy.median(trans_error) - numpy.median(slam_trans_error)))
    print("absolute_translational_error.std %f m" % (numpy.std(trans_error) - numpy.std(slam_trans_error)))
    print("absolute_translational_error.min %f m" % (numpy.min(trans_error) - numpy.min(slam_trans_error)))
    print("absolute_translational_error.max %f m" % (numpy.max(trans_error) - numpy.max(slam_trans_error)))
    # print "%f" % len(trans_error)
    print("absolute_translational_errorGT.rmse %f m" % (numpy.sqrt(
        numpy.dot(trans_errorGT, trans_errorGT) / len(trans_errorGT)) - numpy.sqrt(
        numpy.dot(slam_trans_errorGT, slam_trans_errorGT) / len(slam_trans_errorGT))))
    matplotlib.use('Agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_traj(ax, slam_stamps, slam_xyz_full_aligned.transpose().A, '-', "red", "orbslam3")
    plot_traj(ax, test_stamps, test_xyz_full_aligned.transpose().A, '-', "blue", "test")
    plot_traj(ax, ground_truth_stamps, ground_truth_xyz_full.transpose().A, '-', "black", "ground truth")
    ax.legend()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    plt.axis('equal')
    fig_path = test_file_dir + "result_fig_" + current_time + ".pdf"
    plt.savefig(fig_path, format="pdf")
    sys.stdout = stdout
