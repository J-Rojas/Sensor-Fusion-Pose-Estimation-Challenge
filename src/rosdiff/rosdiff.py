#!/usr/bin/python

import argparse
import sys
import numpy as np
import rosbag
import datetime

appTitle = 'ROSbag diff finder'

def get_summary(bag):
    summary = {}
    for topic, msg, t in bag:
        if topic not in summary:
            summary[topic] = {
                'count': 0,
                'times': [],
            }
        summary[topic]['count'] += 1
        summary[topic]['times'].append(t.to_sec())

    return summary

def compare_summaries(summary1, summary2):
    diffs = {}
    for topic in summary1:
        if topic not in summary2:
            diffs[topic] = {
                'count': summary1[topic]['count'],
                'tsDiffs': summary1[topic]['count'],
                'maxTSDiff': 0,
                'mean': 0,
                'std': 0,
            }
            continue

        val1 = summary1[topic]
        val2 = summary2[topic]
        del summary2[topic]

        diffCount = abs(val1['count'] - val2['count'])

        ts1 = val1['times']
        ts2 = val2['times']

        dts = [a - b for a,b in zip(ts1, ts2)]
        tsDiffs = 0
        maxDiff = 0
        for t in dts:
            ta = abs(t)
            if ta >= 0.001:
                tsDiffs += 1
            if ta > maxDiff:
                maxDiff = ta
        a = np.array(dts)
        mean = np.mean(a)
        std = np.std(a)

        diffs[topic] = {
            'count': diffCount + tsDiffs,
            'tsDiffs': tsDiffs,
            'maxTSDiff': maxDiff,
            'mean': mean,
            'std': std,
        }

    for topic in summary2:
        diffs[topic] = {
            'count': summary2[topic]['count'],
            'tsDiffs': summary2[topic]['count'],
            'maxTSDiff': 0,
            'mean': 0,
            'std': 0,
        }

    return diffs

def print_diffs_report(diffs):
    topics = sorted(diffs.keys())
    for topic in topics:
        val = diffs[topic]
        print(topic)
        print('\t# missing/different: %d' % val['count'])
        print('\t# incorrect timestamps: %d' % val['tsDiffs'])
        print('\ttimestamp differences mean / std (s): %.3f / %.3f' % (val['mean'], val['std']))
        print('\tmaximum difference in timestamps (s): %.3f\n' % val['maxTSDiff'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=appTitle)
    parser.add_argument('--dataset1', type=str, default="dataset.bag", help='Dataset/ROS Bag 1 name')
    parser.add_argument('--dataset2', type=str, default="dataset.bag", help='Dataset/ROS Bag 2 name')
    args = parser.parse_args()

    dataset1 = args.dataset1
    dataset2 = args.dataset2

    print('reading rosbag %s' % dataset1)
    bag1 = rosbag.Bag(dataset1, 'r')
    topicTypesMap = bag1.get_type_and_topic_info().topics

    print('reading rosbag %s' % dataset2)
    bag2 = rosbag.Bag(dataset2, 'r')
    topicTypesMap = bag2.get_type_and_topic_info().topics

    summary1 = get_summary(bag1)
    summary2 = get_summary(bag2)

    diffs = compare_summaries(summary1, summary2)
    print_diffs_report(diffs)
