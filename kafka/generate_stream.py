#!/usr/bin/env python

"""Generates a stream to Kafka from a time series csv file.
"""

import argparse
import csv
import json
import sys
import time
from dateutil.parser import parse
from confluent_kafka import Producer
import socket
import pandas as pd
import numpy as np
import random
import copy
from datetime import datetime
from load_dataset_bi import load_wifi_data, load_synth_data, load_ICU_dataset, load_airquality_dataset


"""
motion:
10000 x 20

Bafu:
50000 x 10

Gas:
1000 x 100

"""

def read_config():
  # reads the client configuration from client.properties
  # and returns it as a key-value map
  config = {}
  with open("client.properties") as fh:
    for line in fh:
      line = line.strip()
      if len(line) != 0 and line[0] != "#":
        parameter, value = line.strip().split('=', 1)
        config[parameter] = value.strip()
  return config



def acked(err, msg):
    if err is not None:
        print("Failed to deliver message: %s: %s" % (str(msg.value()), str(err)))
    else:
        print("Message produced: %s" % (str(msg.value())))


def data_transform(X, ratio, nan=0):
    np.random.seed(7)
    original_shape = X.shape
    X = X.flatten()
    X_intact = np.copy(X)  # keep a copy of originally observed values in X_intact
    # select random indices for artificial mask
    indices = np.where(~np.isnan(X))[0].tolist()  # get the indices of observed values
    indices = np.random.choice(indices, int(len(indices) * ratio), replace=False)
    # create artificially-missing values by selected indices
    X[indices] = np.nan  # mask values selected by indices

    X = X.reshape(original_shape)
    return X



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--filename', type=str, help='Time series csv file.', default='./data.csv')
    parser.add_argument('--topic', type=str, help='Name of the Kafka topic to stream.', default='my-stream')
    parser.add_argument('--speed', type=float, default=1, required=False, help='Speed up time series by a given multiplicative factor.')
    parser.add_argument('--dataset', type=str, default='Motion', required=False, help='Speed up time series by a given multiplicative factor.')
    parser.add_argument('--miss_ratio', type=float, default=0.5, required=False, help='missing ratio')

    parser.add_argument('--sequence_len', type=int, default=600, required=False, help='the length of the stream')

    args = parser.parse_args()
    topic = args.topic
    p_key = args.filename

    # conf = {'bootstrap.servers': "localhost:9092",
    #         'client.id': socket.gethostname()}
    #
    # conf = {'bootstrap.servers': "130.226.173.106:9092",
    #                 'client.id': socket.gethostname()}

    conf = read_config()

    if args.dataset == 'Gas':
        dataset_name = 'drift6'
    elif args.dataset == 'Water':
        dataset_name = 'bafu'
    elif args.dataset == 'Motion':
        dataset_name = 'motion'

    producer = Producer(conf)
    Feature_df = pd.read_csv(f'../Datasets/{dataset_name}_normal.txt', sep=' ', header=None, index_col=None)
    print('dataset:', args.dataset, Feature_df.shape)
    Feature_df = Feature_df.iloc[:args.sequence_len, :]


    shp = Feature_df.shape
    print('shp of stream:', shp)

    # Feature_df.to_csv(f'./impute_detail/true_{args.dataset}_ratio_{args.ratio}_seq_{args.sequence_len}.csv', sep='\t', index=True, header=False)
    cnt = 0
    total_data_length = Feature_df.shape[0]

    X = Feature_df.values

    # X_intact, X = block_data_transform(X, n_mb=args.n_mb)
    X_intact = copy.copy(X)
    # first_half_X = X.iloc[args.sequence_len/2, :]

    X = data_transform(copy.copy(X), ratio=args.miss_ratio)

    # miss_Feature_df = pd.DataFrame(X).to_csv(f'./impute_detail/miss_{args.dataset}_ratio_{args.ratio}_seq_{args.sequence_len}.csv', sep=',', index=True, header=False)

    while cnt <= total_data_length - 1:
        try:
            print('cnt', cnt)
            test_value = X[cnt].tolist()
            print('test value', test_value)
            truth_value = X_intact[cnt].tolist()
            print('true value', truth_value)
            time.sleep(1)
            result = {}
            result['test'] = test_value
            result['truth'] = truth_value
            result['index'] = cnt
            if cnt == 0:
                result['meta_info'] = {'miss_ratio': args.miss_ratio, 'seq_len': args.sequence_len, 'dataset': args.dataset}
            jresult = json.dumps(result)
            cnt += 1
            producer.produce(topic, key=p_key, value=jresult, callback=acked)
            producer.flush()
        except TypeError:
            sys.exit()

    print('end time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

