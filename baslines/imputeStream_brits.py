#!/usr/bin/env python
"""Consumes stream for printing all messages to the console.
"""
import argparse
import json
import time
import socket
import sys
import numpy as np
import pandas as pd
from confluent_kafka import Consumer, KafkaError, KafkaException
import resource
import torch
from resource import *
import psutil
import random
import copy
from datetime import datetime
import tracemalloc
import os
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from pypots.data import load_specific_dataset, mcar, masked_fill
from pypots.imputation import BRITS
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from pypots.utils.metrics import calc_mae, calc_rmse, calc_mre
import sys
from codecarbon import EmissionsTracker


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


def get_model_size(model):
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('Model Size: {:.3f} MB'.format(size_all_mb))
    return size_all_mb

def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

# decorator function
def profile(func):
    def wrapper(*args, **kwargs):
        mem_before = process_memory()
        print('before', mem_before)
        result = func(*args, **kwargs)
        mem_after = process_memory()
        print('after', mem_after)
        print("555, {}:consumed memory:", mem_before, mem_after, mem_after - mem_before)
        return result
    return wrapper



def msg_process(msg):
    # Print the current time and the message.
    time_start = time.strftime("%Y-%m-%d %H:%M:%S")
    print('msg', type(msg))

    # test_value = list(map(lambda x:json.loads(x.value())['test'], msg))
    # true_value = list(map(lambda x:json.loads(x.value())['truth'], msg))

    loaded_js = json.loads(msg.value())

    index = loaded_js['index']
    test_value = loaded_js['test']
    true_value = loaded_js['truth']

    if index == 0:
        global miss_ratio, seq_len, dataset
        meta_info = loaded_js['meta_info']
        miss_ratio = meta_info['miss_ratio']
        seq_len = meta_info['seq_len']
        dataset = meta_info['dataset']

    test_value = np.array(test_value).reshape(1, -1)
    true_value = np.array(true_value).reshape(1, -1)


    print('current index', index)
    window_data.append(test_value)

    if len(window_data) > (window_len - period):
        true_value_list.append(true_value)
        test_value_list.append(test_value)

    if len(window_data) < window_len:
        print('current window len', len(window_data))
    else:
        test_value = np.concatenate(test_value_list, axis=0)
        true_value = np.concatenate(true_value_list, axis=0)
        test_mask = ((~np.isnan(true_value)) ^ (~np.isnan(test_value))).astype(np.float32)

        msg_values_win = np.array(window_data)

        msg_values_win = np.squeeze(msg_values_win, axis=1)
        print('window data shp', msg_values_win.shape)

        X_mask = np.isnan(msg_values_win).astype(int)
        print('X_mask shp', X_mask.shape)

        msg_values_win = np.nan_to_num(msg_values_win, nan=0)

        X = copy.copy(msg_values_win)
        in_channels = X.shape[1]
        print('in_channels:', in_channels)

        X = torch.FloatTensor(X).to(device)
        X_mask = torch.LongTensor(X_mask).to(device)

        norm_adj = torch.FloatTensor(adj)

        edge_index_hori = norm_adj.nonzero()
        print('edge_index_hori', edge_index_hori)

        edge_weight = norm_adj[edge_index_hori[:,0], edge_index_hori[:,1]]

        edge_index = edge_index_hori.t().contiguous()

        print('edge index', edge_index, edge_index.shape)
        print('edge_weight', edge_weight, edge_weight.shape)

        X = masked_fill(X, X_mask, np.nan)
        num_of_channel = X.shape[1]
        X = torch.unsqueeze(X, dim=0)

        # Model training. This is PyPOTS showtime.

        imputer = BRITS(n_steps=args.window_len, n_features=num_of_channel, rnn_hidden_size=64, epochs=50, device=device)

        st = datetime.now()

        imputer.fit(X)  # train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model.
        imputation = imputer.impute(X)  # impute the originally-missing values and artificially-missing values

        current_time = datetime.now()
        elapsed_time = (current_time - st).total_seconds()*1000
        print('elapsed time:', elapsed_time)
        elapsed_time_list.append(round(elapsed_time, 4))

        imputation = torch.FloatTensor(imputation)
        X_imputed = copy.copy(imputation)

        X_imputed = torch.squeeze(X_imputed, dim=0)
        print('X_imputed shp:', X_imputed.shape, type(X_imputed))

        test_X = X_imputed[-period:,:]
        # raw_test_X = raw_out[[-1]]

        true_X = copy.copy(true_value)
        true_X = np.nan_to_num(true_X, nan=0)
        true_X = torch.FloatTensor(true_X).to(device)

        test_mask = torch.FloatTensor(test_mask).to(device)

        mae_error = calc_mae(test_X, true_X, test_mask)
        print('mae_error', mae_error)

        mse_eror = calc_rmse(test_X, true_X, test_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)

        mre_error = calc_mre(test_X, true_X, test_mask)

        print('valid impute value samples:', (test_X[torch.where(test_mask == 1)][:10]))
        print('valid true value samples:', (true_X[torch.where(test_mask == 1)][:10]))

        num_params = sum(p.numel() for p in imputer.model.parameters() if p.requires_grad) / 1e6
        print('num of Parameters:', num_params)
        global model_size
        model_size = get_model_size(imputer.model)
        print('BRITS Model Size:', model_size)


        mae_error_stream.append(round(mae_error.item(), 4))
        mse_error_stream.append(round(mse_eror.item(), 4))
        mre_error_stream.append(round(mre_error.item(), 4))
        index_stream.append(index)
        # imputed_data_stream.append(test_X)
        # raw_imputed_data_stream.append(raw_test_X)

        for pi in range(period):
            left_test_value = window_data.pop(0)
            left_1 = test_value_list.pop(0)
            left_2 = true_value_list.pop(0)
        # print(f'current indexxx:{index}, remaininig window list, test list, true list:', len(window_data), len(test_value_list), len(true_value_list))


if __name__ == "__main__":
    tracemalloc.start()
    tracker = EmissionsTracker(project_name="imputation", save_to_file=False)


    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topic', type=str, help='Name of the Kafka topic to stream.', default='my-stream')
    parser.add_argument("--epochs", type=int, default=100)


    period = 10
    window = period + 40
    # parser.add_argument('--method', type=str, default=f'brits_p{period}', required=False, help='feature propagation')
    parser.add_argument('--method', type=str, default=f'brits', required=False, help='feature propagation')

    parser.add_argument("--window_len", type=int, default=window)

    parser.add_argument("--period", type=int, default=period)

    parser.add_argument("--tau", type=int, default=5)


    parser.add_argument('--prefix', type=str, default='', required=False, help='')

    window_data = []
    test_value_list = []
    true_value_list = []
    energy_consume_list = []
    mae_error_stream = []
    mse_error_stream = []
    mre_error_stream = []
    index_stream = []
    elapsed_time_list = []
    imputed_data_stream = []
    raw_imputed_data_stream = []

    #meta info from source
    dataset = None
    seq_len = None
    miss_ratio = None
    model_size = None

    args = parser.parse_args()
    window_len = args.window_len
    period = args.period


    adj = np.zeros((window_len, window_len))

    for i in range(window_len):
        adj[i, i + 1:i + 1 + args.tau] = 1
        adj[i + 1:i + 1 + args.tau, i] = 1


    torch.random.manual_seed(2021)
    device = torch.device('cpu')
    epochs = args.epochs
    dt_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # conf = {'bootstrap.servers': 'localhost:9092',
    #         'default.topic.config': {'auto.offset.reset': 'smallest'},
    #         'group.id':args.method} #'_'.join([args.method, dt_str])

    conf = read_config()

    conf["group.id"] = args.method
    conf["auto.offset.reset"] = "earliest"

    consumer = Consumer(conf)
    running = True
    flag = 0
    try:
        while running:
            consumer.subscribe([args.topic])
            # msg = consumer.consume(num_messages=6, timeout=-1)
            # msg = consumer.consume(num_messages=1, timeout=-1)

            msg = consumer.poll(3)
            if msg is None or msg == []:
                print('waiting...')
                if flag == 1:
                    print('done!!! time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    perform_stream = [mae_error_stream, mse_error_stream, mre_error_stream, elapsed_time_list]
                    perform_stream_df = pd.DataFrame(perform_stream, index=['mae', 'rmse', 'mre', 'time'], columns=index_stream).T
                    perform_stream_df.to_csv(f'./exp_results_detail/per_{dataset}_{args.method}_ratio_{miss_ratio}_seq_{seq_len}_period_{period}_win_{window_len}_epoch_{args.epochs}.csv', sep='\t', index=True, header=True)
                    avg_df = perform_stream_df.mean(axis=0).round(3)
                    # avg_df = perform_stream_df.iloc[100-args.window_len:, :].mean(axis=0).round(3)
                    avg_df = pd.DataFrame(avg_df).T
                    # index_st = perform_stream_df.index[0] + 100-args.window_len
                    index_st = perform_stream_df.index[0]
                    index_ed = perform_stream_df.index[-1]
                    index_range = str(index_st) + '-' + str(index_ed)
                    avg_df.index = [index_range]
                    avg_df.columns = perform_stream_df.columns

                    current_memo, peak_memo = tracemalloc.get_traced_memory()
                    tracemalloc.stop()


                    total_memo = round(peak_memo / 1024 ** 2, 2) + model_size
                    avg_df['memo'] = round(total_memo, 2)  # for nn models, now the unit becomes MB
                    avg_df['total_energy'] = round(sum(energy_consume_list), 6)
                    avg_df['avg_energy'] = round(np.average(energy_consume_list), 8)
                    avg_df['heap'] = round(peak_memo / 1024 ** 2, 2)


                    avg_df.to_csv(f'./exp_results/{dataset}_{args.method}_ratio_{miss_ratio}_seq_{seq_len}_period_{period}_win_{window_len}_epoch_{args.epochs}.csv', sep=',', index=True, header=True)

                    # imputed_data_arr = np.concatenate(imputed_data_stream, axis=0)
                    # imputed_data_df = pd.DataFrame(imputed_data_arr, index=index_stream)
                    # imputed_data_df.to_csv( f'./impute_detail/impu_{dataset}_ratio_{miss_ratio}_seq_{seq_len}_win_{window_len}_epoch_{args.epochs}.csv', sep='\t', index=True, header=False)

                    break

            elif msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition event
                    sys.stderr.write('%% %s [%d] reached end at offset %d\n' %(msg.topic(), msg.partition(), msg.offset()))
                elif msg.error().code() == KafkaError.UNKNOWN_TOPIC_OR_PART:
                    sys.stderr.write('Topic unknown, creating %s topic\n' %(args.topic))
                elif msg.error():
                    raise KafkaException(msg.error())
            else:

                flag = 1
                tracker.start_task("imputation")
                emissions_infos = tracker.stop_task()
                # print('emissions_infos', emissions_infos)
                msg_process(msg)
                emissions_infos = tracker.stop_task("imputation")
                energy_consume_list.append(round(emissions_infos.energy_consumed, 8))




    except KeyboardInterrupt:
        print('perceived keyboard interrupt!!')


    finally:
        # Close down consumer to commit final offsets.
        print('perceive close information!')

        consumer.close()

