#!/usr/bin/env python
"""Consumes stream for printing all messages to the console.
"""
import tracemalloc
import argparse
import json
import time
import socket
import sys
import numpy as np
import pandas as pd
from confluent_kafka import Consumer, KafkaError, KafkaException
import torch
import psutil
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pypots.utils.metrics import cal_mae, cal_rmse, cal_mre
from torch_geometric.data import Data
import copy
from datetime import datetime
import os
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.transforms import FeaturePropagation
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

max_tau = 5

def select_tau(X_mask):
    miss_ratio = np.sum(X_mask) / (X_mask.shape[0] * X_mask.shape[1])
    tau = int(miss_ratio*max_tau)
    if tau <= 0:
        tau = 1
    return tau


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
        print('test value, true value shp:', true_value.shape)

        X_mask = np.isnan(msg_values_win).astype(int)
        print('X_mask shp', X_mask.shape)
        X_mask_cp = copy.copy(X_mask)

        msg_values_win = np.nan_to_num(msg_values_win, nan=0)

        X = copy.copy(msg_values_win)
        in_channels = X.shape[1]
        print('in_channels:', in_channels)

        X = torch.FloatTensor(X).to(device)
        X_mask = torch.LongTensor(X_mask).to(device)

        X_imputed = copy.copy(X)


        st = datetime.now()

        adj = np.zeros((window_len, window_len))
        tau = select_tau(X_mask_cp)
        print('selected tau:', tau)

        for i in range(window_len):
            adj[i, i + 1:i + 1 + tau] = 1
            adj[i + 1:i + 1 + tau, i] = 1

        # for i in range(window_len):
        #     min_end = min(i+1+tau, window_len)
        #     for j in range(i+1, min_end):
        #         adj[i, j] = 1



        norm_adj = torch.FloatTensor(adj)
        edge_index_hori = norm_adj.nonzero()
        print('edge_index_hori', edge_index_hori)
        edge_weight = norm_adj[edge_index_hori[:,0], edge_index_hori[:,1]]
        edge_index = edge_index_hori.t().contiguous()
        print('edge index', edge_index, edge_index.shape)
        print('edge_weight', edge_weight, edge_weight.shape)

        data = Data(x=X_imputed, edge_index=edge_index, edge_attr=edge_weight)

        # edge_index = knn_graph(X_imputed, args.k, batch=None, loop=False)
        # data = Data(x=X_imputed, edge_index=edge_index)
        # print('edge index', edge_index, edge_index.shape)


        transform = FeaturePropagation(missing_mask=X_mask.bool(), num_iterations=args.epochs)

        data = transform(data)

        current_time = datetime.now()
        elapsed_time = (current_time - st).total_seconds()*1000
        print('elapsed time:', elapsed_time)
        elapsed_time_list.append(round(elapsed_time, 4))

        X_imputed = data.x
        print('X_imputed shp:', X_imputed.shape)

        test_X = X_imputed[-period:,]

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
        mae_error_stream.append(round(mae_error.item(), 4))
        mse_error_stream.append(round(mse_eror.item(), 4))
        mre_error_stream.append(round(mre_error.item(), 4))
        index_stream.append(index)

        total_index_stream.extend(list(range(index-period+1, index+1)))
        imputed_data_stream.append(test_X)

        for pi in range(period):
            left_test_value = window_data.pop(0)
            left_1 = test_value_list.pop(0)
            left_2 = true_value_list.pop(0)
        # print(f'current indexxx:{index}, remaininig window list, test list, true list:', len(window_data), len(test_value_list), len(true_value_list))


if __name__ == "__main__":
    tracemalloc.start()
    tracker = EmissionsTracker(project_name="imputation", save_to_file=False)

    # tracker.start_task(f"{args.method}_win_{args.window_len}_tau_{tau}_p_{args.p}")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topic', type=str, help='Name of the Kafka topic to stream.', default='my-stream')
    parser.add_argument("--epochs", type=int, default=100)
    # parser.add_argument("--k", type=int, default=3)

    period = 10

    window = period + 40

    # parser.add_argument('--method', type=str, default=f'fp_tau_da_p{period}', required=False, help='feature propagation')

    parser.add_argument('--method', type=str, default=f'fp_tau_da5', required=False, help='feature propagation')

    parser.add_argument("--window_len", type=int, default=window)
    # parser.add_argument("--tau", type=int, default=5)
    parser.add_argument("--period", type=int, default=period)
    parser.add_argument('--prefix', type=str, default='', required=False, help='')

    window_data = []
    test_value_list = []
    true_value_list = []
    mae_error_stream = []
    mse_error_stream = []
    mre_error_stream = []
    energy_consume_list = []
    index_stream = []
    total_index_stream = []

    elapsed_time_list = []
    imputed_data_stream = []
    raw_imputed_data_stream = []

    #meta info from source
    dataset = None
    seq_len = None
    miss_ratio = None

    args = parser.parse_args()

    window_len = args.window_len
    period = args.period

    torch.random.manual_seed(2021)
    device = torch.device('cpu')
    epochs = args.epochs
    dt_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # conf = {'bootstrap.servers': 'localhost:9092',
    #         'default.topic.config': {'auto.offset.reset': 'smallest'},
    #         'group.id':args.method } #'_'.join([args.method, dt_str])

    conf = read_config()

    conf["group.id"] = args.method
    conf["auto.offset.reset"] = "earliest"

    consumer = Consumer(conf)
    running = True
    flag = 0
    try:
        while running:
            consumer.subscribe([args.topic])
            # msg = consumer.consume(num_messages=p, timeout=-1)
            # msg = consumer.consume(num_messages=1, timeout=-1)
            msg = consumer.poll(3)
            if msg is None or msg == []:
                print('waiting...')
                if flag == 1:
                    print('done!!! time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    perform_stream = [mae_error_stream, mse_error_stream, mre_error_stream, elapsed_time_list]
                    perform_stream_df = pd.DataFrame(perform_stream, index=['mae', 'rmse', 'mre', 'time'], columns=index_stream).T
                    perform_stream_df.to_csv(f'./exp_results_detail/per_{dataset}_{args.method}_ratio_{miss_ratio}_seq_{seq_len}_period_{period}_win_{window_len}_tau_d_epoch_{args.epochs}.csv', sep='\t', index=True, header=True)
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

                    avg_df['memo'] = round(peak_memo/1024**2,2) #now the unit becomes MB
                    avg_df['total_energy'] = round(sum(energy_consume_list), 6)
                    avg_df['avg_energy'] = round(np.average(energy_consume_list), 8)
                    avg_df.to_csv(f'./exp_results/{dataset}_{args.method}_ratio_{miss_ratio}_seq_{seq_len}_period_{period}_win_{window_len}_tau_d_epoch_{args.epochs}.csv', sep=',', index=True, header=True)


                    imputed_data_arr = np.concatenate(imputed_data_stream, axis=0)
                    imputed_data_df = pd.DataFrame(imputed_data_arr, index=total_index_stream)
                    imputed_data_df.to_csv( f'./impute_detail/impu_{dataset}_{args.method}_ratio_{miss_ratio}_seq_{seq_len}_period_{period}_win_{window_len}_tau_d_epoch_{args.epochs}.csv', sep='\t', index=True, header=False)
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
                msg_process(msg)
                emissions_infos = tracker.stop_task("imputation")
                energy_consume_list.append(round(emissions_infos.energy_consumed, 8))

    except KeyboardInterrupt:
        print('perceived keyboard interrupt!!')


    finally:
        # Close down consumer to commit final offsets.
        print('perceive close information!')

        consumer.close()

