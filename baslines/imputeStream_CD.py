
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
from pypots.utils.metrics import cal_mae, cal_rmse, cal_mre

import copy
from datetime import datetime
import tracemalloc
import os
from codecarbon import EmissionsTracker
from utils.cd_modules.recovery import centroid_recovery



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
        X_mask = (np.isnan(msg_values_win)).astype(int)
        print('X_mask shp', X_mask.shape)


        X = copy.copy(msg_values_win)
        in_channels = X.shape[1]
        print('in_channels:', in_channels)


        st = datetime.now()


        X_input = copy.deepcopy(X)

        X_imputed = centroid_recovery(X_input, truncation=truncate)

        current_time = datetime.now()
        elapsed_time = (current_time - st).total_seconds()*1000
        print('elapsed time:', elapsed_time)
        elapsed_time_list.append(round(elapsed_time, 4)) # count time cost

        print('X_imputed', X_imputed.shape)
        test_X = X_imputed[-period:,:]

        true_X = copy.copy(true_value)
        true_X = np.nan_to_num(true_X, nan=0)

        print('test_X true_X shpppp', test_X.shape, true_X.shape)
        mae_error = cal_mae(test_X, true_X, test_mask)

        mse_eror = cal_rmse(test_X, true_X, test_mask)

        mre_error = cal_mre(test_X, true_X, test_mask)

        print('valid impute value samples:', (test_X[np.where(test_mask == 1)][:10]))
        print('valid true value samples:', (true_X[np.where(test_mask == 1)][:10]))

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


if __name__ == "__main__":
    tracemalloc.start()
    tracker = EmissionsTracker(project_name="imputation", save_to_file=False)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topic', type=str, help='Name of the Kafka topic to stream.', default='my-stream')

    # parser.add_argument("--window_len", type=int, default=window)
    parser.add_argument("--p", type=int, default=10)

    parser.add_argument('--method', type=str, default=f'CD', required=False, help='feature propagation')

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
    total_index_stream = []

    imputed_data_stream = []

    #meta info from source
    dataset = None
    seq_len = None
    miss_ratio = None

    args = parser.parse_args()

    period = args.p
    window = period + 150
    truncate = 0  ### 0.3, 0.5, 0.7
    # truncate = 1 ### 0.9
    window_len = copy.copy(window)


    torch.random.manual_seed(2021)
    device = torch.device('cpu')
    conf = {'bootstrap.servers': 'localhost:9092',
            'default.topic.config': {'auto.offset.reset': 'smallest'},
            'group.id': args.method}
    consumer = Consumer(conf)
    running = True
    flag = 0
    try:
        while running:
            consumer.subscribe([args.topic])

            msg = consumer.poll(3)
            if msg is None or msg == []:
                print('waiting...')
                if flag == 1:
                    print('done!!! time:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    perform_stream = [mae_error_stream, mse_error_stream, mre_error_stream, elapsed_time_list]
                    perform_stream_df = pd.DataFrame(perform_stream, index=['mae', 'rmse', 'mre', 'time'], columns=index_stream).T
                    # perform_stream_df.to_csv(f'./exp_results_detail/per_{dataset}_{args.method}_ratio_{miss_ratio}_seq_{seq_len}_period_{period}_win_{window_len}.csv', sep='\t', index=True, header=True)
                    avg_df = perform_stream_df.mean(axis=0).round(3)
                    avg_df = pd.DataFrame(avg_df).T
                    index_st = perform_stream_df.index[0]
                    index_ed = perform_stream_df.index[-1]
                    index_range = str(index_st) + '-' + str(index_ed)
                    avg_df.index = [index_range]
                    avg_df.columns = perform_stream_df.columns

                    current_memo, peak_memo = tracemalloc.get_traced_memory()
                    tracemalloc.stop()

                    avg_df['memo'] = round(peak_memo / 1024**2, 2)  # now the unit becomes MB
                    avg_df['total_energy'] = round(sum(energy_consume_list), 6)
                    avg_df['avg_energy'] = round(np.average(energy_consume_list), 8)

                    avg_df.to_csv(f'./exp_results/{dataset}_{args.method}_ratio_{miss_ratio}_seq_{seq_len}_period_{period}_win_{window_len}.csv', sep=',', index=True, header=True)

                    imputed_data_arr = np.concatenate(imputed_data_stream, axis=0)
                    imputed_data_df = pd.DataFrame(imputed_data_arr, index=total_index_stream)
                    # imputed_data_df.to_csv(f'./impute_detail/impu_{dataset}_{args.method}_ratio_{miss_ratio}_seq_{seq_len}_period_{period}_win_{window_len}.csv', sep='\t', index=True, header=False)

                    break
            elif msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition event
                    sys.stderr.write('%% %s [%d] reached end at offset %d\n' %
                                     (msg.topic(), msg.partition(), msg.offset()))
                elif msg.error().code() == KafkaError.UNKNOWN_TOPIC_OR_PART:
                    sys.stderr.write('Topic unknown, creating %s topic\n' %
                                     (args.topic))
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

