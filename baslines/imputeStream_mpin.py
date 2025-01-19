#!/usr/bin/env python
"""Consumes stream for printing all messages to the console.
"""
import argparse
import json
import time
import sys
import numpy as np
import pandas as pd
from confluent_kafka import Consumer, KafkaError, KafkaException
import torch
import psutil
import copy
from datetime import datetime
import tracemalloc
import os
from models.DynamicGNN import DynamicGCN, DynamicGAT, DynamicGraphSAGE, StaticGCN, StaticGraphSAGE, StaticGAT
from torch_geometric.nn import knn_graph
from models.regressor import MLPNet
import torch.optim as optim
from pypots.utils.metrics import cal_mae, cal_rmse, cal_mre
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


def build_GNN(in_channels, out_channels, k, base):
    if base == 'GAT':
        gnn = DynamicGAT(in_channels=in_channels, out_channels=out_channels, k=k).to(device)
    elif base == 'GCN':
        gnn = DynamicGCN(in_channels=in_channels, out_channels=out_channels, k=k).to(device)
    elif base == 'SAGE':
        gnn = DynamicGraphSAGE(in_channels=in_channels, out_channels=out_channels, k=k).to(device)

    return gnn

def build_GNN_static(in_channels, out_channels, k, base):
    if base == 'GAT':
        gnn = StaticGAT(in_channels=in_channels, out_channels=out_channels, k=k).to(device)
    elif base == 'GCN':
        gnn = StaticGCN(in_channels=in_channels, out_channels=out_channels, k=k).to(device)
    elif base == 'SAGE':
        gnn = StaticGraphSAGE(in_channels=in_channels, out_channels=out_channels, k=k).to(device)
    return gnn



def window_imputation(X, X_mask, edge_index):
    print('input X shp', X.shape)
    in_channels = X.shape[1]
    print('in_channels:', in_channels)
    X = np.nan_to_num(X)

    X = torch.FloatTensor(X).to(device)
    X_mask = torch.LongTensor(X_mask).to(device)
    # mean_f = torch.FloatTensor(mean_f).to(device)
    # std_f = torch.FloatTensor(std_f).to(device)

    gnn = build_GNN_static(in_channels=in_channels, out_channels=args.out_channels, k=args.k, base=args.base)
    gnn2 = build_GNN_static(in_channels=in_channels, out_channels=args.out_channels, k=args.k, base=args.base)

    model_list = [gnn, gnn2]
    regressor = MLPNet(args.out_channels, in_channels).to(device)

    trainable_parameters = []
    for model in model_list:
        trainable_parameters.extend(list(model.parameters()))

    trainable_parameters.extend(list(regressor.parameters()))
    filter_fn = list(filter(lambda p: p.requires_grad, trainable_parameters))

    num_of_params = sum(p.numel() for p in filter_fn)

    print('number of trainable parameters:', num_of_params)


    opt = optim.Adam(filter_fn, lr=args.lr, weight_decay=args.weight_decay)
    model_size = get_model_size(gnn) + get_model_size(gnn2) + get_model_size(regressor)
    print('MPIN_model size:', model_size)

    graph_impute_layers = len(model_list)
    st = datetime.now()

    # X_knn = X * X_mask
    X_knn = copy.deepcopy(X)

    # edge_index = knn_graph(X_knn, args.k, batch=None, loop=False, cosine=False)

    for pre_epoch in range(args.epochs):

        gnn.train()
        gnn2.train()
        regressor.train()
        opt.zero_grad()
        loss = 0
        X_imputed = copy.copy(X)

        # edge_index = None
        for i in range(graph_impute_layers):
            X_emb, edge_index = model_list[i](X_imputed, edge_index)
            pred = regressor(X_emb)
            X_imputed = X*X_mask + pred*(1 - X_mask)
            temp_loss = torch.sum(torch.abs(X - pred) * X_mask) / (torch.sum(X_mask) + 1e-5)
            loss += temp_loss

        loss.backward()
        opt.step()
        train_loss = loss.item()
        print('{n} epoch loss:'.format(n=pre_epoch), train_loss)

        trans_X = copy.copy(X_imputed)
    print('output imputed X shp', trans_X.shape)

    return trans_X, model_size

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

        st = datetime.now()

        X_imputed, mpin_model_size = window_imputation(X, 1-X_mask, edge_index)
        global model_size
        model_size = copy.copy(mpin_model_size)

        current_time = datetime.now()
        elapsed_time = (current_time - st).total_seconds()*1000
        print('elapsed time:', elapsed_time)
        elapsed_time_list.append(round(elapsed_time, 4))

        print('X_imputed shp:', X_imputed.shape)


        test_X = X_imputed[-period:, :]

        true_X = copy.copy(true_value)
        true_X = np.nan_to_num(true_X, nan=0)
        true_X = torch.FloatTensor(true_X).to(device)

        test_mask = torch.FloatTensor(test_mask).to(device)

        mae_error = cal_mae(test_X, true_X, test_mask)
        print('mae_error', mae_error)

        mse_eror = cal_rmse(test_X, true_X, test_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)

        mre_error = cal_mre(test_X, true_X, test_mask)

        print('valid impute value samples:', (test_X[torch.where(test_mask == 1)][:10]))
        print('valid true value samples:', (true_X[torch.where(test_mask == 1)][:10]))


        mae_error_stream.append(round(mae_error.item(), 4))
        mse_error_stream.append(round(mse_eror.item(), 4))
        mre_error_stream.append(round(mre_error.item(), 4))
        index_stream.append(index)

        imputed_data_stream.append(test_X.detach().numpy())
        total_index_stream.extend(list(range(index-period+1, index+1)))


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
    period = 10
    window_len = period + 40

    # parser.add_argument('--method', type=str, default=f'mpin_p{period}', required=False, help='message propagation imputation network')

    parser.add_argument('--method', type=str, default=f'mpin', required=False, help='message propagation imputation network')

    parser.add_argument("--window_len", type=int, default=window_len)
    parser.add_argument("--period", type=int, default=period)

    parser.add_argument("--tau", type=int, default=5)


    parser.add_argument('--prefix', type=str, default='', required=False, help='')

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument('--base', type=str, default='SAGE')

    parser.add_argument("--out_channels", type=int, default=256)
    parser.add_argument("--k", type=int, default=10)

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.1)


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
    total_index_stream = []

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
                    perform_stream_df.to_csv(f'./exp_results_detail/per_{dataset}_{args.method}_ratio_{miss_ratio}_seq_{seq_len}_period_{period}_win_{window_len}_tau_{args.tau}_epoch_{args.epochs}.csv', sep='\t', index=True, header=True)
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

                    avg_df.to_csv(f'./exp_results/{dataset}_{args.method}_ratio_{miss_ratio}_seq_{seq_len}_period_{period}_win_{window_len}_tau_{args.tau}_epoch_{args.epochs}.csv', sep=',', index=True, header=True)

                    imputed_data_arr = np.concatenate(imputed_data_stream, axis=0)
                    imputed_data_df = pd.DataFrame(imputed_data_arr, index=total_index_stream)
                    imputed_data_df.to_csv(f'./impute_detail/impu_{dataset}_{args.method}_ratio_{miss_ratio}_seq_{seq_len}_period_{period}_win_{window_len}_tau_{args.tau}_epoch_{args.epochs}.csv',sep='\t', index=True, header=False)
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

