# Lightweight Online Imputation of Sensor Data Streams via Temporal Feature Propagation



This is the repository of TFP project which consists of code, datasets, instructions to reproduce experimental results.


# Implementation Details 

### Requirements
- Pytorch 1.8.1
- Numpy 1.19.2
- Pandas 1.1.3
- Sklearn 0.24.1
- pypots
- torch_geometric
- confluent-kafka 2.2.0
- codecarbon 2.3.4

You may use " pip3 install -r requirements.txt" to install the above python libraries.
Notably, Docker is in need to help create Kafka data streams.

### Usage

***Produce a sensor data stream***: 

``` 
cd kafka; 
docker compose up --build; # start Kafka engine
python3 produceKafkaStream.py --dataset Motion --miss_ratio 0.9 # produce a sensor data stream with specified sparsity levels 
```

***Impute a sensor data stream***: 

*wTFP/wTFPd*

``` 
python3 imputeStream_wTFP.py --p 10 --tau 5; p: the numer of new instances, tau: retrospect length 
``` 

*baselines*

``` 
python3 imputeStream_CD.py --p 10; #CD
python3 imputeStream_mpin.py --p 10; #mpin
python3 imputeStream_MICE.py --p 10; #MICE
python3 imputeStream_MF.py --p 10; #MF
....

``` 
The performance results are stored in the folder: ./exp_results/. For other method-specific parameters (e.g., K value for KNN), they are tuned to a proper value. 

***Close the sensor data stream***: 
```
cd kafka; 
docker compose down;
```


### Explaination of Parameters

- dataset: sensor data stream dataset, e.g., Motion, Water, and Gas
- miss_ratio: the sparsity level of sensor data stream, e.g., 0.9
- tau: retrospect length
- p: the number of incremental/new instances



### Acknowledgements

We appreciate the work of SAITS, and their contributed codes available in [here](https://github.com/WenjieDu/SAITS).



