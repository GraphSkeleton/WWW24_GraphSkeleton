## Graph-Skeleton

This anonymous repository is for a code demo on dataset DGraph in  "Graph-Skeleton: 10% Node is Sufficient to Represent Massive Graph Data" (NeurIPS'23)



### Install

Install libboost

```shell
sudo apt-get install libboost-all-dev
```

### Compile

Compile Graph-Skeleton

```shell
mkdir build
cd build
cmake ..
make
cd ..
```

### Data Download

The currently code demo is based on the dataset [DGraph](https://dgraph.xinye.com/dataset). Please unizp the dataset folder and organize as follows:
```
.
--DGraphFin
   └─dgraphfin.npz
```

### GraphS-Skeleton Generation
To generate skeleton graphs, a graph compressio script is provided. Please note that in our original paper, hyper-parameters "d1", "d2" are set as 2 and 1, you can also modify the setting of "d" in the script to change the node fetching distance. This script will generate three different sekeleton graphs (i.e., $\alpha$, $\beta$ and $\gamma$).

```
python xinye_compression.py
```

### GNN Deployment
We provide a GraphSAGE training \& evaluation pipline as the GNN deployment demo. hyper-parameters "cut: no, zip" indicates deployment on origianl graph or skeleton graph.
Please specify the "file_path" with the corrsponding original or skeleton graph data path in `dgraph_sage.py`, you can also modify other parameters for training in the file or cmd:

```
python dgraph_sage.py --cut no --batch-size 65536 --lr 0.005 --epoch 200 --num-layers 3 --iter 10 # deployment on original graph
python dgraph_sage.py --cut zip --batch-size 65536 --lr 0.005 --epoch 200 --num-layers 3  --iter 10 # deployment on skeleton graph
```

### Demos on more datasets and models are coming soon!


