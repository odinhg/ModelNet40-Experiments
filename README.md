# ModelNet40 Experiments 

Learning on point cloud data using set and graph neural networks.

## TODO

### Implement
- [ ] Edge density function
- [ ] Delaunay GNN
- [ ] Set of Sets
- [ ] Graph of Sets
- [ ] Graph of Graphs

### Run experiments for:
- [ ] Delaunay GNN
- [ ] Set of Sets
- [ ] Graph of Sets
- [ ] Graph of Graphs

Run experiments with $m\in\{8, 16, 32, 64, 128\}$, $k\in\{\frac{1024}{m}, \frac{2048}{m}, \frac{4096}{m}\}$ and both with and without the edge density feature (where applicable). Results are added to the [following document](https://docs.google.com/spreadsheets/d/1OzJbgeIN__HjD_G7s68ooIhPvgB9MsWIcwAFCDZRGAc/edit?usp=sharing).

## Constructing graphs and sets

We now describe how the different graphs and sets are constructed for the different experiments.

Let $P\subseteq\mathbb{R}^3$ be a point cloud consisting of $10^4$ points uniformly sampled from a 3D model in ModelNet40. Let $m, k$ be positive integers and let $C$ be a farthest point sample of $P$ with $|C|=m$.

### Delaunay GNN

In this case, there is no $k$ as we construct the graph directly on the centroids $C$ in a single step. We let $m\in\{1024, 2048, 4096\}$ and construct the Delaunay graph on $C$ (the $1$-skeleton of the Delaunay triangulation of $C$). Node features are the coordinates of each point and edge features are as descibed below.

### Set of Sets

Here, we (uniformly, with replacement) sample $k$ points from each of the Voronoi cells $\operatorname{Vor}(c)$ where $c\in C$. The collection of such subsamples are represented by tensors of shape `(m, k, 3)`.

### Graph of Sets

Here, we do the same as for Set of Sets, but we include the Delaunay graph for the global graph with nodes $C$.

### Graph of Graphs

Here, we do the same as for Graph of Sets, but we also include the Delaunay graph constructed on each subsample. Note that we construct the subgraphs in the network itself and not the dataset class.

## Edge features

We include (up to) two edge features to be used for the graphs constructed.

### Distance feature

Given $p>0$, a point cloud $P$ and two points $c_1, c_2 \in C\subseteq P$, we consider the value $\frac{1}{(d(c_1, c_2)+1)^p}\in[0,1]$ as an edge feature.

### Edge density function

Given a point cloud $P$ and two points $c_1, c_2 \in C\subseteq P$, define the set 

$$
\operatorname{N}^C_2(c_1, c_2)=\{p\in P\mid\text{ the two points in }C\text{ closest to }p\text{ is }c_1\text
{ and }c_2\}.
$$

We define the *edge-density function* $\operatorname{ED}\colon C\times C\to[0,1]$ by
$$
\operatorname{ED}(c_1, c_2) = \frac{|\operatorname{N}^C_2(c_1, c_2)|}{|P|}.
$$

The edge density value is higher for pair of points having many points as their closest neighbours.

## Dataset instructions

We use the same dataset format (HDF5) as in the [original DeepSets implementation](https://github.com/manzilzaheer/DeepSets/tree/master/PointClouds#data).

Perform the following steps to compile the dataset:

1. Download ModelNet10 or ModelNet40 from [Princeton ModelNet](https://modelnet.cs.princeton.edu/)
2. Run `python generate_dataset.py <modelnet_dir> <output_dir> <n_points>`.

### Example usage:

If we have the following directory structure:

```
data/
    ModelNet40/
        desk/
            train/
            test/
        bed/
            train/
            test/
        ...
```

then running 
```python
python generate_dataset.py data/ModelNet40 data 10000
``` 
will save the HDF5 dataset consisting of point clouds with 10000 uniformly sampled points each to the file `data/ModelNet40_cloud.h5`.
