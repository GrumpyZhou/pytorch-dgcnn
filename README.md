# pytorch-dgcnn
Pytorch implementation of Dynamic Graph CNN for Learning on Point Clouds (EdgeConv)

### Setup Running Environment
It has been tested on Linux Ubuntu 16.04.6 with 
````
Python 3.7
Pytorch 1.1
CUDA 10.0
````
We recommend to use *Anaconda* to manage packages. Run following lines to automatically setup a ready environment for our code.
````
conda env create -f environment.yml
conda activte pyplc
````
Otherwise, one can try to download all required packages seperately according to their offical documentation.

### Classification
#### Training DGCNN for classification on ModelNet40 
````
# Classification  ModelNet10/ModelNet40
python -m classifier --train  --gpu 1 --val 5 --epoch 200 \
                     --dataset 'ModelNet40' \
                     --network 'DGCNNCls' --K 20 \
                     --batch 32 --worker 6 \
                     --lr 0.001 --weight_decay 0\
                     --lrd_factor 0.5 --lrd_step 20 \
                     --odir 'outputs' \
                     --visport 9333 --vishost 'localhost' --visenv 'main' --viswin 'DGCNNCls_ModelNet40.K20'
````
##### Visdom (optional)
As you see in the example above, we use Visdom server to visualize the training process. Make sure you have visdom.server runing under correct host and port. If you DON'T want it, just remove the last line `--visport ...  --viswin ...`.


#### Testing a trained classifier
````
python -m classifier --test  --gpu 1\
                     --dataset 'ModelNet40' \
                     --network 'DGCNNCls' --K 20 \
                     --odir 'outputs' \
                     --batch 32 --worker 6 \
                     --ckpt 'ckpt_to_test.pth'

````
### Segementation
#### Training DGCNN for semantic segementation on ShapeNet all categories
````
python -m segmenter  --train  --gpu 3 --val 5 --epoch 100 \
                     --dataset 'ShapeNet' --cat 'All' \
                     --network 'DGCNNSeg' --K 20 \
                     --odir 'outputs' \
                     --batch 16 --worker 6 \
                     --lr 0.001 --weight_decay 0\
                     --lrd_factor 0.5 --lrd_step 20\
                     --visport 9333 --vishost 'localhost' --visenv 'main' --viswin 'DGCNNSeg_ShapeNet.All.K20'
````                     

#### Testing a trained segmenter
````
python -m segmenter  --test  --gpu 2 \
                     --dataset 'ShapeNet' --cat 'All' \
                     --network 'DGCNNSeg' --K 20 \
                     --odir 'outputs' \
                     --batch 16 --worker 6 \
                     --ckpt 'ckpt_to_test.pth'
````

