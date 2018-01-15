### Installation (sufficient for the Dual-MFA-VQA)
#### 1. Clone the Faster R-CNN repository
``` 
cd
git clone --recursive https://github.com/rbgirshick/py-faster-rcnn.git
```

#### 2. Build the Cython modules
``` 
cd py-faster-rcnn/lib
make
```

#### 3. Build Caffe 

- Install caffe libraries
``` 
cd py-faster-rcnn/caffe-fast-rcnn
sudo apt-get install libatlas-base-dev
sudo apt-get install libprotobuf-dev
sudo apt-get install libleveldb-dev
sudo apt-get install libsnappy-dev
sudo apt-get install libopencv-dev
sudo apt-get install libboost-all-dev
sudo apt-get install libhdf5-serial-dev
sudo apt-get install libgflags-dev
sudo apt-get install libgoogle-glog-dev
sudo apt-get install liblmdb-dev
sudo apt-get install protobuf-compiler
```

- Update `Makefile.config` by uncommenting two lines
``` 
sudo cp Makefile.config.example Makefile.config
vim Makefile.config
# In your Makefile.config, make sure to have this line uncommented
WITH_PYTHON_LAYER := 1
# Unrelatedly, it's also recommended that you use CUDNN
USE_CUDNN := 1
```
- Build Caffe
``` 
sudo make all
```


#### 4. Build pycaffe
``` 
sudo make -j8 && sudo make pycaffe
```

### Test demo (Optional)
``` 
cd $FRCN_ROOT
./tools/demo.py
```


### Download pre‐computed Faster R‐CNN detectors
``` 
cd py-faster-rcnn
. ./data/scripts/fetch_faster_rcnn_models.sh
```





### Main Problems
#### cuDNN v5 isn't supported
- If you meet following error when build caffe:
``` 
Makefile:563: recipe for target '.build_release/src/caffe/blob.o' failed
```
- It is because cuDNN v5 isn't supported unofficially. Just following

``` 
cd caffe-fast-rcnn
git remote add caffe https://github.com/BVLC/caffe.git
git fetch caffe
git merge -X theirs caffe/master
# then quit with ctrl-X
```
- And update `Makefile.config` for two lines
``` 
# vim Makefile.config
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial 
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial
```

- Build Caffe again
``` 
sudo make all
```


#### AttributeError: can't set attribute

- If you meet following error when running demo:
``` 
Traceback (most recent call last):
File "./tools/demo.py", line 135, in <module>
net = caffe.Net(prototxt, caffemodel, caffe.TEST)
AttributeError: can't set attribute
```
- Comment one line in `python_layer.hpp` and rebuild
``` 
cd ~/py-faster-rcnn
cd caffe-fast-rcnn/include/caffe/layers
vim python_layer.hpp
# self_.attr("phase") = static_cast(this->phase_);
cd caffe-fast-rcnn
sudo make -j8 && sudo make pycaffe
```


#### GUI error
- If you meet the GUI error when running demo, you can add two lines in `demo.py` as following 
``` 
import _init_paths
import matplotlib
matplotlib.use('Agg')
from fast_rcnn.config import cfg
.................
```

- then add one line `demo.py/vis_detections()` as following 
``` 
def vis_detections(im, class_name, dets, thresh=0.5):
    .................
    plt.savefig("../picFaster.jpg")
```
