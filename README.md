MOnte carlo code for QUIck proton dose calculation (moqui)
==========================================================

<img src="images/moqui_logo.jpg">

### Installation
#### Requirements
- GDCM (Please refer to GDCM v2 [installation guide](http://gdcm.sourceforge.net/wiki/index.php/Compilation#Quick_start)) (libgdcm-dev in Ubuntu 22)
    - If you get spurious warnings in CMake, and they annoy you, consider installing (Ubuntu 22): libgdcm-tools libvtkgdcm-cil libvtkgdcm-dev libvtkgdcm-java python3-vtkgdcm
- CUDA
- ZLIB
- The code has been tested with GDCM v2 and CUDA v10.2, as well as GDCM v3 and CUDA v8.0

#### Obtaining the code
```bash
$ git clone https://github.com/mghro/moquimc.git
```

#### Compile the phantom case
```bash
$ cd moquimc/tests/mc/phantom
$ mkdir build
$ cd build
$ cmake ..
$ make
```
- You can specify a custom CUDA path in the cmake command, for example: `-DCUDAToolkit_ROOT=/opt/cuda-8.0 -DCMAKE_CUDA_COMPILER=/opt/cuda-8.0/bin/nvcc`. The default is the nvcc found within thhe PATH environment variable.
- You can specify a custom CUDA compute capability via `-DCMAKE_CUDA_ARCHITECTURES=20`. The default is to use CUDA compute capability 7.5

#### Running the phantom example
```bash
$ python create_phantom.py # create water phantom
$ ./phantom_env --lxyz 100 100 350 --pxyz 0.0 0.0 -175 --nxyz 200 200 350 --spot_energy 200.0 0.0 --spot_position 0 0 0.5 --spot_size 30.0 30.0 --histories 100000 --phantom_path ./water_phantom.raw --output_prefix ./ --gpu_id 0 > ./log.out
```

### Authors
Hoyeon Lee    
Jungwook Shin  
Joost M. Verburg  
Mislav Bobić  
Brian Winey  
Jan Schuemann  
Harald Paganetti  

### Acknowledgements
This work is supported by NIH/NCI R01 234210 "Fast Individualized Delivery Adaptation in Proton Therapy"   


