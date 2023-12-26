MOnte carlo code for QUIck proton dose calculation (moqui)
=======

<img src="images/moqui_logo.jpg">

### Installation
#### Requirements
* GDCM > 2
  * Please refer to GDCM v2 [installation guide](http://gdcm.sourceforge.net/wiki/index.php/Compilation#Quick_start)
  * You can also install GDCM v3 using package manager
* CUDA
- The code has been tested with GDCM v2 and CUDA v10.2 on Ubuntu 20.04
- The code has been tested with GDCM v3 and CUDA v11.8 on Ubuntu 22.04

#### Obtaining the code
```bash
$ git clone https://github.com/mghro/moquimc.git
```

#### Compile the phantom case
```bash
$ cd moquimc/tests/mc/phantom
$ cmake .
$ make
```
- You may need to modify the CUDA configuration on the CMakeList.txt
- The default is to use CUDA compute capability 7.5

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


