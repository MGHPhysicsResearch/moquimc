MOnte carlo code for QUIck proton dose calculation (moqui)
=======

<img src="images/moqui_logo.jpg">

### Installation
#### Requirements
* GDCM > 2
  * Please refer to GDCM v2 [installation guide](https://sourceforge.net/projects/gdcm/)
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
#### Update Dec/26/2023
- There have been large updates in moqui and we added new features
  - Statistical uncertainty based stopping criteria (Please refer to the example input parameter *Statistical stopping criteria*)
  - Robust options (setup errors and density scaling, Please refer to the example input parameter *Robust options*)
  - Aperture handling
  - Support multiple calibration curves (You can override machine selection and define multiple calibration curves for a machine)
  - Unit weights per spot for Dij calculation (This only works for Dij scorer. The *UnitWeight* will be the absolute number of particles simulated)

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


