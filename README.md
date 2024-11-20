MOnte carlo code for QUIck proton dose calculation (moqui)
==========================================================

<img src="images/moqui_logo.jpg">

### Installation
#### Requirements
- GDCM > 2
  - Please refer to GDCM [installation guide](https://sourceforge.net/projects/gdcm/)
  - You can also install GDCM v3 using package manager (`libgdcm-dev` in Ubuntu 22)
    - If you get spurious warnings in CMake, and they annoy you, consider installing (Ubuntu 22): `libgdcm-tools libvtkgdcm-cil libvtkgdcm-dev libvtkgdcm-java python3-vtkgdcm`
- CUDA
- The code has been tested with
  - GDCM v2 and CUDA v10.2 on Ubuntu 20.04
  - GDCM v3 and CUDA v11.8 on Ubuntu 22.04
  - GDCM v3 and CUDA v8.0 on Ubuntu 22.04
- ZLIB
- Python3 (for phantom example)

#### Obtaining the code
```bash
$ git clone https://github.com/ferdymercury/moquimc.git
```

#### Compile the phantom case
```bash
$ cd moquimc
$ mkdir build
$ cd build
$ cmake ..
$ make
```
- You can specify a custom CUDA path in the cmake command, for example: `-DCUDAToolkit_ROOT=/opt/cuda-8.0 -DCMAKE_CUDA_COMPILER=/opt/cuda-8.0/bin/nvcc`. The default is the nvcc found within thhe PATH environment variable.
- You can specify a custom CUDA compute capability via `-DCMAKE_CUDA_ARCHITECTURES=20`. The default is to use CUDA compute capability 7.5

#### Running the phantom example
```bash
$ python ../tests/mc/phantom folder/create_phantom.py # create water phantom in /tmp/, you need to install numpy
$ ./tests/mc/phantom/phantom_env --lxyz 100 100 350 --pxyz 0.0 0.0 -175 --nxyz 200 200 350 --spot_energy 200.0 0.0 --spot_position 0 0 0.5 --spot_size 30.0 30.0 --histories 100000 --phantom_path /tmp/water_phantom.raw --output_prefix ./ --gpu_id 0 > ./log.out
```

Or simply:
```bash
$ ctest -V -R phantom_env
```

#### Update Dec/26/2023
- There have been large updates in moqui and we added new features
  - Statistical uncertainty based stopping criteria (Please refer to the example input parameter *Statistical stopping criteria*)
  - Robust options (setup errors and density scaling, Please refer to the example input parameter *Robust options*)
  - Aperture handling
  - Support multiple calibration curves (You can override machine selection and define multiple calibration curves for a machine)
  - Unit weights per spot for Dij calculation (This only works for Dij scorer. The *UnitWeight* will be the absolute number of particles simulated)

#### Getting calibration curves
- moqui uses fitted functions for calibration curves
- You need to obtain stopping power ratio to water and radiation length per density and define *compute_rsp_* and *compute_rl_* functions in patient_material_t
- To obtain the curves:
  1. Obtain material information using TOPAS
  2. Calculate correction factors for desired SPR curve
  3. Calculate fitting curves and implement them in moqui
  4. You can refer to the fit_rsp.py for the curve fitting
- You can find the TOPAS extensions and example parameter file under treatment_machines/TOPAS
- These are updated version of the HU extension in TOPAS (https://github.com/topasmc/extensions/tree/master/HU)


### Authors
Hoyeon Lee (leehoy12345@gmail.com)  
Jungwook Shin  
Joost M. Verburg  
Mislav Bobić  
Brian Winey  
Jan Schuemann  
Harald Paganetti  

### Acknowledgements
This work is supported by NIH/NCI R01 234210 "Fast Individualized Delivery Adaptation in Proton Therapy"   


