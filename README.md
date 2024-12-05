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
  - GDCM v3 and CUDA v12.6 on Alma Linux 9
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
$ python3 ../tests/mc/phantom/create_phantom.py # create water phantom in /tmp/, you need to install numpy
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


### Notes
You might need for old Tesla C2070 commands such as:
- Install patched nvidia-390 driver on Ubuntu 22: https://launchpad.net/%7Edtl131/+archive/ubuntu/nvidiaexp
- Install gcc5 and cuda8: https://askubuntu.com/questions/1442001/cuda-8-and-gcc-5-on-ubuntu-22-04-for-tesla-c2070
- Error with stncpy: https://stackoverflow.com/questions/76531467/nvcc-cuda8-gcc-5-3-no-longer-compiles-with-o1-on-ubuntu-22-04
- Error with float128: https://askubuntu.com/questions/1442001/cuda-8-and-gcc-5-on-ubuntu-22-04-for-tesla-c2070
- `cmake ../ -DCUDAToolkit_ROOT=/opt/cuda-8.0 -DCMAKE_CUDA_COMPILER=/opt/cuda-8.0/bin/nvcc -DCMAKE_C_COMPILER=/opt/gcc5/gcc -DCMAKE_CXX_COMPILER=/opt/gcc5/g++ -DCMAKE_CUDA_ARCHITECTURES=20`
- This might also be needed depending on the platform or CMake version: `export PATH=/opt/gcc5:$PATH`
- Need to fine-tune QtCreator adding a new custom compiler /opt/cuda-8.0/bin/nvcc and edit .config/clangd/config.yaml file with
```
CompileFlags:
Add:
  [
    '--cuda-path="/opt/cuda-8.0/"',
    --cuda-gpu-arch=sm_20,
    '-L"/opt/cuda-8.0/lib64/"',
    -lcudart,
  ]
```
- See https://github.com/clangd/clangd/issues/858 and https://github.com/clangd/clangd/issues/1815

For an Ampere GPU NVIDIA A40:
- `cmake -DCUDAToolkit_ROOT=/usr/local/cuda-12.6 -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc`

### Acknowledgements
This work is supported by NIH/NCI R01 234210 "Fast Individualized Delivery Adaptation in Proton Therapy"


