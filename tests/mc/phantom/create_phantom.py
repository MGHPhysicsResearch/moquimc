import os, sys, glob
import numpy as np

nz = 350
ny = 200
nx = 200
ph_water = np.zeros([nz, ny, nx], dtype=np.int16)
ph_water.tofile('/tmp/water_phantom.raw', sep='', format='')
