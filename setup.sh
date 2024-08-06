export LD_LIBRARY_PATH=/apps/ault/spack/opt/spack/linux-centos8-zen/gcc-8.4.1/cuda-11.7.1-ivzet6dv3q5x2sgmqypy2nge2s34a3ah/lib64/:$LD_LIBRARY_PATH
spack load py-pip
spack load cuda@11
spack load nvhpc
spack load cudnn@8
