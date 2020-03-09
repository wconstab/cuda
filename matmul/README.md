Install MKL
https://software.intel.com/en-us/articles/installing-intel-free-libs-and-python-apt-repo
followed instructions and installed intel-mkl-2020.0-088 2020.0-088

Install nsight-compute
made account on nvidia developer zone, downloaded 
nsight-compute-linux-2019.5.0.14-27346997.run
added makefile target with workaround for https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters#SolnAdminTag
running nv-nsight-cu-cli --mode=launch 
had to open inbound TCP ports 49152 - 49216 on AWS CLI
able to connect to local process launched this way using sudo -E, otherwise can't access perf ctr
