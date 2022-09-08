# Convolution Benchmarks
![image](https://user-images.githubusercontent.com/74956/188198140-b109f11f-f9e6-4664-81da-2b2bc754435b.png)
# Pre-requisites
Create a virtual environment and install python dependencies.
```
python3 -m venv ~/venv/convperf
source ~/venv/convperf/bin/activate
pip install -r requirements.txt
```
For multi-threaded libxsmm, you will need to install OpenMP. You will have to install
the appropriate package for your compiler. For example, if your compiler is clang++-14,
then you can install OpenMP by doing the following.
```
sudo apt install libomp-14-dev
```
# Build Instructions
```
cmake -GNinja -B build .
cmake --build build
```
# Run Benchmarks
```
python convperf.py --benchmark_tool build/tools/benchmark_conv --runners iree,xsmm
```
