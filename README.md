# SBIQUE
Score Based Image Quality Enhancement

### Compiling command with g++
```
g++ -ggdb main.cpp brisque.cpp metrics.cpp utils.cpp sharpen.cpp resultData.cpp libsvm/svm.cpp -lstdc++fs -std=c++17 -o exec.out `pkg-config --cflags --libs opencv`
```
### OpenCV
The easiest way to get opencv is to run the following command
```
sudo apt-get install libopencv-dev
```
