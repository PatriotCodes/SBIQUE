# SBIQUE
Score Based Image Quality Enhancement

### Compiling command with g++
```
g++ -ggdb main.cpp brisque.cpp metrics.cpp libsvm/svm.cpp -lstdc++fs -std=c++17 -o exec.out `pkg-config --cflags --libs opencv`
```
