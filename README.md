# nnP
The parallella version of my feed forward - back propagation neural network engine.

Source files for the basic feed forward system based on allocating nodes to cores.

Before you compile this program, check the #defines at the top of the nn.hpp file to make sure that the PATHTOKERNELFILE and PATHTOCLDEFSFILE reflect where these files are on your system. They must be in the same directory.

The Code::Blocks cpb file is included but if you prefer the command line looks like this:

g++ -std=c++11 -Wall -fexceptions -g -g -I/usr/local/browndeer/include -I/home/linaro/Work/nnP -c /home/linaro/Work/nnP/nnP.cpp -o obj/Debug/nnP.o

g++ -L/usr/local/browndeer/lib -o bin/Debug/nnP obj/Debug/nnP.o -lstdcl -locl

Make sure that the paths are correct for your system.

PS This was not added by Andreas. I had not changed the author when I made my first commit.
