# nnP
This fork is to freeze the files referred to in my blog post titled, "Getting it Done - What I learnt from finishing the Neural Network Algorithm". 

It is not a finished system as such. Development will continue in the master branch.

If you want to download it and get it running the t3.nn network int /testData is the one that I've been using to test it. The t3t.sh file contains the shell commands to load, train and save the network. The cldefs.inc file contain the parameters needed by the kernels to run this network.

To run your own network with a different topology you must change the code in the train() function to use JIT compilation in nn.hpp (comment out line 347 and uncomment line 350). The run() function is currently set to use JIT compilation (see line 186).  If you wish to stick to the same topology for a while you can change the train() and run() functions back to linked kernels once the cldefs.inc file has been written. Make sure you use the correct syntax in the data and training files (see the master branch wiki).

Please direct any feedback to me via the parallella forum or via the blog post on Google Blogger.

nick
