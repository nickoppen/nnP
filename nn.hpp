#ifndef _nn_h
#define _nn_h

#define ennVersion "(1,2,0)"
#define PATHTOKERNALFILE "//home//parallella//Work//nnP//nn.cl"
#define PATHTOCLDEFSFILE "//home//parallella//Work//nnP//cldefs.inc"

#define CORECOUNT 16

using namespace std ;

#include <sys/stat.h> // POSIX only

#include <sstream>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <time.h>   // only used for the random seed generator
#include "nnFile.hpp"

#include <stdcl.h>

const int maxNodes = 64;
const int maxLayers = 3;	// no longer used

enum layer_modifier { BIAS_NODE, TRANSITION_SIGMOID, TRANSITION_LINEAR, TRANSITION_BINARY };
enum node_modifier { INPUT_BINARY, INPUT_UNIFORM, INPUT_BIPOLAR };

struct nodeData
{
  public:
	node_modifier inputType;        /// not used right now other than in randomise()
	float p;			            /// P(x=1) = p
	bool pIsOneHalf;
	float nodeValue;
//	float bias;
//	vector<float> * incomingWeights;
};

struct layerData
{
  public:
	unsigned int nodeCount;
	bool hasBiasNode;
	layer_modifier transition;
	vector<nodeData> * nodeInfo;
};


typedef void (*funcRunCallback)(const int, void *);
typedef void (*funcTrainCallback)(void *);
typedef void (*funcTestCallback)(const int, vector<float>*, vector<float>*, vector<float>*, vector<float>*, void *);


class nn
{
	public:
						nn(int inputLayerWidth, int hiddenLayerWidth, int outputLayerWidth, string & newName, float learningRateParam = 0.1)
						/*
						 * Create a new network with
						 * inputLayerWidth input nodes,
						 * hiddenLayerWidth hidden nodes,
						 * outputLayerWidth output nodes and
						 * learningRateParam as the learning rate with
						 * newName as the network name
						 */
                        {
							vector<unsigned int> widths(3);		// only while the number of layers is restricted to 3
							widths[0] = inputLayerWidth;
							widths[1] = hiddenLayerWidth;
							widths[2] = outputLayerWidth;

							setNetworkTopology(&widths);
							networkName = newName;
							clLearningRate = (cl_float)learningRateParam;

                            randomise();
                            majorVersion = minorVersion = revision = 0;
                            networkName = newName;

                        }

                        nn(vector<unsigned int>* networkTopo, string & newName, float learningRate = 0.1)
                        {

							setNetworkTopology(networkTopo);
							networkName = newName;
							clLearningRate = (cl_float)learningRate;

                            randomise();
                            majorVersion = minorVersion = revision = 0;
                            networkName = newName;

                        }

                        // nn(int layerCount, int* layerWidths, float learningRateParam, string & newName)

                        nn(NNFile * newFile)
                        /*
                         * Reconstruct a network from a saved file with the wrapper newFile
                         */
                        {
                        	newFile->readInFile((void*)this);
                        };

                        ~nn()
                        /*
                         * The network object destructor.
                         *
                         * ALWAYS make sure you call this function.
                         *
                         */
                        {
                            int i;
                            unsigned int j;

                            if (clInputLayer)   clfree((void*)clInputLayer);
                            if (clOutputLayer)  clfree((void*)clOutputLayer);
                            if (clNodeBiases)   clfree((void*)clNodeBiases);
                            if (clWeights)      clfree((void*)clWeights);
                            if (clLayerWidths)  clfree((void*)clLayerWidths);
                            if (clOutputError)  clfree((void*)clOutputError);

 // not not a pointer                       	delete errorVector;
                        	for (i=0; i<layerCount; i++)
                        	{
                                for (j=0; j<(*layers)[i].nodeCount; i++)
                                    delete (*layers)[i].nodeInfo;
                        	}
                        	delete layers;

                        }


	// operation
			void		run(NNFile * inFile, funcRunCallback runComplete = NULL)
			/*
			 * Run the contents of the data file wrapped by inFile calling the runComplete callback
			 * for each line.
			 *
			 *	typedef void (*funcRunCallback)(const int index, void * thisNetwork);
			 *	index is the index of the row that has just been run
			 *	theNetwork is a void pointer to this object.
			 *
			 *	Call ((nn*)theNetwork)->runResult(vector<float>* existingVector) to retrieve the result
			 *
			 */
                        {
                            runCallback = runComplete;      // Cannot pass callback via readInFile yet
							inFile->readInFile((void*)this);
                        }

            void		run(vector<float> * inputVector, funcRunCallback runComplete = NULL, const int index = 0)
            /*
             * Pass inputVector to the input layer and trigger it to execute the network logic. Call the
             * runComplete callback if it is not NULL.
             *
			 *	typedef void (*funcRunCallback)(const int index, void * thisNetwork);
			 *	index is the index of the row that has just been run
			 *	theNetwork is a void pointer to this object.
			 *
			 *	Call ((nn*)theNetwork)->runResult(vector<float>* existingVector) to retrieve the result
			 *
			 * Note: this version is not multi threaded so waitForActivation and blockTillValue do nothing
             */
                        {
            				unsigned int i;
            				void * openHandle;
            				cl_kernel krn;
            				clndrange_t ndr;
//            				char strInfo[128];
//            				CONTEXT * pCon = stdcpu;        // the cpu context !! all the storage exists in the atdacc context so this will not work as is
            				CONTEXT * pCon = stdacc;

            				cl_float * clDebug;
            				clDebug = (cl_float*)clmalloc(pCon, 2048*sizeof(float), 0);
            				for(i=0;i<2048;i++) clDebug[i]=-1000;


            				for (i=0; i < (*layers)[0].nodeCount; i++)
                                clInputLayer[i] = (*inputVector)[i];

///                            openHandle = clopen(pCon, 0, CLLD_NOW);                  /// linked in version - the elf file must be linked into the executable at link time

                            writeDefsFile();
                            openHandle = clopen(pCon, PATHTOKERNALFILE, CLLD_NOW);      /// JIT compile from file version

///                            appendDefsToKernalString(); //TODO
///                            openHandle = clsopen(pCon, str_k_forward, CLLD_NOW);     /// string version (not done yet)

                            ///  Get the handle to the kernel
                            krn = clsym(pCon, openHandle, "k_forward", CLLD_NOW);

//                            clGetKernelInfo(krn, CL_KERNEL_FUNCTION_NAME, sizeof(strInfo), strInfo, NULL);

                            ndr = clndrange_init1d(0, 16, 16);      // get the core count from a cl call

                            /// transfer the inputdata biases and wieghts to the acc using clsync(,,, C_MEM_DEVICE|CL_EVENT_NOWAIT)
                            clmsync(pCon, 0, clOutputLayer, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
                            clmsync(pCon, 0, clInputLayer, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
                            clmsync(pCon, 0, clNodeBiases, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
                            clmsync(pCon, 0, clWeights, CL_MEM_DEVICE|CL_EVENT_NOWAIT);

                            clmsync(pCon, 0, clDebug, CL_MEM_DEVICE|CL_EVENT_NOWAIT);

   //cout <<   "Calling clforka\n";
                            clforka(pCon, 0, krn, &ndr, CL_EVENT_NOWAIT,
                                        clInputLayer,
                                        clNodeBiases,
                                        clWeights,
                                        clOutputLayer,
                                        clDebug);

   //cout <<   "Transferring memory contents from the Epiphany using clmsync\n";
                            clmsync(pCon, 0, clOutputLayer, CL_MEM_HOST|CL_EVENT_NOWAIT);
                            clmsync(pCon, 0, clDebug, CL_MEM_HOST|CL_EVENT_NOWAIT);
                            clflush(pCon, 0, 0);
                            clwait(pCon, 0, CL_ALL_EVENT);

/// test
                            i=0;
                            if (clDebug[i] >= -1000)           /// if we have put anything in the debug buffer
                            {
                                filebuf fbuf;
                                fbuf.open(".//nn.csv", std::ios::out);
                                ostream fout(&fbuf);
                                fout.precision(12);

                                while ((clDebug[i] > -999) && (i<2048))
                                {
                                    if (clDebug[i] > 999)
                                        fout << "\n";
                                    else
                                        fout  << clDebug[i] << ",";
                                    i++;
                                }
                                fout.flush();
                                fbuf.close();
                            }

                            if (runComplete != NULL)
                            {
                                runComplete(index, (void*)this);
                            }
                            else
                            {
                                    if (runCallback != NULL)    // this is the storage var for call backs that have been passed in when running from a file (saves having to pass the callback via networkFile class)
                                    {
                                        runCallback(index, (void*)this);
                                    }
                            }

                        }

            void		run(vector<float> * inputVector, vector<float> * outputVector)
            /*
             * Run a single input vector and return the result. This call is designed to
             * run synchronously.
             */
                        {
                            run(inputVector);
                            // wait for the result
                            runResult(outputVector);
                        }

            vector<float> * runResult(vector<float> * outputVector)
			/*
			 * Set and return outputVector from the last run.
			 *
			 * Call this quickly - I'm not sure how long it will be before the result is written
			 * over by the next output.
			 *
			 */
                        {
            				unsigned int outI;
//            				float fl;

            				for (outI = 0;  outI < layerNWidth(); outI++)
                            {

  //          					(*outputVector)[outI] = (*layers)[2].nodeInfo->operator[](outI).nodeValue;	// copy the contents
                                (*outputVector)[outI] = clOutputLayer[outI];
  //                              fl = clOutputLayer[outI];
                            }
                            return outputVector;
                        }

            void 		train(NNFile * trFile, funcTrainCallback trComplete = NULL)
            /*
             * Train the network using the training set in the file wrapped by trFile. Call the trComplete callback once
             * when training is complete.
             *
             * typedef void (*funcTrainCallback)(void * nnObj); passes an anomymous pointer to this object back via the callback
             *
             * call ((nn*)nnObj)->trainingError(vector<float>* existingVector); to retrieve the most recent training error vector
             *
             */
                        {
                            // call train with each vector
//                            unsigned int i;
//
//                            for (i=0; i < trFile->inputLines(); i++)
//                                train(trFile->inputSet(i), trFile->outputSet(i));	// don't pass the call back because we only want it called at the end not after each training set

            				trFile->readInFile((void*)this, true);

            				// block til complete
                            incrementRevision();

                            if (trComplete != NULL)
                                trComplete((void*)this);
                        }

            void		train(vector<float> * inputVector, vector<float> * desiredVector, funcTrainCallback trComplete = NULL)
            /*
             * Train the network with the single pair, inputVector and desiredVector. Call the trComplete callback if it is not NULL
             * when training is complete.
             *
             * typedef void (*funcTrainCallback)(void * nnObj); passes an anomymous pointer to this object back via the callback
             *
             * call ((nn*)nnObj)->trainingError(vector<float>* existingVector); to retrieve the most recent training error vector
             *
             */
                        {
                            try
                            {
                                unsigned int i;
                                void * openHandle;
                                cl_kernel krn;
                                clndrange_t ndr;
    //            				char strInfo[128];
    //            				CONTEXT * pCon = stdcpu;        // the cpu context !! all the storage exists in the atdacc context so this will not work as is
                                CONTEXT * pCon = stdacc;
                                cl_float * clDesiredOutput = (cl_float*)clmalloc(pCon, desiredVector->size() * sizeof(float), 0);

                                cl_float * clDebug;
                                clDebug = (cl_float*)clmalloc(pCon, 2048*sizeof(float), 0);
                                for(i=0;i<2048;i++) clDebug[i]=-1000;
                                clWeightDeltas = (cl_float*)clmalloc(pCon, totalWeights*sizeof(float), 0);      // temporary: space for core's to share incoming weight deltas


                                for (i=0; i < (*layers)[0].nodeCount; i++)
                                    clInputLayer[i] = (*inputVector)[i];
                                for (i=0; i < (*layers)[layerCount-1].nodeCount; i++)
                                    clDesiredOutput[i] = (*desiredVector)[i];

                                openHandle = clopen(pCon, 0, CLLD_NOW);                  /// linked in version - the elf file must be linked into the executable at link time

                                writeDefsFile();
///                                openHandle = clopen(pCon, PATHTOKERNALFILE, CLLD_NOW);      /// JIT compile from file version

    ///                            appendDefsToKernalString(); //TODO
    ///                            openHandle = clsopen(pCon, str_k_forward, CLLD_NOW);     /// string version (not done yet)

                                ///  Get the handle to the kernel
                                krn = clsym(pCon, openHandle, "k_train", CLLD_NOW);

    //                            clGetKernelInfo(krn, CL_KERNEL_FUNCTION_NAME, sizeof(strInfo), strInfo, NULL);

                                ndr = clndrange_init1d(0, 16, 16);      // get the core count from a cl call

                                /// transfer the inputdata biases and wieghts to the acc using clsync(,,, C_MEM_DEVICE|CL_EVENT_NOWAIT)
                                clmsync(pCon, 0, clInputLayer, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
                                clmsync(pCon, 0, clDesiredOutput, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
                                clmsync(pCon, 0, clNodeBiases, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
                                clmsync(pCon, 0, clWeights, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
                                clmsync(pCon, 0, clOutputError, CL_MEM_DEVICE|CL_EVENT_NOWAIT);     /// not sure if I have to sync this one here

                                clmsync(pCon, 0, clDebug, CL_MEM_DEVICE|CL_EVENT_NOWAIT);
                                clmsync(pCon, 0, clWeightDeltas, CL_MEM_DEVICE|CL_EVENT_NOWAIT);

       //cout <<   "Calling clforka\n";
                                clforka(pCon, 0, krn, &ndr, CL_EVENT_NOWAIT,
                                            clInputLayer,
                                            clDesiredOutput,
                                            clNodeBiases,
                                            clWeights,
                                            clOutputError,
                                            clLearningRate,
                                            clWeightDeltas,
                                            clDebug);

       //cout <<   "Transferring memory contents from the Epiphany using clmsync\n";
                                clmsync(pCon, 0, clOutputError, CL_MEM_HOST|CL_EVENT_NOWAIT);   /// The final output error
                                clmsync(pCon, 0, clWeights, CL_MEM_HOST|CL_EVENT_NOWAIT);       /// The modified weights

                                clmsync(pCon, 0, clWeightDeltas, CL_MEM_HOST|CL_EVENT_NOWAIT);  // testing
                                clmsync(pCon, 0, clDebug, CL_MEM_HOST|CL_EVENT_NOWAIT);         // testing
                                clflush(pCon, 0, 0);
                                clwait(pCon, 0, CL_ALL_EVENT);

    /// test
                                i=0;
                                if (clDebug[i] >= -1000)           /// if we have put anything in the debug buffer
                                {
                                    filebuf fbuf;
                                    fbuf.open(".//nn.csv", std::ios::out);
                                    ostream fout(&fbuf);
                                    fout.precision(12);

                                    while ((clDebug[i] > -999) && (i<2048))
                                    {
                                        if (clDebug[i] > 999)
                                        {
                                            fout << clDebug[i];
                                            fout << "\n";
                                        }
                                        else
                                            fout  << clDebug[i] << ",";
                                        i++;
                                    }
                                    fout.flush();
                                    fbuf.close();
                                }


                                    if (trComplete != NULL)
                                        trComplete((void*)this);
                            }
                            catch (internal_Error & iErr)
                            {
                                cout << iErr.mesg;// << " last error:" << iErr.lastError;
                            }

                            hasChanged = true;
                        }

            status_t	trainingError(vector<float> * errorVector)
            /*
             * Return the most recent error vector generated by the most recent training set.
             *
             * Note: the errorVector must exist and be the right size
             *
             */
						{
                            unsigned int i;
                            unsigned int outputNodeCount = layerNWidth();

            				if (errorVector->size() != outputNodeCount)
            					return FAILURE;
            				else
                                for (i = 0; i < outputNodeCount; i++)
                                    (*errorVector)[i] = clOutputError[i];
							return SUCCESS;
						}

            void		test(NNFile * testFile, funcTestCallback testComplete = NULL)
            /*
             * Run the data component of the training file inside the wrapper testFile and
             * compare the output generated by the network to the desired output. Calculate the difference.
             *
             * The call back function funcTestCallback is called once for every line in the input file and has the following form:
             *
             * typedef void (*funcTestCallback)(const int index, vector<float>* inputVector, vector<float>* desiredOutput, vector<float>* outputVector, vector<float>* errorVector, void * thisObject);
             * index: the row number in the file
             * inputVector: a pointer to the test data vector
             * desiredOutput: a pointer to the desired output vector
             * outputVector: a pointer to the actual output from the net
             * errorVector: a pointer to a vector containing the desired minus the actual output
             * thisObject: an anonymous pointer to this object
             *
             */
						{

            				testFile->readInFile((void*)this, false);	// false to indicate that the file is NOT a training file

						}

            void		test(const int index, vector<float> * inputVector, vector<float> * desiredOutput, funcTestCallback testComplete = NULL)
            /*
             * Test a single input vector and compare the result with the givine output vector. Then
             * compare the output generated by the network to the desired output. Calculate the difference.
             *
             * The call back function funcTestCallback is called once and has the following form:
             *
             * typedef void (*funcTestCallback)(const int index, vector<float>* inputVector, vector<float>* desiredOutput, vector<float>* outputVector, vector<float>* errorVector, void * thisObject);
             * index: the row number in the file
             * inputVector: a pointer to the test data vector
             * desiredOutput: a pointer to the desired output vector
             * outputVector: a pointer to the actual output from the net
             * errorVector: a pointer to a vector containing the desired minus the actual output
             * thisObject: an anonymous pointer to this object
             *
             */
						{
            				size_t i;
            				vector<float> outputVec((*layers)[2].nodeCount);

            				// run
							run(inputVector);

							// block til value

							// compare
//							theOutputLayer->returnOutputVector(&outputVec);

							for (i = 0; i != errorVector.size(); i++)
								errorVector[i] = outputVec[i] - (*desiredOutput)[i];

							if (testComplete != NULL)
								testComplete(index, inputVector, desiredOutput, &outputVec, &errorVector, (void*)this);
						}

            void		randomise()
            /*
             * Randomise the weights and biases in the network thereby restarting the training cycle from a different place.
             */
						{
							unsigned int layer;
							float linkWeightVectorLength;
							unsigned int faninToNode;   // the number of incoming links
                            unsigned int firstLink, lastLink;
							unsigned int linkIndex, nodeI;
							unsigned int nodeIndex = 0;
                            int newRand;
                            float numerator, denominator;
                            float weight, weightMax;
                           	node_modifier input_type;
                           	bool pEqualsOneHalf;
                           	float p;


                            srand(time(NULL));

							for (layer = 1; layer < (unsigned int)layerCount; layer++)
							{
                                // set the weight to a random number
                                // pEqualsOneHalf == true assumes p == 0.5
                                // for uniform inputs p is the upper most positive value expected

                                faninToNode = (*layers)[layer-1].nodeCount;

                                for (nodeI=0; nodeI<(*layers)[layer].nodeCount; nodeI++)
                                {
                                    input_type = (*layers)[layer].nodeInfo->operator[](nodeI).inputType;
                                    pEqualsOneHalf = (*layers)[layer].nodeInfo->operator[](nodeI).pIsOneHalf;
                                    p = (*layers)[layer].nodeInfo->operator[](nodeI).p;

                                    if ((input_type == INPUT_BINARY) && pEqualsOneHalf)
                                    {
                                        numerator = (float)5.1;
                                        denominator = sqrt((float)faninToNode);
                                    }
                                    else if ((input_type == INPUT_BINARY) && !pEqualsOneHalf)
                                    {
                                        numerator = (float)2.55;
                                        denominator = sqrt((float)faninToNode * p * (1 - p));
                                    }
                                    else if ((input_type == INPUT_BIPOLAR) && pEqualsOneHalf)
                                    {
                                        numerator = (float)2.55;
                                        denominator = sqrt((float)faninToNode);
                                    }
                                    else if ((input_type == INPUT_BIPOLAR) && !pEqualsOneHalf)
                                    {
                                        numerator = (float)1.28;
                                        denominator = sqrt((float)faninToNode * p * (1 - p));
                                    }
                                    else if (input_type == INPUT_UNIFORM)
                                    {
                                        numerator = (float)4.4;
                                        denominator = p * sqrt((float)faninToNode);
                                    }
                                    else
                                    {
                                        throw;	// opps!
                                    }

                                    weightMax = numerator / denominator;

                                    firstLink = clNodeWeightIndex[nodeIndex];       // the index into the weight array where this node's links start
                                    if (((cl_int)layer == (layerCount - 1)) && (nodeI == (*layers)[layer].nodeCount) - 1)       //ie this is the verly lasy output nnode
                                        lastLink = totalWeights;
                                    else
                                        lastLink = clNodeWeightIndex[nodeIndex+1];       // the index into the weight array where the next node's links start
                                    linkWeightVectorLength = 0.0;

                                    for(linkIndex=firstLink; linkIndex<lastLink; linkIndex++)
                                    {
                                        newRand = rand();
                                        weight = (weightMax - ((float)newRand / ((float)RAND_MAX / (2 * weightMax))));
                                        linkWeightVectorLength += weight * weight;
                                        clWeights[linkIndex] = weight;
                                    }

                                    newRand = rand();
                                    linkWeightVectorLength = sqrt(linkWeightVectorLength);
                                    clNodeBiases[nodeIndex++] = linkWeightVectorLength - ((float)newRand / ((float)RAND_MAX / (2 * linkWeightVectorLength)));
                                }
							}

							hasChanged = true;
							incrementMinorVersion();
						}

	// access

			status_t 	saveTo(string * strPath)
			/*
			 * Save the network to a file called <network Name>_<majorVersion>_<minorVersion>_<revision>.enn in the path supplied in string object strPath.
			 *
			 * Note: if you have the path name already as a C string call saveTo(const char *) rather than this function
			 *
			 */
			{
				return saveTo(strPath->c_str());
			}

			status_t	saveTo(const char * cstrPath)
			/*
			 * Save the network to a file called <network Name>_<majorVersion>_<minorVersion>_<revision>.enn in the path supplied in C string cstrPath
			 */
			{
				fstream * pFile;
				status_t rVal;
				char cstrPathFile[255];	// dumb
				char cstrFileName[25]; // dumb

				if (checkExists(cstrPath, false))
				{
					sprintf(cstrPathFile, "%s//%s", cstrPath, defaultName(cstrFileName));

					pFile = new fstream();
					pFile->open(cstrPathFile, ios::out);
					rVal = saveTo(pFile);
					pFile->close();
					delete pFile;

					return rVal;
				}
				else
					throw format_Error(ENN_ERR_NON_FILE);

				return SUCCESS;
			}

			status_t	saveTo(fstream * pFile)
			/*
			 * Save the network to the file stream pointed to by pFile. The name of the file will not be changed.
			 */
			{
				string strContent;
				status_t rVal;
				rVal = saveOn(&strContent);
				(*pFile) << strContent;

				return rVal;
			}

			// save to disk
			status_t	saveOn(string * strOut)
			/*
			 *  save the net in the given existing string
			 */
			{
				stringstream ss;
				unsigned int layerI, nodeI, linkI;
				unsigned int nodeIndex = 0;
				unsigned int weightIndex = 0;

				ss.precision(8);

				ss << "version" << ennVersion << "\nname(" << networkName << "," << majorVersion << "," << minorVersion << "," << revision << ")\n" ;
				//ss << "version(1,0,0)" << "\nname(" << networkName << "," << majorVersion << "," << minorVersion << "," << revision << ")\n" ;       // compatabiity

				//ss << "networkTopology(" << (*layers)[0].nodeCount << "," <<  (*layers)[1].nodeCount <<  "," <<  (*layers)[2].nodeCount << ")\n"; // compatability 3 layer network
				ss << "networkTopology(" << layerCount << ";";
				for (layerI=0 ; layerI< (unsigned int)(layerCount - 1); layerI++)
                    ss << (*layers)[layerI].nodeCount << ",";
                ss <<  (*layers)[layerI].nodeCount << ")\n"; // finish with the )

				ss << "learning(" << clLearningRate << "," << clTrainingMomentum << ")\n";

				ss << "comment(link(layer, to node, from node, weight))\n";
				ss << "comment(node(layer, node, bias))\n";

				// call the detail storage process here
                for (layerI = 1; layerI < (unsigned int)layerCount; layerI++)
                {
                    ss << "comment(Storing layer:" << layerI <<")\n";
                    ss << "comment(TBD:layer modifiers)\n";
                    for (nodeI=0; nodeI<(*layers)[layerI].nodeCount; nodeI++)
                    {
                        for (linkI=0; linkI<(*layers)[layerI-1].nodeCount; linkI++)
                            ss << "link(" << layerI << "," << nodeI << "," << linkI << "," << clWeights[weightIndex++] << ")\n";
                            //ss << "link(" << (layerI - 1) << "," << linkI << "," << nodeI << "," << clWeights[weightIndex++] << ")\n";        // compatability mode
                        ss << "node(" << layerI << "," << nodeI << "," << clNodeBiases[nodeIndex++] << ")\n";
                        ss << "comment(TBD:node modifiers)\n";
                    }
                }


				hasChanged = false;

				(*strOut) = ss.str();

				return SUCCESS;
			}

	// Modify
			status_t	alter(int newIn, int newHidden, int newOut)
			/*
			 * Alter the topology of the network to be
			 * newIn: the new number of input nodes
			 * newHidden: the new number of hidden nodes
			 * newOut: the new number of output nodes
			 *
			 * This will randomise the network and increment the major version resetting the minorVerions and revision
			 *
			 */
			{
//				unsigned int layerNo = 0;

//				delete theInputLayer;
//				delete theHiddenLayer;
//				delete theOutputLayer;

//                net.setHiddenNodes(newHidden);
//                net.setOutputNodes(newOut);
//                net.setStandardInputNodes(newIn);

//                theInputLayer = new inputLayer(net, layerNo++);		// deleted in ~nn
//                theHiddenLayer = new hiddenLayer(net, layerNo++);	// deleted in ~nn
//                theOutputLayer = new outputLayer(net, layerNo++);	// deleted in ~nn

//                theInputLayer->connectNodes(theHiddenLayer->nodeList());
//                theHiddenLayer->connectNodes(theOutputLayer->nodeList());

                randomise();
                incrementMajorVersion();

                hasChanged = true;
				return SUCCESS;
			}

//			status_t	alter(unsigned int layer, layer_modifier mod, bool boolAdd = true)
//			/*
//			 * Alter a layer within the network. Currently you can only add or remove a bias node from layer zero (the input layer)
//			 *
//			 * This will randomise the network and increment the major version resetting the minorVerions and revision
//			 *
//			 */
//			{
//				network_description newNet;
//
//				newNet = net;	// keep all the old values
//
////				delete theInputLayer;
////				delete theHiddenLayer;
////				delete theOutputLayer;
//
//                newNet.setInputLayerBiasNode(boolAdd);
//
//                setup(newNet);
//
//                randomise();
//                incrementMajorVersion();
//
//                hasChanged = true;
//
//                return SUCCESS;
//			}

            char *	defaultName(char * buffer)
            /*
             * Return the default name for the network, which is: <network Name>_<majorVersion>_<minorVersion>_<revision>.enn
             *
             * Note: the calling function must make sure that there is enough room in the buffer
             *
             */
            {
            	sprintf(buffer, "%s_%d_%d_%d.nn", networkName.c_str(), majorVersion, minorVersion, revision);

            	return buffer;
            }

			bool		needsSaving() { return hasChanged; }	// Return true if the network has changed since it was last saved.

	// Build - callbacks for the networkFile that is reading in the netowrk from a .enn file

			void setNetworkTopology(vector<unsigned int> * layerWidths)
			{
			    cl_int i;
                unsigned int j;
                unsigned int nodeIndex = 0;     // the index into the nodeWeightIndex array (which is flat unlike the  layers->node structure)
                int    prevLayerNodeCount;
                cl_int weightIndex = 0;     // the index into weight array


                layerCount = (cl_int)layerWidths->size();
//				cout << "Layer widths:" << layerCount << " - ";
//				for (i = 0; i<layerCount; i++)
//                    cout << (*layerWidths)[i] << " " ;
//                cout << "\n";

				layers = new vector<layerData>(layerCount);

				setupLayer(&((*layers)[0]), (*layerWidths)[0], 0); 	// arg 2 is the previous layer width therefore 0 for the input layer
				for (i=1; i<layerCount; i++)
                    setupLayer(&((*layers)[i]), (*layerWidths)[i], (*layerWidths)[i-1]);


                clLayerWidths = (cl_int*) clmalloc(stdacc, layerCount * sizeof(cl_int), 0);
                for (i=0; i<layerCount; i++)
                    clLayerWidths[i] = (*layers)[i].nodeCount;

                clInputLayer = (cl_float*) clmalloc(stdacc, (size_t)layerZeroWidth() * sizeof(cl_float), 0);
                clOutputLayer = (cl_float*) clmalloc(stdacc, (size_t)layerNWidth() * sizeof(cl_float), 0);
                clOutputError = (cl_float*) clmalloc(stdacc, (size_t)layerNWidth() * sizeof(cl_float), 0);
                //testing
                for (i=0;i<(cl_int)layerNWidth();i++)
                    clOutputLayer[i] = -1.0;
                //\\testing

                totalWeights = 0;
                nodeBiasArraySize = 0;
                largestDerivedLayer = 0;
                largestInputLayer = (*layerWidths)[0];
                maxWeightsPerCore = 0;
                totalDerivedNodes = (*layerWidths)[0];  /// input layer is copied to the derived value array to streamline forward and back passes
                for (i=1; i < layerCount; i++)
                {
                    totalWeights += (*layerWidths)[i] * (*layerWidths)[i-1];
                    nodeBiasArraySize += (*layerWidths)[i];
                    largestDerivedLayer = (largestDerivedLayer < (*layerWidths)[i]) ? (*layerWidths)[i] : largestDerivedLayer;
                    totalDerivedNodes += (*layerWidths)[i];
                    largestInputLayer = ((largestInputLayer < (*layerWidths)[i]) && (i != (layerCount - 1))) ? (*layerWidths)[i] : largestInputLayer;
                    maxWeightsPerCore += (((*layerWidths)[i] / CORECOUNT) + 1) * (*layerWidths)[i-1];
                }
//                cout << "total weights: " << totalWeights << " total Node Biases " << nodeBiasArraySize << "\n";

                clWeights = (cl_float*) clmalloc(stdacc, totalWeights * sizeof(cl_float), 0);
                clNodeBiases = (cl_float*) clmalloc(stdacc, nodeBiasArraySize * sizeof(cl_float), 0);
                clNodeWeightIndex = (cl_int*) clmalloc(stdacc, nodeBiasArraySize * sizeof(cl_int), 0);
                for (i=1; i < layerCount; i++)
                {
                    prevLayerNodeCount = (*layers)[i-1].nodeCount;
                    for(j=0; j<(*layers)[i].nodeCount; j++)
                    {
                        clNodeWeightIndex[nodeIndex++] = weightIndex;
                        weightIndex += prevLayerNodeCount;
                    }
                }
			}

			void setNodeBias(unsigned int layer, unsigned int node, float bias)
			{
			    unsigned int i;
			    unsigned int offset = 0;
//				layers->operator[](layer).nodeInfo->operator[](node).bias = bias;

                for(i=1; i<layer; i++)          ///  bias array starts at 0 for layer 1
                    offset += clLayerWidths[i];

                clNodeBiases[offset + node] = (cl_float)bias;
//				cout << "b," << layer << "," << node  << "," << bias <<  "," << offset << "," << "\n";
			}

			void setLinkWeight(unsigned int layer, unsigned int fromNode, unsigned int toNode, float weight)
			{
			    unsigned int i;
			    unsigned int offset = 0;

//				layers->operator[](layer).nodeInfo->operator[](toNode).incomingWeights->operator[](fromNode) = weight;

                for(i=1; i < layer; i++)
                {
                    offset += clLayerWidths[i] * clLayerWidths[i-1]; // gets us to the begining of the layer where the link goes
                }
                offset += clLayerWidths[i-1] * toNode;
                offset += fromNode;

                clWeights[offset] = (cl_float)weight;
//				cout << "weight layer: " << layer << " from " << fromNode << " to " << toNode << " weight " << weight << " stored as " << clWeights[offset] << " \n";
			}


			void setName(string * name)
			{
//				cout << "name:" << (*name) << "\n";
				networkName = *name;
			}

			void setVersion(unsigned int major, unsigned int minor, unsigned int revis)
			{
//				cout << "version " << major << " " << minor << " " << revis << "\n";
				majorVersion = major;
				minorVersion = minor;
				revision = revis;
			}

			void setTrainingLearningRate(float learningRate)
			{
//				cout << "setting LR:" << learningRate << "\n";
				clLearningRate = (cl_float)learningRate;
			}

			void setTrainingMomentum(float momentum)
			{
//				cout << "setting momentum:" << momentum << "\n";
				clTrainingMomentum = (cl_float)momentum;
			}

			void setHasBiasNode(unsigned int layer, bool hasBiasNode)
			{
//				cout << "layer: " << layer << "bias node:" << hasBiasNode << "\n";
				(*layers)[layer].hasBiasNode = hasBiasNode;		// the node is added at setup time and is only used if hasBiasNode is true
			}

			void setNodeModifier(unsigned int layer, unsigned int node, node_modifier mod)
			{

			    /// not implemented yet
			    throw;
			}

	// Access

			unsigned int layerZeroWidth()
			{
				return (*layers)[0].nodeCount;
			}

			unsigned int layerNWidth()
			{
				return (*layers)[layerCount-1].nodeCount;
			}

    // Setup

	private:
            void		incrementRevision() { revision++; }
            void		incrementMinorVersion() { minorVersion++; revision = 0; }
            void		incrementMajorVersion() { majorVersion++; minorVersion = revision = 0; }

            void		setupLayer(layerData * layer, unsigned int width, unsigned int previousLayerWidth)
            {
            	unsigned int nodeI;

//            	cout << "layer width: " << width << " prev:" << previousLayerWidth << "\n";

            	layer->nodeCount = width;
            	layer->nodeInfo = new vector<nodeData>(width + 1); // add the space now for the bias node
            	layer->transition = TRANSITION_SIGMOID;
            	layer->hasBiasNode = false;

            	for(nodeI = 0; nodeI < width; nodeI++)
            	{
//            		cout << "adding node: " << nodeI << " content\n";
            		layer->nodeInfo->operator[](nodeI).inputType = INPUT_UNIFORM;
            		layer->nodeInfo->operator[](nodeI).p = 0.5;
            		layer->nodeInfo->operator[](nodeI).pIsOneHalf = true;

            		if (previousLayerWidth != 0)	// previous  == 0 indicates that the layer is the input layer therefore does not need any incoming weights
            		{
//            			layer->nodeInfo->operator[](nodeI).incomingWeights = new vector<float>(previousLayerWidth + 1); // add one in case the previous layer has a bias node

//            			cout << "space for incoming weights: " << layer->nodeInfo->operator[](nodeI).incomingWeights->size() << "\n";
            		}
//            		else
//            			cout << "no incoming weight space allocated\n";

            		//layer->nodeInfo->operator[](nodeI).bias = rand();
            	}

            	// and set up the bias node in case it gets switched on later
//        		cout << "adding bias node content\n";
        		layer->nodeInfo->operator[](nodeI).inputType = INPUT_BINARY;
        		layer->nodeInfo->operator[](nodeI).p = 1;
        		layer->nodeInfo->operator[](nodeI).pIsOneHalf = false;
        		layer->nodeInfo->operator[](nodeI).nodeValue = 1;

            }

    // Other
            bool checkExists(const char * fileName, bool boolShouldBeFile = true)
            {
            	struct stat fileAtt;

            	if (stat(fileName, &fileAtt) != 0)
            		return false;
            	else
            		if (boolShouldBeFile)
            			return S_ISREG(fileAtt.st_mode);
            		else
            			return S_ISDIR(fileAtt.st_mode);

            }

            void writeDefsFile()
            {
				fstream * pFile;
				cl_int i;

				if (checkExists(PATHTOCLDEFSFILE, true))
				{
					pFile = new fstream();
					pFile->open(PATHTOCLDEFSFILE, ios::out);
					(*pFile) << "#define CORECOUNT " << CORECOUNT << "\n";
					(*pFile) << "#define LAYERCOUNT " << layerCount << "\n#define OUTPUTLAYER " << (layerCount - 1) << "\n";
					(*pFile) << "#define MAXWEIGHTSPERCORE " << maxWeightsPerCore << "\n";
					(*pFile) << "#define LARGESTDERIVEDLAYER " << largestDerivedLayer << "\n";
					(*pFile) << "#define LARGESTINPUTLAYER " << largestInputLayer << "\n";
					(*pFile) << "#define TOTALNODES " << totalDerivedNodes << "\n";
//					(*pFile) << "#define TOTALWEIGHTS " << totalWeights << "\n";
					(*pFile) << "#define INITWIDTHARRAY {";
					for (i=0; i<layerCount-1; i++)
                        (*pFile) << (*layers)[i].nodeCount << ",";
                    (*pFile) << (*layers)[i].nodeCount << "}\n";

					pFile->close();
					delete pFile;
				}
				else
					throw format_Error(ENN_ERR_CL_DEFS_NOT_FOUND);

            }

	private:
	cl_int              layerCount;
    vector<layerData> * layers;

    cl_int      *       clLayerWidths;
    cl_float    *       clInputLayer;
    cl_float    *       clOutputLayer;
    cl_float    *       clOutputError;
    cl_float    *       clWeights;          /// all of the network's weights in one big array
    cl_float    *       clNodeBiases;       /// the biases for each node used to calculate the node output
    cl_int      *       clNodeWeightIndex;  /// the index into the weight array where this node's weights star
    cl_float    *       clWeightDeltas;     //  temporary: shared space for cores to share their incoming weight deltas for back prop

    size_t              totalWeights;       /// the number of weights in the whole network

    /// variables to be written to the cl defs file
    unsigned int        maxWeightsPerCore;  /// the largest number of weights between two layers
    unsigned int        nodeBiasArraySize;  /// the sum of all of the layer widths - the minimum size of an array that can fit all nodes
    unsigned int        largestDerivedLayer;/// the widest layer not including the input layer
    unsigned int        largestInputLayer;  /// the widest layer that provides input to a subsequent layer (i.e. ever layer exept the output layer)
    unsigned int        totalDerivedNodes;  /// the number of derived nodes in the whole network

    cl_float			clLearningRate;
    cl_float			clTrainingMomentum;

	// house keeping
	bool				hasChanged;					// set to true after randomisaton or training

	// testing
	vector<float>		errorVector;				// pass a pointer to this vector in the test callback

	// identificaton
	unsigned int		majorVersion;
	unsigned int		minorVersion;
	unsigned int		revision;					// a number to distinguish different versions of the same net
    string				networkName;

    funcRunCallback     runCallback;

};

#endif
