#ifndef _networkfile_h
#define _networkfile_h

#include "nnFile.hpp"
#include "nn.hpp"

/*
 * The eNN file wrapper hierarchy:
 *
 *	NNFile
 *		networkFile
 *		dataFile
 *			inputFile
 *			trainingFile
 *
 * networkFile takes a text file formated by the neural net by the SaveTo or SaveOn functions,
 * reads it in completely and then responds to the access calls to pass back each value. The
 * default file extension for a network file is .enn
 *
 * You should not have to edit a .enn file directly yourself.
 */

class networkFile : public NNFile
{
		public:
                        networkFile(ifstream * theFile) : NNFile(theFile)
                        {
//                            hiddenBiases = NULL;
//                            outputBiases = NULL;
//                            hasInputBiasNode = false;
                        }

                        networkFile() : NNFile()
                        {
//                            hiddenBiases = NULL;
//                            outputBiases = NULL;
//                            hasInputBiasNode = false;
                        }

                        networkFile(const char * cstrFileName) :NNFile(cstrFileName)
                        {

                        }

                        networkFile(string * strFileName) :NNFile(strFileName)
                        {

                        }


                        virtual ~networkFile() //: ~NNFile()
                        {
//                            if (hiddenBiases != NULL)
//                                delete hiddenBiases;
//                            if (outputBiases != NULL)
//                                delete outputBiases;
                        }

        void			setTo(ifstream * theFile)
                        {
//                            if (hiddenBiases != NULL)
//                                delete hiddenBiases;
//                            if (outputBiases != NULL)
//                                delete outputBiases;
                            NNFile::setTo(theFile);
                        }

 virtual nnFileContents	fileType()
						{
							return NETWORK;
						}


	// access
/*		float			linkValue(unsigned int layer, unsigned int node, unsigned int link);
		float			biasValue(unsigned int layer, unsigned int node);

		unsigned int	majorVersion() { return major; }	// major reconstruction of the network (will differ from other major versions by having different inputs etc.
		unsigned int	minorVersion() { return minor; }	// minor versions have different starting point for training
		unsigned int	revision() { return revis; }		// revisions have different amounts of training

//        void			networkDescription(network_description * netDes)	// structure passed in by the caller
//                        {
//                            (*netDes) = net;
//}

        void			networkName(string * netName)
                        {
                            (*netName) = name;
                        }

        twoDFloatArray *linkWeights(unsigned int layer)
                        {
                            switch (layer)
                            {
                                case 0:
                                    return &inputLinkWghts;
                                    break;
                                case 1:
                                    return &hiddenLinkWghts;
                                    break;
                                case 2:
                                    throw format_Error(ENN_ERR_LINK_ON_OUTPUT);
                                    break;
                                default:
                                    throw format_Error(ENN_ERR_TOO_MANY_LAYERS);
                            }
                        }

        vector<float> *	nodeBiases(unsigned int layer)
                        {
                            switch (layer)
                            {
                                case 0:
                                    throw format_Error(ENN_ERR_INPUT_NODE_BIAS_REQUESTED);
                                    break;
                                case 1:
                                    return hiddenBiases;
                                    break;
                                case 2:
                                    return outputBiases;
                                    break;
                                default:
                                    throw format_Error(ENN_ERR_TOO_MANY_LAYERS);
                            }
                        }
*/
	private:
        status_t		decodeLine(string * strLine)
                        {
                            string verb = "";
                            string arguements = "";

                            if (verbArguement(strLine, verb, arguements))
                            {
                                if (verb ==  "link")
                                {
                                	status_t rVal;
#ifdef _DEBUG_
                                        	cout << "Decode Link\n";
 #endif

                                    rVal = decodeLink(&arguements);
//                                    cout << "done Decode Link - dumping\n";
//                                    inputLinkWghts.writeOn(cout);
//                                    cout << "Decode Link done with dump - exit\n";
                                    return rVal;
                                }
                                if (verb ==  "node")
                                {
#ifdef _DEBUG_
                                        	cout << "Decode node\n";
#endif
                                    return decodeNode(&arguements);
                                }
                                if (verb == "version")
                                {
#ifdef _DEBUG_
                                        	cout << "Decode version\n";
#endif
                                    // no need to check the file version just yet
                                    return SUCCESS;
                                }
                                if (verb == "name")
                                {
#ifdef _DEBUG_
                                        	cout << "Decode Name\n";
#endif
                                    return decodeName(&arguements);
                                }
                                if (verb ==  "networkTopology")
                                {
#ifdef _DEBUG_
                                        	cout << "Decode Topo\n";
#endif
                                    return decodeNetworkTopology(&arguements);
                                }
                                if (verb == "comment")
                                {
                                    // do nothing with comments
                                    return SUCCESS;
                                }
                                if (verb == "learning")
                                {
#ifdef _DEBUG_
                                        	cout << "Decode Learning\n";
#endif
                                    return decodeLearning(&arguements);
                                }
                                if (verb == "layerModifier")
                                {
#ifdef _DEBUG_
                                        	cout << "Decode Layer mod\n";
#endif
                                        	// need to actually decode the layerModifer clause
                                    return decodeLayerModifier(&arguements);
                                }

                                errMessage = ENN_ERR_UNK_KEY_WORD;
                                errMessage += ": ";
                                errMessage += verb.c_str();
                                throw format_Error(errMessage.c_str());
                            }
                            throw format_Error(ENN_ERR_NON_FILE);
                        }

        status_t		decodeLink(string * strBracket)
                        {
                            unsigned int layer;
                            unsigned int toNode;
                            unsigned int fromNode;
                            float		 linkWeight;

                            std::string::size_type		 startPos;

                            startPos = 1;
                            layer = nextUIValue(strBracket, startPos);
                            toNode = nextUIValue(strBracket, startPos);
                            fromNode = nextUIValue(strBracket, startPos);
                            linkWeight = nextFValue(strBracket, startPos, ')');

#ifdef _DEBUG_
                                        	cout << "Link: layer-" << layer << " innode-" << node << " outnode-" << link << " weight: " << linkWeight << "\n";
#endif
                              ((nn*)theNetwork)->setLinkWeight(layer, fromNode, toNode, linkWeight);

                            return SUCCESS;
                        }

        status_t		decodeNode(string * strBracket)
                        {
                            unsigned int layer;
                            unsigned int node;
                            float		 nodeBias;

                            std::string::size_type		 startPos;

                            startPos = 1;
                            layer = nextUIValue(strBracket, startPos);
                            node = nextUIValue(strBracket, startPos);
                            nodeBias = nextFValue(strBracket, startPos, ')');

#ifdef _DEBUG_
                                        	cout << "Node: layer-" << layer << " node-" << node << " bias-" << nodeBias << "\n";
#endif

                              ((nn*)theNetwork)->setNodeBias(layer, node, nodeBias);

                            return SUCCESS;
                        }

        status_t		decodeName(string * strBracket)
                        {
							std::string::size_type		startPos;
							std::string::size_type		commaPos;
							std::string					name;

							unsigned int major;
							unsigned int minor;
							unsigned int revis;

                            // name
                            startPos = 1;	// start at 1 to skip the opening bracket
                            commaPos = strBracket->find(',', startPos);	// find the first comma
                            name = strBracket->substr(1, commaPos - 1);
                            startPos = ++commaPos;
                            ((nn*)theNetwork)->setName(&name);

                            major = nextUIValue(strBracket, startPos);
                            minor = nextUIValue(strBracket, startPos);
                            revis = nextUIValue(strBracket, startPos, ')');
                            ((nn*)theNetwork)->setVersion(major, minor, revis);

#ifdef _DEBUG_
                                        	cout << "Name: " << name << " major-" << major << " minor-" << minor << " revision" << revis << "\n";
#endif

                            return SUCCESS;
                        }

        status_t		decodeNetworkTopology(string * strBracket)
                        {
        					vector<unsigned int> layerWidths(maxLayers);

                            status_t returnVal = NNFile::decodeNetworkTopology(strBracket, maxLayers, &layerWidths);
                            ((nn*) theNetwork)->setNetworkTopology(&layerWidths);

                            return returnVal;
                        }

        status_t		decodeVersion(string * strBracket)
                        {
        					string curVer(ennVersion);
        					if (*strBracket == curVer)
        						return SUCCESS;
        					else
        						throw format_Error(ENN_ERR_UNSUPPORTED_ENN_FILE_FORMAT);

        					return FAILURE;		// just to keep the formatter happy
                        }

        status_t		decodeLearning(string * strBracket)
                        {
        					std::string::size_type	startPos;

                            startPos = 1;
                            ((nn*)theNetwork)->setTrainingLearningRate(nextFValue(strBracket, startPos));
                            ((nn*)theNetwork)->setTrainingMomentum(nextFValue(strBracket, startPos, ')'));

                            return SUCCESS;
                        }

        status_t		decodeLayerModifier(string * strBracket)
						{
        					unsigned int whichLayer;
        					string modifier = "";
        					string value = "";
        					std::string::size_type	startPos = 1;

        					whichLayer = nextUIValue(strBracket, startPos);
        					keyValue(strBracket, startPos, modifier, value);

        					if (modifier == "biasNode")
        					{
								if (whichLayer == 0)	// expand to include all but output layer
								{
									if (value == "true")
										((nn*)theNetwork)->setHasBiasNode(whichLayer, true);
									else
										((nn*)theNetwork)->setHasBiasNode(whichLayer, false);
								}
								else
									throw format_Error(ENN_ERR_BIAS_NODE_ON_INVALID_LAYER);
        					}
        					else
        						throw format_Error(ENN_ERR_UNK_MODIFIER);

        					return SUCCESS;

						}

        status_t		readInLines(bool shouldNotBeHere)
						{
							return FAILURE;
						}

	private:
        /*
         * All data is passed to the network as it is read in
         */
};

#endif
