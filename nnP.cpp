
#include "nn.hpp"
#include "networkFile.hpp"
#include "dataFile.hpp"

void callback_RunComplete(const int index, void * caller)
{
	vector<float> * resVec;	// created by the net
	unsigned int i;

	resVec = new vector<float>(((nn*)caller)->layerNWidth());
	resVec = ((nn*)caller)->runResult(resVec);
	cout << "Node,Value";
	for(i = 0; i< resVec->size(); i++)
		cout << "\n" << i << "," << (*resVec)[i];
	cout << "\n";
	delete resVec;
}

void callback_TrainingComplete(void * caller)
{
	vector<float> * errVec;	// created by the net
	unsigned int i;

	errVec = new vector<float>(((nn*)caller)->layerNWidth());

	if ( ((nn*)caller)->trainingError(errVec) == SUCCESS)
	{
		cout << "Node,Error";
		for(i = 0; i< errVec->size(); i++)
			cout << "\n" << i << "," << (*errVec)[i];
		cout << "\n";
	}
	else
		cout << "Error: failed to retrieve the error vector.\n";

	delete errVec;
}

void callback_TestComplete(const int index, vector<float>* inputVector, vector<float>* desiredOutput, vector<float>* actualOutput, vector<float> * errVec, void * caller)
{
	unsigned int i;

	if (index == 0)		// title row
	{
		cout << "Index";
		for(i = 0; i< inputVector->size(); i++)
			cout << "\tInput:" << i;
		for(i = 0; i< desiredOutput->size(); i++)
			cout << "\tDesired:" << i;
		for(i = 0; i< actualOutput->size(); i++)
			cout << "\tActual:" << i;
		for(i = 0; i< errVec->size(); i++)
			cout << "\tError:" << i;
		cout << "\n";
	}

	cout << index;
	for(i = 0; i< inputVector->size(); i++)
		cout << "\t" << (*inputVector)[i];
	for(i = 0; i< desiredOutput->size(); i++)
		cout << "\t" << (*desiredOutput)[i];
	for(i = 0; i< actualOutput->size(); i++)
		cout << "\t" << (*actualOutput)[i];
	for(i = 0; i< errVec->size(); i++)
		cout << "\t" << (*errVec)[i];
	cout << "\n";
}

int main(int argc, char *argv[])
{
	nn * theNet = NULL;
	string argvI;
	string strArg;
	string strSave;
	int in, out, hidden;
//	float fVal;
	bool unknownFlag = false;
	bool quiet = false;


	int i;

	string nnName = "TestNet";
	cout.precision(12);

	for (i=1; i<argc; i++)
	{
		argvI = argv[i];
		if (argvI == "-n")
		{
				if (theNet != NULL)
				{
					cout << "A network was already loaded and has been deleted.\n";
					delete theNet;
				}

				try
				{
					networkFile netFile(argv[++i]);
					theNet = new nn(&netFile);

					if (!quiet)
						cout << "Done with -n\n";

				}
				catch (format_Error & e)
				{
					cout << e.mesg << "\n";
				}
		}
		else

			if (argvI == "-t")
			{
				if (theNet == NULL)
					cout << "A network must be loaded before it is trained.\n";
				else
					try
					{
						trainingFile trFile(argv[++i]);
						theNet->train(&trFile, &callback_TrainingComplete);

						if (!quiet)
							cout << "Done with -t\n";

					}
					catch (format_Error & e)
					{
						cout << e.mesg << "\n";
					}
			}
			else

				if ((argvI == "-r") || (argvI == "-run"))
				{
					if (theNet == NULL)
						cout << "A network must be loaded before it is run.\n";
					else
						try
						{
							inputFile dataFile(argv[++i]);
							theNet->run(&dataFile, &callback_RunComplete);

							if (!quiet)
								cout << "Done with -r or -run\n";

						}
						catch (format_Error & e)
						{
							cout << e.mesg << "\n";
						}
				}
				else
					if (argvI == "-s")
					{
						if (theNet == NULL)
							cout << "A network must be loaded before it is saved.\n";
						else
							try
							{
								theNet->saveTo(argv[++i]);

								if (!quiet)
									cout << "Done with -s\n";

							}
							catch (format_Error & e)
							{
								cout << e.mesg << "\n";
							}
					}
					else
						if (argvI == "-c")
						{
							if (theNet != NULL)
							{
								cout << "A network was already loaded and has been deleted.\n";
								delete theNet;
							}
							try
							{
							    unsigned int layer, layerCount;
								layerCount = atoi(argv[++i]);

							    vector<unsigned int> layers(layerCount);
								for (layer=0; layer<layerCount; layer++)
                                    layers[layer] = atoi(argv[++i]);
								strArg = argv[++i];
								theNet = new nn(&layers, strArg);

								if (!quiet)
									cout << "Done with -c\n";

							}
							catch (format_Error & e)
							{
								cout << e.mesg << "\n";
							}
						}
						else
							if (argvI == "-rand")
							{
								if (theNet == NULL)
									cout << "A network must be loaded before it is randomised.\n";
								else
									try
									{
										theNet->randomise();

										if (!quiet)
											cout << "Done with -rand\n";

									}
									catch (format_Error & e)
									{
										cout << e.mesg << "\n";
									}

							}
							else
								if (argvI == "-at")
								{
									if (theNet == NULL)
										cout << "A network must be loaded before it is altered.\n";
									else
										try
										{
											in = atoi(argv[++i]);
											hidden = atoi(argv[++i]);
											out = atoi(argv[++i]);
											theNet->alter(in, hidden, out);

											if (!quiet)
												cout << "Done with -at\n";

										}
										catch (format_Error & e)
										{
											cout << e.mesg << "\n";
										}

								}
								else
									if (argvI == "-am")
									{
										if (theNet == NULL)
											cout << "A network must be loaded before it is altered.\n";
										else
											try
											{
												unsigned int layerNo;
												size_t colonPos;
												string modifier;
												string key;
												string value;


												layerNo = atoi(argv[++i]);
												modifier = argv[++i];
												colonPos = modifier.find(":");

												if (layerNo == 0)
												{
													key = modifier.substr(0, colonPos);
													if (key == "biasNode")
													{
														value = modifier.substr(colonPos + 1);
														if (value == "true")
															theNet->alter(layerNo, BIAS_NODE, true);
														else
															if (modifier.substr(colonPos + 1, modifier.size() - colonPos) == "false")
																theNet->alter(layerNo, BIAS_NODE, false);
															else
																cout << "Invalid argument for modifier biasNode: " << value << "\n";
													}
													else
														cout << "Modifer:" << key << " not valid\n";
												}
												else
													cout << "Invalid layer number:" << layerNo << "\n";


												if (!quiet)
													cout << "Done with -am\n";

											}
											catch (format_Error & e)
											{
												cout << e.mesg << "\n";
											}

									}
									else
										if (argvI == "-test")
										{
											if (theNet == NULL)
												cout << "A network must be loaded before it is tested.\n";
											else
												try
												{
													trainingFile testFile(argv[++i]);
													theNet->test(&testFile, callback_TestComplete);

													if (!quiet)
														cout << "Done with -test\n";

												}
												catch (format_Error & e)
												{
													cout << e.mesg << "\n";
												}

										}
										else
											if (argvI == "-q+")
												quiet = true;
											else
												if (argvI == "-q-")
												{
													quiet = false;
													cout << "Done with -q-\n";
												}
												else
												{
													unknownFlag = true;
													cout << "Don't know that one:" << argvI << "\n";
												}

	}

	if (unknownFlag)
	{
		cout << "\n-n %file load a network from %file\n";
		cout << "-t %file training set %file and write the final error vector to standard output\n";
		cout << "(-r | -run) %file run from input set in %file and write each result vector to standard output\n";//and save the results in outPath\n";
		cout << "-rand randomise the network\n";
		cout << "-test %file run the input/output pattern from training file %file and write the final difference vector to standard output\n";
		cout << "-s %path save on %path (with no trailing /) Use . to write to the working directory.\n";
		cout << "-c %i %h %o %n create a new randomised network with %i input nodes %h hidden nodes %o output nodes, called %n (with a learning rate of 0.1)\n";
		cout << "-at %i %h %o alter the topology to be %i input nodes %h hidden nodes %o output nodes\n";
		cout << "-am 1 (biasNode:true OR biasNode:false) add or remove an input bias node to layer one (default is false)\n";
		cout << "-q+ OR -q- Switch quiet mode on (-q+) or off (-q-) +q+ supresses the 'Done with...' after each command line arguement\n";
	}
	if (theNet != NULL)
		delete theNet;

	return 1;
}
