#ifndef _datafile_h
#define _datafile_h

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
 *	The inputFile contains just input data. These are the "production" files that contain the data that you
 *	want to classify.
 *
 *	The trainingFile contains known, preclassified data sets and their desired output separated by a ;
 *
 *	See the file content descriptions for more information.
 *
 *	All data is stored as floats in the standard template class vector. These classes contain a vectors of vector<float>
 *	buried in the class twoDFloatArray.
 *
 *	To access an input set call NNFile::inputFile::inputSet(unsigned int) or NNFile::trainingFile::inputSet(unsigned int) where the arguement is
 *	the row that you want. Similarly call trainingFile::outputSet(unsigned int) to retrieve the output set.
 *
 *	The easiest way to use these classes is to pass your file reference to your newly created network and let it sort out the data.
 */

class dataFile: public NNFile
{
public:
	dataFile() :
			NNFile()
	{
		lineCount = 0;
//                        inputArray = NULL;
	}

	dataFile(ifstream * theFile) :
			NNFile(theFile)
	{
		lineCount = 0;
//                        inputArray = NULL;
	}

	dataFile(const char * cstrFileName) :
			NNFile(cstrFileName)
	{
		lineCount = 0;
	}

	dataFile(string * strFileName) :
			NNFile(strFileName)
	{
		lineCount = 0;
	}

	virtual ~dataFile() //: ~NNFile()
	{
	}

public:
	// access
	unsigned int inputLines() // return how many lines have been read in
	{ // return how many lines have been read in
		return lineCount;

	}


	virtual nnFileContents fileType() = 0;

private:
	virtual status_t decodeLine(string * strLine) = 0;

protected:
	virtual status_t readInLines(bool isTrainingFile) = 0;

	unsigned int lineCount;

};

class inputFile: public dataFile
{
public:
	inputFile() :
			dataFile()
	{
	}
	inputFile(ifstream * theFile) :
			dataFile(theFile)
	{
	}
	inputFile(const char * cstrFileName) :
			dataFile(cstrFileName)
	{
	}
	inputFile(string * strFileName) :
			dataFile(strFileName)
	{
	}

	virtual ~inputFile() // : ~dataFile()
	{
	}

	virtual nnFileContents fileType()
	{
		return DATA;
	}

protected:
	status_t readInLines(bool shouldNotGetHere)
	{
		return FAILURE;
	}

private:
	status_t decodeLine(string * strLine)
	{
		std::string::size_type bracketPos;
		string verb = "";
		string arguements = "";
		status_t decodeResult;

		bracketPos = strLine->find('(', 0);
		if (bracketPos == std::string::npos)
			throw format_Error(ENN_ERR_NON_FILE);

		if (verbArguement(strLine, verb, arguements))
		{
			if (verb == "inputVector")
			{
#ifdef _DEBUG_
				cout << "Decoding Input Vector\n";
#endif

				return decodeInputVector(&arguements);
			}
			if (verb == "networkTopology")
			{
#ifdef _DEBUG_
				cout << "Decode Topology\n";
#endif

				vector<unsigned int> layerWidths(maxLayers);
				decodeResult = decodeNetworkTopology(&arguements, maxLayers, &layerWidths);
				if (layerWidths[0] != ((nn*) theNetwork)->layerZeroWidth())
					throw format_Error(ENN_ERR_NONMATCHING_TOPOLOGY);

				return decodeResult;
			}

//                            errMessage.Format("%s: %s", ENN_ERR_UNK_KEY_WORD, verb.GetBuffer());
			errMessage = ENN_ERR_UNK_KEY_WORD;
			errMessage += ": ";
			errMessage += verb;
			throw format_Error(errMessage.c_str());
		}
		else
			throw format_Error(ENN_ERR_NON_FILE);

		return FAILURE; // will not happen
	}

	status_t decodeInputVector(string * fragment)
	{
		float inputValue;
		unsigned int node;
		std::string::size_type startPos;
		unsigned int inputNodeCount;
		vector<float> lineVector(
				inputNodeCount = ((nn*) theNetwork)->layerZeroWidth());

		startPos = 1;

#ifdef _DEBUG_
		cout << "Input Values,";
#endif

		for (node = 0; node < (inputNodeCount - 1); node++) //>
		{
			inputValue = nextFValue(fragment, startPos);
			lineVector[node] = inputValue;
#ifdef _DEBUG_
			cout << " Node " << node << ": " << inputValue;
#endif
		}
		inputValue = nextFValue(fragment, startPos, ')');
#ifdef _DEBUG_
		cout << " Node " << node << ": " << inputValue << "\n";
#endif
		lineVector[node] = inputValue;

		((nn*) theNetwork)->run(&lineVector, NULL, lineCount); // run immediately

		return SUCCESS;
	}

};

class trainingFile: public dataFile
{
public:
	trainingFile() :
			dataFile()
	{
	}

	trainingFile(ifstream * theFile) :
			dataFile(theFile)
	{
	}

	trainingFile(const char * cstrFileName) :
			dataFile(cstrFileName)
	{

	}

	trainingFile(string * strFileName) :
			dataFile(strFileName)
	{

	}

	virtual ~trainingFile() //: ~dataFile()
	{
	}


	virtual nnFileContents fileType()
	{
		return TRAIN_TEST;
	}
protected:
	status_t readInLines(bool isTrainingFile)
	{
		inputToTrain = isTrainingFile;
		return NNFile::readInLines();
	}

private:
	status_t decodeLine(string * strLine)
	{
		std::string::size_type bracketPos;
		string verb = "";
		string arguements = "";
		status_t decodeResult;

		bracketPos = strLine->find('(', 0);
		if (bracketPos == std::string::npos)
			throw format_Error(ENN_ERR_NON_FILE);

		if (verbArguement(strLine, verb, arguements))
		{
			if (verb == "networkTopology")
			{
#ifdef _DEBUG_
				cout << "Decode Topology\n";
#endif
				vector<unsigned int> layerWidths(maxLayers);
				decodeResult = decodeNetworkTopology(&arguements, maxLayers, &layerWidths);
				if ((layerWidths[0] == ((nn*) theNetwork)->layerZeroWidth())
						&& (layerWidths[3] == ((nn*) theNetwork)->layerNWidth()))
					return decodeResult;
				else
					throw format_Error(ENN_ERR_NONMATCHING_TOPOLOGY);

				return decodeResult;
			}
			if (verb == "inputOutputVector")
			{
#ifdef _DEBUG_
				cout << "Decode Input/Output Vector\n";
#endif

				return decodeTrainingVector(&arguements);
			}
			errMessage = ENN_ERR_UNK_KEY_WORD;
			errMessage += ": ";
			errMessage += verb;
			throw format_Error(errMessage.c_str());

		}
		else
			throw format_Error(ENN_ERR_NON_FILE);

		return FAILURE; // will not happen
	}

	status_t decodeTrainingVector(string * fragment)
	{
		float readValue;
		unsigned int node;
		std::string::size_type startPos;

		unsigned int inWidth = ((nn*) theNetwork)->layerZeroWidth();
		unsigned int outWidth = ((nn*) theNetwork)->layerNWidth();
		vector<float> inVector(inWidth);
		vector<float> outVector(outWidth);

		startPos = 1;

#ifdef _DEBUG_
		cout << "Input Vector,";
#endif
		for (node = 0; node < (inWidth - 1); node++) //>
		{
			readValue = nextFValue(fragment, startPos);
			inVector[node] = readValue;
#ifdef _DEBUG_
			cout << " Node: " << node << ": " << readValue;
#endif
		}
		readValue = nextFValue(fragment, startPos, ';');
#ifdef _DEBUG_
		cout << " Node: " << node << ": " << readValue << "\n";
#endif
		inVector[node] = readValue;

#ifdef _DEBUG_
		cout << "Output Vector,";
#endif
		for (node = 0; node < (outWidth - 1); node++) //>
		{
			readValue = nextFValue(fragment, startPos);
			outVector[node] = readValue;
#ifdef _DEBUG_
			cout << " Node: " << node << ": " << readValue;
#endif
		}
		readValue = nextFValue(fragment, startPos, ')');
#ifdef _DEBUG_
		cout << " Node: " << node << ": " << readValue << "\n";
#endif
		outVector[node] = readValue;

		if (inputToTrain)
			((nn*) theNetwork)->train(&inVector, &outVector);
		else
			((nn*)theNetwork)->test(lineCount, &inVector, &outVector);

		return SUCCESS;
	}

	bool inputToTrain;
};

#endif
