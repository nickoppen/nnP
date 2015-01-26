
#ifndef _nnfile_h
#define _nnfile_h

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include "errStruct.hpp"

enum nnFileContents {DATA, NETWORK, TRAIN_TEST};

/*
 * The eNN file wrapper hierarchy:
 *
 *	NNFile
 *		networkFile
 *		dataFile
 *			inputFile
 *			trainingFile
 *
 *	NNFile implements the primative functions used by all file types. NNFile and sub-classes do all the format checking and pass
 *	clean data to the network.
 *
 *	Once you have created the object and set the file pointer, call readInFile() to load all the data from the file into memory.
 *
 *	If you are having trouble with your data files, compile your application with the _DEBUG_ compiler directive set and that will
 *	pass every input line, raw text and the values read onto standard output.
 *
 *	See the file content definitions for more details on the various file types.
 *
 *	NOTE: no files contain spaces.
 */


class NNFile

{
	public:
                                    NNFile()
                                    {
                                        pFile = NULL;
                                        errMessage = "";
                                    }

                                    NNFile(ifstream * theFile)
                                    {
                                        pFile = theFile;
                                        errMessage = "";
                                    }

                                    NNFile(const char * cstrFileName)
                                    {
                                    	setTo(cstrFileName);
                                    }

                                    NNFile(string * strFileName)
                                    {
                                    	setTo(strFileName->c_str());
                                    }

                                    virtual ~NNFile()
                                    {
                                        pFile->close();
                                        delete pFile;
                                    }

            status_t				setTo(ifstream * theFile)
                                    {
                                        pFile = theFile;
                                        return SUCCESS;
                                    }

            status_t				setTo(const char * cstrFileName)
									{
										if (checkExists(cstrFileName))
											pFile = new ifstream(cstrFileName);
										else
											throw format_Error(ENN_ERR_NO_FILE_FOUND);
										return SUCCESS;
									}

            status_t				readInFile(void * net)
                                    {
//                                        if (pFile->gcount())		// make sure that the file has something in it
            								theNetwork = net;
                                            return readInLines();
//                                        else
//                                            throw format_Error(ENN_ERR_NON_FILE);
                                    }

            status_t				readInFile(void * net, bool isTrainingFile)
									{
            							// check for file content as above
            							theNetwork = net;
            							return readInLines(isTrainingFile);

									}

            virtual nnFileContents	fileType() = 0;


	protected:
            status_t				readInLines()
                                    {
                                        size_t		 strLength = 1;
                                        string		 fragment;
                                        status_t	 decodeResult = SUCCESS;

                                        while (!(pFile->eof()))
                                        {
                                        	getline((*pFile), fragment);
                                        	strLength = fragment.length();

#ifdef _DEBUG_
                                        	cout << "\n" << fragment << "\n";
#endif

                                            if (strLength > 1)
                                                if ((decodeResult = this->decodeLine(&fragment)) != SUCCESS)
                                                    throw format_Error(ENN_ERR_NON_FILE);

                                            fragment.clear();
                                        }
                                        return decodeResult;
                                    }

    virtual status_t				readInLines(bool isTrainingFile) = 0;

    virtual	status_t				decodeLine(string * strLines) = 0;

	protected:
            unsigned int			nextUIValue(string * fragment, std::string::size_type & startPos, const char limiter = ',')
                                    {
            							std::string::size_type 	endPos;
                                        char	strValue[] = "          ";

                                        endPos = fragment->find(limiter, startPos);	// find the delimiter
                                        if (endPos < startPos)
                                        {
                                            throw format_Error(ENN_ERR_LINE_DECODE_FAILED);
                                        }

                                        fragment->copy(strValue, (endPos - startPos), startPos);
                                        startPos = ++endPos;
                                        return atoi(strValue);
                                    }

            float					nextFValue(string * fragment, std::string::size_type & startPos, const char limiter = ',')
                                    {
            							std::string::size_type	endPos;
                                        char	strValue[] = "                         ";

                                        endPos = fragment->find(limiter, startPos);	// find the delimiter
                                        if (endPos < startPos) // if the delimiter is not found endPos is -1
                                        {
                                            //throw format_Error(ENN_ERR_LINE_DECODE_FAILED + limiter);
                                            throw format_Error(ENN_ERR_LINE_DECODE_FAILED);
                                        }

                                        fragment->copy(strValue, (endPos - startPos), startPos);
                                        startPos = ++endPos;
                                        return (float)atof(strValue);
                                    }

            int						verbArguement(string * line, string & verb, string & arg)
                                    {
                                        // splits verb(args) into verb and (args)
            							std::string::size_type bracketPos;

                                        // pull off the word preceeding the (
                                        bracketPos = line->find('(', 0);
                                        if (bracketPos >= 0)
                                        {
                                        	verb = line->substr(0, bracketPos);
											arg = line->substr(bracketPos);
                                            return 1;
                                        }
                                        return 0;

                                    }

            status_t				keyValue(string* line, std::string::size_type & startPos, string & key, string & value, const char separator = ':', const char limiter = ',')
									{
										std::size_t sepPos;
										std::size_t endPos;

										sepPos = line->find(separator, startPos);
										if (sepPos > 1)
										{
											key = line->substr(startPos, sepPos - startPos);
											endPos = line->find(limiter, sepPos + 1);
											if ((endPos < startPos) || (endPos > line->size()))		// there are no more modifiers
											{
												endPos = line->find(')', sepPos);
												startPos = 0;
											}
											else
												startPos = endPos++; // return the begining of the next key:value pair

											value = line->substr(sepPos+1, endPos - (sepPos + 1));

											return SUCCESS;
										}
										else
											throw format_Error(ENN_ERR_KEY_VALUE_FORMAT_ERROR);

									}

            status_t				decodeNetworkTopology(string * fragment, const int maxLayers, vector<unsigned int> * layerWidths)
                                    {
            							std::string::size_type	startPos = 1;
            							unsigned int layerCount, layer;

            							layerCount = nextUIValue(fragment, startPos, ';');
            							if (layerWidths->size() != layerCount)
                                            layerWidths->resize(layerCount);

            							for (layer=0; layer < layerCount-1; layer++)
                                            (*layerWidths)[layer] = nextUIValue(fragment, startPos);

            							(*layerWidths)[layerCount-1] = nextUIValue(fragment, startPos, ')');
#ifdef _DEBUG_
                                      	cout << "Network Topo - InputNodes: " << (*layerWidths)[0] << " HiddenNodes: " << (*layerWidths)[1] << " OutputNodes: " << (*layerWidths)[2] << "\n";
#endif

                                        return SUCCESS;
                                    }

	private:
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

	protected:
            ifstream	*			pFile;		// temporary storage deleted by the doc
            string					errMessage;
            void		*			theNetwork;
};

#endif
