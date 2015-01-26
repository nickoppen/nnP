#ifndef _errStruct_h
#define _errStruct_h

#include <assert.h>

enum status_t { FAILURE, SUCCESS };

const char ENN_ERR_UNK_KEY_WORD[] = "Unknown Key Word";
const char ENN_ERR_LINE_TOO_LONG[] = "Line too long";
const char ENN_ERR_CONTENT_IN_NETWORK_FILE[] = "Content error in network file";
const char ENN_ERR_INPUT_NODE_BIAS[] = "Bias recorded for input Node";
const char ENN_ERR_TOO_MANY_LAYERS[] = "Too many layers recorded in network file";
const char ENN_ERR_INPUT_NODE_BIAS_REQUESTED[] = "Input Nodes have no biases";
const char ENN_ERR_INPUT_NODE_WEIGHT_REQUESTED[] = "Input Nodes have no weights";
const char ENN_ERR_LAYER_DOES_NOT_EXIST[] = "Layer does not exist";
const char ENN_ERR_NON_FILE[] = "This file does not seem to be a eNN file or the file does not exist";
const char ENN_ERR_LINK_ON_OUTPUT[] = "Link value recorded for output node";
const char ENN_ERR_LINE_DECODE_FAILED[] = "Did not find expected line delimiter";
const char ENN_ERR_UNK_MODIFIER[] = "Unknown Layer Modifier";
const char ENN_ERR_BIAS_NODE_ON_INVALID_LAYER[] = "Bias node requested on an output only node";
const char ENN_ERR_KEY_VALUE_FORMAT_ERROR[] = "Key:Value format error";
const char ENN_ERR_NO_FILE_FOUND[] = "File does not exist";
const char ENN_ERR_NONMATCHING_TOPOLOGY[] = "Data or training file topology does not match netowrk";
const char ENN_ERR_UNSUPPORTED_ENN_FILE_FORMAT[] = "The version of the enn file is not supported";
const char ENN_ERR_CL_DEFS_NOT_FOUND[] = "The definitions file for the cl kernals was not found";

struct format_Error
{
	const char * mesg;
	format_Error(const char * message) { mesg = message; }
/*	format_Error(const char * message, const char * extra)
	{
		static char * buffer[255];

		sprintf(buffer, "%s %s", message, extra);
		mesg = buffer;
	}	// append two parts together
 */
};

const char ENN_ERR_TRAIN_WAITFORTRAIN[] = "Zero return from ResetEvent";
const char ENN_ERR_TRAIN_TRAIN[] = "Zero return from SetEvent";
const char ENN_ERR_TRAIN_WAITFOREVENT[] = "WAIT_FAILED return from Wait";

struct internal_Error
{
	const char * mesg;
//	DWORD lastError;--------------------------------------------------------------------------------
	internal_Error(const char * message)
	{
		mesg = message;
//		lastError = GetLastError();
	}
};
#endif	// _errStruct_h
