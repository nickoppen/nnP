#include "/home/parallella/Work/nnP/coreId16.inc"
//#include <e32_opencl_ext.h>
//#include <coprthr_device.h>
//#include <e32_opencl_ext.h>
#include "/home/parallella/Work/nnP/cldefs.inc"
/// cldefs.inc contains #defines for all static variables
/// example contents of cldefs.inc
///#define CORECOUNT 16
///#define LAYERCOUNT 4
///#define OUTPUTLAYER 3                 // LAYERCOUNT -1
///#define MAXWEIGHTTOLAYER 1024
///#define LARGESTDERIVEDLAYER 32
///#define LARGESTINPUTLAYER 32          // max of all the layers that feed into other layers
///#define TOTALDERIVEDNODES 58  /// the sum of the nodes from layer 1 onwards
///#define INITWIDTHARRAY {32,32,16,16}

typedef struct
{
    int firstNode;          /// Stores the index into the global array of the first node processed by this core
    int lastNode;           /// Stores the index into the global array  of the last node processed by this core
    int firstWeight;        /// Stores the index into the global array of weights of the first weight of the first node
    int lastWeight;         /// Stores the index into the global array of weights of the last weight of the last node
    int nodeIndexOffset;    /// Stores the index into the blobal array of the location of the first node in the layer
    int wgtIndexOffset;     /// Stores the index into the global array of the location of the first weight of the first node of the current layer
}   idx;                    /// idx is stored in an array for each layer

void forwardPass(   float * in,
                    float * g_nodeBiases,
                    float * biases,
                    float * g_weights,
                    float * wgt,
                    float * derived,
                    int   * widths,
                    idx * coreIndex,
           __global float * debug)
{
    int n, i, w;            /// node, input, weight
    int d = 0;              /// debug
    int gid = get_global_id(0);
    int layer;
    int localFirstNode, localLastNode;      /// the  index of the first and last nodes in the current layer
    int firstWeight, lastWeight;
//    int nodeIndexOffset = 0;
//    int wgtIndexOffset = 0;
    int destNodesPerCore, destNodesModulus;
    int curLayerWidth, prevLayerWidth;      /// convenience variables - saves having to do an array look up all the time
    float activationQuant;
    unsigned int core[] = {core00, core01, core02, core03, core10, core11, core12, core13, core20, core21, core22, core23, core30, core31, core32, core33};
    unsigned int coreI;
    unsigned int localCoreId = LOCAL_MEM_ADDRESS_BASE(gid);


    /// local storage
//    __private int   widths[] = INITWIDTHARRAY;

    if(gid==0)
        for (i=0;i<TOTALDERIVEDNODES;i++)
            debug[i] = 0;

    coreIndex[1].nodeIndexOffset = coreIndex[0].wgtIndexOffset = 0;     /// everything starts at 0
    for(layer = 1; layer<LAYERCOUNT; layer++)
    {
        curLayerWidth = widths[layer];
        prevLayerWidth = widths[layer-1];

        destNodesPerCore = curLayerWidth / CORECOUNT;                   /// all cores get this many
        destNodesModulus = curLayerWidth % CORECOUNT;                   /// the remainder are assigned one per node starting from gid == 0

        coreIndex[layer].firstNode = coreIndex[layer].nodeIndexOffset + ((gid * destNodesPerCore) + min(gid, destNodesModulus)); /// all node biases are in one big array so nodeIndexOffset records where the current layer starts
        coreIndex[layer].lastNode = coreIndex[layer].firstNode + destNodesPerCore + ((gid < destNodesModulus) ? 1 : 0);
        localFirstNode = coreIndex[layer].firstNode - coreIndex[layer].nodeIndexOffset;                   /// firstNode - nodeIndexOffset is the node index within the current  layer
        localLastNode = coreIndex[layer].lastNode - coreIndex[layer].nodeIndexOffset;                     /// localFirstNode and localLastNode align with the derived value array
        coreIndex[layer].firstWeight = coreIndex[layer].wgtIndexOffset + (localFirstNode * prevLayerWidth);
        coreIndex[layer].lastWeight = coreIndex[layer].firstWeight + ((localLastNode - localFirstNode) * prevLayerWidth);

      ///memcopy(...);     /// only copy in the g_weights that are needed for this node
//      memcpy(wgt, g_weights + (coreIndex[layer].firstWeight * sizeof(float)), (coreIndex[layer].lastWeight - coreIndex[layer].firstWeight));
        w=0;
        for (i = coreIndex[layer].firstWeight; i < coreIndex[layer].lastWeight; i++)
            wgt[w++] = g_weights[i];

        /// memcopy(..);
        if (layer > 1)                             /// input values to first hidden layer is passed in by the caller
        {
            n = coreIndex[layer].nodeIndexOffset - prevLayerWidth;    /// start from the begining of the previous layer's values in the derived value array
            for (i=0; i<prevLayerWidth; i++)
                in[i] = derived[n++];
        }

        ///memcopy(..);
        for (n = coreIndex[layer].firstNode; n < coreIndex[layer].lastNode; n++)
            biases[n] = g_nodeBiases[n];              /// allocate enough space for a whole bias vector in the layer but only copy the one this core needs


        firstWeight = 0;                            /// only the g_weights relevant to thse nodes have been copied into local memory
        lastWeight = prevLayerWidth;               /// check boundry condition on the very last weight into the output layer
        for (n = coreIndex[layer].firstNode; n < coreIndex[layer].lastNode; n++)
        {
            activationQuant = 0.0;
            i=0;                                    /// i is the index into the input vector which starts for 0 for every node;
            for (w=firstWeight; w<lastWeight; w++)
            {
                activationQuant += in[i++] * wgt[w];
            }

            derived[n] = (1.0 / (1.0 + (float)exp(-(biases[n] + activationQuant))));      // sigmoid function f(t) = 1/(1 + e^(-t))

            firstWeight = lastWeight;
            lastWeight += prevLayerWidth;
        }

//        if (layer < OUTPUTLAYER)
//        {
            /// transmit the node values calculated here to all other cores.
            for (coreI = 0; coreI < CORECOUNT; coreI++)
            {
                if (core[coreI] != localCoreId)
                    for (n=coreIndex[layer].firstNode; n < coreIndex[layer].lastNode; n++)
                        *(float *)NEIGHBOUR_LOC(core[coreI], derived,  n, (sizeof(float))) = derived[n];

            }
            /// make sure that every core has passed all values before proceeding onto the next layer
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

            if (layer < LAYERCOUNT)
            {
                coreIndex[layer + 1].nodeIndexOffset = coreIndex[layer].nodeIndexOffset + curLayerWidth; /// the length of the node bias array is the sum of the layer widths
                coreIndex[layer + 1].wgtIndexOffset = coreIndex[layer + 1].wgtIndexOffset + (curLayerWidth * prevLayerWidth);
            }
//       }
//       else
/*         if (layer == OUTPUTLAYER)
       {
            if (gid==0)
            {
                for (i = 1; i < LAYERCOUNT; i++)
                {
                    debug[d++] = (float)coreIndex[i].firstNode;
                    debug[d++] = (float)coreIndex[i].lastNode;
                    debug[d++] = (float)coreIndex[i].firstWeight;
                    debug[d++] = (float)coreIndex[i].lastWeight;
                    debug[d++] = 1000.00;
                }
                for (i=0;i<TOTALDERIVEDNODES;i++)
                    debug[d++] = derived[i];
            }
        }
 */   }
}

///======================================================================================================================

///         FEED FORWARD

///======================================================================================================================
__kernel void k_forward(    __global float * g_inVals,         /// incoming: the input values to the net
                            __global float * g_nodeBiases,     /// incoming: g_nodeBiases all in one big array
                            __global float * g_weights,        /// incoming: g_weights for all layers in one big array
                            __global float * g_outVals,        /// outgoing: the results of the run
                            __global float * debug)
{
    int n0, n;
    __private int   widths[] = INITWIDTHARRAY;
    __private idx   coreIndex[LAYERCOUNT];
    __private float in[LARGESTINPUTLAYER];
    __private float derived[TOTALDERIVEDNODES];
    __private float wgt[MAXWEIGHTTOLAYER];       /// space for local storage of weights ... is filled by the forward pass and used later to train
    __private float biases[TOTALDERIVEDNODES];


    for (n = 0; n < widths[0]; n++)
        in[n] = g_inVals[n];

    forwardPass(in, g_nodeBiases, biases, g_weights, wgt, derived, widths, coreIndex, debug);

    n0 = coreIndex[OUTPUTLAYER].firstNode - (TOTALDERIVEDNODES - widths[OUTPUTLAYER]);    /// convert the index of the final derived layer back to a zero base
    for(n=coreIndex[OUTPUTLAYER].firstNode; n<coreIndex[OUTPUTLAYER].lastNode; n++)
        g_outVals[n0++] = derived[n];        /// put the last derived vector into g_outVals for transmission to the host


}

///======================================================================================================================

///         TRAIN

///======================================================================================================================
__kernel void k_train(    __global float * g_inVals,          /// incoming: the input values to the new
                          __global float * g_desiredVals,     /// incoming: the desired outputvalues
                          __global float * g_nodeBiases,      /// incoming: g_nodeBiases all in one big array
                          __global float * g_weights,         /// incoming: g_weights for all layers in one big array
                          __global float * g_error,          /// outgoing: the cumulative differentials between the actual output and the deisred output
                          __global float   g_learningRate,
                          __global float * g_weightDeltas,
                          __global float * debug)
{
//    int firstNode, lastNode, localFirstNode, localLastNode;
    int n, layer_firstLocalNode, layer_localNodeIndexer, w;
    int prevLayer_firstGlobalNode, prevLayer_globalNodeIterator;
    int layer;                                          /// counts from n to 1
    int curLayerWidth, nextLayerWidth, prevLayerWidth, firstWeight, lastWeight;
//    int outboundNodesCoreGid;
//    int destNodesPerCore, destNodesModulus;
//    int nodeIndexOffset = 0;
//    int wgtIndexOffset = 0;
    int gid = get_global_id(0);
    int d = 0;

    float wErr, w0;         /// local copies of the weight error and the weight
    float learningRate = g_learningRate;

    __private idx   coreIndex[LAYERCOUNT];
    __private int   widths[] = INITWIDTHARRAY;
    __private float in[LARGESTINPUTLAYER];              /// local copy of the input values
    __private float derived[TOTALDERIVEDNODES];        // could restrict this to the width of the output layer
    __private float delta[LARGESTDERIVEDLAYER];        // could restrict this to the width of the output layer
    __private float outputError[LARGESTDERIVEDLAYER];       /// each core writes its error here  -  the node error is then the accumulation of these values
    __private float wgt[MAXWEIGHTTOLAYER];                  /// space for local storage of weights ... is filled by the forward pass and used later to train
    __private float biases[TOTALDERIVEDNODES];
//    __private float linkErrors[MAXWEIGHTTOLAYER];           /// SPACE FOR EACH CORE TO SEND THE PREVIOUS LAYER'S OUTBOUND LINK ERRORS // using debug[] for now

    unsigned int core[] = {core00, core01, core02, core03, core10, core11, core12, core13, core20, core21, core22, core23, core30, core31, core32, core33};

    for (n = 0; n < widths[0]; n++)
        in[n] = g_inVals[n];

    forwardPass(in, g_nodeBiases, biases, g_weights, wgt, derived, widths, coreIndex, debug);

    for (layer = OUTPUTLAYER; layer > 0; layer--)
    {
        prevLayerWidth = widths[layer - 1];
        curLayerWidth = widths[layer];

        layer_localNodeIndexer = layer_firstLocalNode = coreIndex[layer].firstNode - coreIndex[layer].nodeIndexOffset;
        if (layer == OUTPUTLAYER)
        {
            /// calculate the OUTPUT layer error
            for (n = coreIndex[OUTPUTLAYER].firstNode; n < coreIndex[OUTPUTLAYER].lastNode; n++)
            {
                outputError[layer_localNodeIndexer] = g_desiredVals[layer_localNodeIndexer] - derived[n];      /// width of desired == width outputlayer
                /// if (lastTrainingSet)
                    g_error[layer_localNodeIndexer] = outputError[layer_localNodeIndexer];                          /// pass the final deltas back
                delta[layer_localNodeIndexer] = derived[n] * (1 - derived[n]) * outputError[layer_localNodeIndexer];      /// calculate the weight update delta for each output node first derivative of the sigmoid function [Read and Marks pg65]
                layer_localNodeIndexer++;
            }
        }
        else
        {
            nextLayerWidth = widths[layer + 1];

            /// for each outbound weight
            firstWeight = coreIndex[layer].firstWeight;
            lastWeight = firstWeight + nextLayerWidth;

            for (n = coreIndex[layer].firstNode; n < coreIndex[layer].lastNode; n++)    // not sure about this
            {
                outputError[layer_localNodeIndexer] = 0;
                for (w = firstWeight; w < lastWeight; w++)          /// not sure if this is right - check the indexes into debug
                    outputError[layer_localNodeIndexer] += g_weightDeltas[w];   ///====================================================== Has to grab one weightDelta from each node
                delta[layer_localNodeIndexer] = derived[n] * (1 - derived[n]) * outputError[layer_localNodeIndexer];      /// Check this
                layer_localNodeIndexer++;
            }

        }

        /// online learning for now
        /// for each inbound weight
        firstWeight = coreIndex[layer].firstWeight;              /// update the __global g_weights array for now
        lastWeight = firstWeight + prevLayerWidth;               /// check boundry condition on the very last weight into the output layer/// the current node has one incoming weight for each node in the previous layer

        prevlayer_firstGlobalNode = coreindex[layer-1].firstNode;
        layer_localNodeIndexer = layer_firstLocalNode;
        for (n = coreIndex[layer].firstNode; n < coreIndex[layer].lastNode; n++)
        {
            prevLayer_globalNodeIterator = prevlayer_firstGlobalNode;       /// saves having to readdress the derived array
            for (w = firstWeight; w < lastWeight; w++)
            {
                //wgt[w] -= learningRate * delta[n] * derived[n];
                w0 = g_weights[w];
                debug[d++] = g_weights[w] = w0 + (learningRate * delta[layer_localNodeIndexer] * derived[prevLayer_globalNodeIterator++]);  /// update the global weight array for now  --  hsould I multiply the weight error by the learning rate here or one line above?

    /* This bit is to send the weight errors directly to the owning node in the previous layer
                /// pass delta * weight to previous layer
                if (outboundNodesCoreGid < destNodesModulus)        // relies on the observation that the first method will work for the first weight sent to the first core without an extra node will still work
                    outboundNodesCoreGid = (int)floor((float)(w/(destNodesPerCore + 1)));
                else
                    outboundNodesCoreGid = (int)(CORECOUNT - ceil((float)(((prevLayerWidth + 1) - w) / destNodesPerCore)));

                *(float *)NEIGHBOUR_LOC(core[outboundNodesCoreGid], linkErrors, (w), (sizeof(float))) = (delta[n] * wgt[w]);  /// <<<<<<<<<<<<<<< w is not correct
    //            if(gid == 0)
    //                debug[d++] = wgt[w];
    */
     //               linkErrors[(n * curLayerWidth) + w] = (delta[n] * wgt[w]);
                /// Use g_weightDeltas to communication between cores for now
                g_weightDeltas[w] = (delta[layer_localNodeIndexer] * w0);      /// sotre the delta * un-updated weight in an array that is parallel to the weight array
                debug[d++] = (delta[layer_localNodeIndexer] * w0);
                debug[d++] = 1000.0;
            }

            /// update the node bias
            biases[n] -= learningRate * outputError[layer_localNodeIndexer];

            firstWeight = lastWeight;
            lastWeight += prevLayerWidth;
            layer_localNodeIndexer++;
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);        /// pause for every core to catch up before going onto the next layer
    }
}
