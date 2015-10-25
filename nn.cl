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
///#define TOTALNODES 58  /// the sum of the nodes from layer 1 onwards
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

void forwardPass(   float * biases,
                    float * wgt,
                    float * derived,
                    int   * widths,
                    idx * coreIndex,
           __global float * debug)
{
    int n, w;            /// node, input, weight
//    int d = 0;              /// debug
    int layer;
    int firstWeight, lastWeight;
    int destNodesPerCore, destNodesModulus;
    int curLayerWidth, prevLayerWidth;      /// convenience variables - saves having to do an array look up all the time
    int prevLayerOutput = 0;                /// index into dervied[] where the previous layer's output start (0 for the input layer)
    float activationQuant;

    unsigned int core[] = {core00, core01, core02, core03, core10, core11, core12, core13, core20, core21, core22, core23, core30, core31, core32, core33};
    unsigned int coreI;
    int gid = get_global_id(0);
    unsigned int localCoreId = LOCAL_MEM_ADDRESS_BASE(gid);


//    if(gid==0)
//        for (d=0;d<TOTALNODES;d++)
//            debug[d] = 0;

    firstWeight = 0;

    for(layer = 1; layer<LAYERCOUNT; layer++)
    {
        prevLayerWidth = widths[layer - 1];
        lastWeight = firstWeight + prevLayerWidth;
//        if (gid == 0 )
//        {
//            debug[d++] = layer;
//            debug[d++] = prevLayerWidth;
//            debug[d++] = firstWeight;
//            debug[d++] = lastWeight;
//            debug[d++] = 1000;
//        }

        for (n = coreIndex[layer].firstNode; n < coreIndex[layer].lastNode; n++)
        {
            activationQuant = 0.0;
            prevLayerOutput = coreIndex[layer-1].nodeIndexOffset;       /// the location in derived[] that stores the first output from the previous layer

            for (w=firstWeight; w<lastWeight; w++)
            {
                activationQuant += derived[prevLayerOutput] * wgt[w];
//                if (gid == 0)
//                {
//                    debug[d++] = activationQuant;
//                    debug[d++] = derived[prevLayerOutput];
//                    debug[d++] = wgt[w];
//                    debug[d++] =  1000;
//                }

                prevLayerOutput++;
            }

            derived[n] = (1.0 / (1.0 + (float)exp(-(biases[n] + activationQuant))));      // sigmoid function f(t) = 1/(1 + e^(-t))
//                if (gid == 0)
//                {
//                    debug[d++] = derived[n];
//                    debug[d++] = biases[n];
//                    debug[d++] =  1000;
//                }


//            if (gid == 0) debug[d++] = 1000;
            firstWeight = lastWeight;
            lastWeight += prevLayerWidth;
//            if (gid == 0)
//            {
//                debug[d++] = firstWeight;
//                debug[d++] = lastWeight;
//                debug[d++] = 1000;
//            }
        }

        /// transmit the node values calculated here to all other cores. (needed for training only)
        for (coreI = 0; coreI < CORECOUNT; coreI++)
        {
            if (core[coreI] != localCoreId)
                for (n=coreIndex[layer].firstNode; n < coreIndex[layer].lastNode; n++)
                    *(float *)NEIGHBOUR_LOC(core[coreI], derived,  n, (sizeof(float))) = derived[n];

        }
        /// make sure that every core has passed all values before proceeding onto the next layer
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

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
                for (i=0;i<TOTALNODES;i++)
                    debug[d++] = derived[i];
            }
        }
 */   }
}

void copyIn(float * g_inVals,
            float * g_nodeBiases,
            float * biases,
            float * g_weights,
            float * wgt,
            float * derived,
            int   * widths,
            idx * coreIndex,
   __global float * debug)
{
    int n, i;           /// node, input,
    int w = 0;          /// weight index
    int d = 0;              /// debug
    int gid = get_global_id(0);
    int layer;
    int localFirstNode, localLastNode;      /// the  index of the first and last nodes in the current layer
    int firstWeight, lastWeight;
    int destNodesPerCore, destNodesModulus;
    int curLayerWidth, prevLayerWidth;      /// convenience variables - saves having to do an array look up all the time

    for (n = 0; n < widths[0]; n++)
    {
        derived[n] = g_inVals[n];
    }

    coreIndex[0].nodeIndexOffset = 0;
    coreIndex[1].nodeIndexOffset = widths[0];
    coreIndex[0].wgtIndexOffset = 0;            /// not used
    coreIndex[1].wgtIndexOffset = 0;            /// no weights into the zeroth layer so layer 1 starts from 0
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

/*        if (gid == 14)
        {
            debug[d++] = layer;
            debug[d++] = coreIndex[layer].firstNode;
            debug[d++] = coreIndex[layer].lastNode;
            debug[d++] = coreIndex[layer].firstWeight;
            debug[d++] = coreIndex[layer].lastWeight;
//            debug[d++] = 1000;
        }
*/

      ///memcopy(...);     /// only copy in the g_weights that are needed to calculate the nodes assigned to this core
//      memcpy(wgt, g_weights + (coreIndex[layer].firstWeight * sizeof(float)), (coreIndex[layer].lastWeight - coreIndex[layer].firstWeight));
//        debug[d++] = layer;
        for (i = coreIndex[layer].firstWeight; i < coreIndex[layer].lastWeight; i++)
        {
            wgt[w] = g_weights[i];
//            if (gid == 14) debug[d++] = wgt[w];
            w++;
        }
//        if (gid == 14) debug[d++] = 1000;

        ///memcopy(..);
        for (n = coreIndex[layer].firstNode; n < coreIndex[layer].lastNode; n++)
            biases[n] = g_nodeBiases[n - widths[0]];              /// allocate enough space for a whole bias vector in the layer but only copy the one this core needs

        if (layer < OUTPUTLAYER)     /// set up for the next pass
        {
            coreIndex[layer + 1].nodeIndexOffset = coreIndex[layer].nodeIndexOffset + curLayerWidth; /// the length of the node bias array is the sum of the layer widths
            coreIndex[layer + 1].wgtIndexOffset = coreIndex[layer].wgtIndexOffset + (curLayerWidth * prevLayerWidth);
        }


    }

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
    __private float derived[TOTALNODES]; /// derived[] and biases[] are maintained in parallel - derived[] contanins a copy of the input values g_inVals[] and biases are blank on those indexes
    __private float biases[TOTALNODES];
    __private float wgt[MAXWEIGHTSPERCORE];       /// space for local storage of weights ... is filled by the forward pass and used later to train


    copyIn(g_inVals, g_nodeBiases, biases, g_weights, wgt, derived, widths, coreIndex, debug);
    forwardPass(biases, wgt, derived, widths, coreIndex, debug);

    /// Copy Out
    n0 = coreIndex[OUTPUTLAYER].firstNode - (TOTALNODES - widths[OUTPUTLAYER]);    /// convert the index of the final derived layer back to a zero base
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
    int n, layer_firstLocalNode, layer_localNodeIndexer, w;
    int prevLayer_firstGlobalNode, prevLayer_globalNodeIterator;
    int nextLayer_firstGlobalWeight;
    int layer;                                          /// counts from n to 1
    int curLayerWidth, nextLayerWidth, prevLayerWidth, firstWeight, lastWeight;

    int gid = get_global_id(0);
    int d = 0;

    float wErr, w0;         /// local copies of the weight error and the weight
    float learningRate = g_learningRate;
    float outputError;       /// temporary storage before working out the delta for each node

    __private idx   coreIndex[LAYERCOUNT];
    __private int   widths[] = INITWIDTHARRAY;
//    __private float in[LARGESTINPUTLAYER];              /// local copy of the input values
    __private float derived[TOTALNODES];        // could restrict this to the width of the output layer
    __private float delta[LARGESTDERIVEDLAYER];        // could restrict this to the width of the output layer
    __private float wgt[MAXWEIGHTSPERCORE];                  /// space for local storage of weights ... is filled by the forward pass and used later to train
    __private float biases[TOTALNODES];

    unsigned int core[] = {core00, core01, core02, core03, core10, core11, core12, core13, core20, core21, core22, core23, core30, core31, core32, core33};

    copyIn(g_inVals, g_nodeBiases, biases, g_weights, wgt, derived, widths, coreIndex, debug);
    forwardPass(biases, wgt, derived, widths, coreIndex, debug);

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
                outputError = g_desiredVals[layer_localNodeIndexer] - derived[n];      /// width of desired == width outputlayer
                /// if (lastTrainingSet)
                    g_error[layer_localNodeIndexer] = outputError;                          /// pass the final deltas back
                delta[layer_localNodeIndexer] = derived[n] * (1 - derived[n]) * outputError;      /// calculate the weight update delta for each output node first derivative of the sigmoid function [Read and Marks pg65]
                layer_localNodeIndexer++;
            }
        }
        else
        {
            nextLayerWidth = widths[layer + 1];

            /// for each outbound weight - i.e. for each inboudn weight of the next layer
            nextLayer_firstGlobalWeight = coreIndex[layer + 1].wgtIndexOffset;

            for (n = coreIndex[layer].firstNode; n < coreIndex[layer].lastNode; n++)    // not sure about this
            {
                outputError = 0;
                for (w = 0; w < nextLayerWidth; w++)
                    outputError += g_weightDeltas[nextLayer_firstGlobalWeight + ( w * curLayerWidth) + layer_localNodeIndexer];   /// g_weightDeltas[] mirrors g_weights[] in that weightDeltas are organised around the INCOMING weights of the next layer
                delta[layer_localNodeIndexer] = derived[n] * (1 - derived[n]) * outputError;                                      /// therefore to pick out the deltas for the current layer you need to pick out the node's delta from each section of the array associated with each next layer node
                layer_localNodeIndexer++;
            }
        }

        /// online learning for now
        /// for each inbound weight
        firstWeight = coreIndex[layer].firstWeight;              /// update the __global g_weights array for now
        lastWeight = firstWeight + prevLayerWidth;               /// check boundry condition on the very last weight into the output layer/// the current node has one incoming weight for each node in the previous layer

        prevLayer_firstGlobalNode = coreIndex[layer-1].firstNode;   /// layer zero (input layer) is also in derived[]
        layer_localNodeIndexer = layer_firstLocalNode;
        for (n = coreIndex[layer].firstNode; n < coreIndex[layer].lastNode; n++)
        {
            prevLayer_globalNodeIterator = prevLayer_firstGlobalNode;       /// saves having to readdress the coreIndex array
            for (w = firstWeight; w < lastWeight; w++)
            {
                w0 = g_weights[w];
                debug[d++] = g_weights[w] = w0 + (learningRate * delta[layer_localNodeIndexer] * derived[prevLayer_globalNodeIterator++]);  /// LR * delta * PREVIOUS LAYER OUTPUT  (input layer is now the first part of derived[])

                /// Use g_weightDeltas to communication between cores for now
                g_weightDeltas[w] = (delta[layer_localNodeIndexer] * w0);      /// sotre the delta * un-updated weight in an array that is parallel to the weight array
                debug[d++] = (delta[layer_localNodeIndexer] * w0);
                debug[d++] = 1000.0;
            }

            /// update the node bias
            biases[n] += learningRate * delta[layer_localNodeIndexer];

            firstWeight = lastWeight;
            lastWeight += prevLayerWidth;
            layer_localNodeIndexer++;
        }
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);        /// pause for every core to catch up before going onto the next layer
    }
}
