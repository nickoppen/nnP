#include "/home/parallella/Work/nnP/coreId16.inc"
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
    int globalStartNode;          /// Stores the index into the global array of the first node processed by this core
    int globalEndNode;           /// Stores the index into the global array  of the last node processed by this core
    int globalStartWeight;        /// Stores the index into the global array of weights of the first weight of the first node
    int globalEndWeight;         /// Stores the index into the global array of weights of the last weight of the last node
    int globalNodeZeroForLayer;    /// Stores the index into the blobal array of the location of the first node in the layer
    int globalWgtZeroForLayer;     /// Stores the index into the global array of the location of the first weight of the first node of the current layer
}   idx;                    /// idx is stored in an array for each layer

void forwardPass(   float * biases,
                    float * wgt,
                    float * derived,
                    int   * widths,
                    idx * coreIndex//, __global float * debug
                )
{
    int n, w;            /// node, weight
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


    firstWeight = 0;        /// called firstWeight bacause every weight is used to calculate the node value

    for(layer = 1; layer<LAYERCOUNT; layer++)
    {
        prevLayerWidth = widths[layer - 1];
        lastWeight = firstWeight + prevLayerWidth;

        for (n = coreIndex[layer].globalStartNode; n < coreIndex[layer].globalEndNode; n++)
        {
            activationQuant = 0.0;
            prevLayerOutput = coreIndex[layer-1].globalNodeZeroForLayer;       /// the location in derived[] that stores the first output from the previous layer

            for (w=firstWeight; w<lastWeight; w++)
            {
                activationQuant += derived[prevLayerOutput] * wgt[w];
                prevLayerOutput++;
            }

            derived[n] = (1.0 / (1.0 + (float)exp(-(biases[n] + activationQuant))));      // sigmoid function f(t) = 1/(1 + e^(-t))

            firstWeight = lastWeight;
            lastWeight += prevLayerWidth;
        }

        /// transmit the node values calculated here to all other cores. (needed for training only)
        for (coreI = 0; coreI < CORECOUNT; coreI++)
        {
            if (core[coreI] != localCoreId)
                for (n=coreIndex[layer].globalStartNode; n < coreIndex[layer].globalEndNode; n++)
                    *(float *)NEIGHBOUR_LOC(core[coreI], derived,  n, (sizeof(float))) = derived[n];

        }
        /// make sure that every core has passed all values before proceeding onto the next layer
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    }
}

void copyIn(float * g_inVals,
            float * g_nodeBiases,
            float * biases,
            float * g_weights,
            float * wgt,
            float * derived,
            int   * widths,
            idx   * coreIndex,
            int   * p_d,
   __global float * debug)
{
    int n, i;           /// node, input,
    int w = 0;          /// weight index
    int d = 0;          /// debug
    int gid = get_global_id(0);
    int layer;
    int layerStartNode, layerEndNode;      /// the  index of the first and last nodes in the current layer
    int destNodesPerCore, destNodesModulus;
    int curLayerWidth, prevLayerWidth;      /// convenience variables - saves having to do an array look up all the time

    /// Copy the input values into derived[] so that they can be treated in the same way as a hidden layer output
    for (n = 0; n < widths[0]; n++)
    {
        derived[n] = g_inVals[n];
    }

    coreIndex[0].globalNodeZeroForLayer = 0;
    coreIndex[1].globalNodeZeroForLayer = widths[0];   /// make sure to start
    coreIndex[0].globalWgtZeroForLayer = 0;            /// not used
    coreIndex[1].globalWgtZeroForLayer = 0;            /// no weights into the zeroth layer so layer 1 starts from 0

    for(layer = 1; layer<LAYERCOUNT; layer++)
    {
        curLayerWidth = widths[layer];
        prevLayerWidth = widths[layer-1];

        destNodesPerCore = curLayerWidth / CORECOUNT;                   /// all cores get this many
        destNodesModulus = curLayerWidth % CORECOUNT;                   /// the remainder are assigned one per node starting from gid == 2

        coreIndex[layer].globalStartNode = coreIndex[layer].globalNodeZeroForLayer + ((gid * destNodesPerCore) + min(gid, destNodesModulus)); /// all node biases are in one big array so globalNodeZeroForLayer records where the current layer starts
        coreIndex[layer].globalEndNode = coreIndex[layer].globalStartNode + destNodesPerCore + ((gid < destNodesModulus) ? 1 : 0);
        layerStartNode = coreIndex[layer].globalStartNode - coreIndex[layer].globalNodeZeroForLayer;                   /// startNode - globalNodeZeroForLayer is the node index within the current  layer
        layerEndNode = coreIndex[layer].globalEndNode - coreIndex[layer].globalNodeZeroForLayer;                     /// layerStartNode and layerEndNode align with the derived value array
        coreIndex[layer].globalStartWeight = coreIndex[layer].globalWgtZeroForLayer + (layerStartNode * prevLayerWidth);
        coreIndex[layer].globalEndWeight = coreIndex[layer].globalStartWeight + ((layerEndNode - layerStartNode) * prevLayerWidth);

      ///memcopy(...);     /// only copy in the g_weights that are needed to calculate the nodes assigned to this core
//      memcpy(wgt, g_weights + (coreIndex[layer].globalStartWeight * sizeof(float)), (coreIndex[layer].globalEndWeight - coreIndex[layer].globalStartWeight));
        for (i = coreIndex[layer].globalStartWeight; i < coreIndex[layer].globalEndWeight; i++)
        {
            wgt[w] = g_weights[i];
            w++;
        }

        ///memcopy(..);
        for (n = coreIndex[layer].globalStartNode; n < coreIndex[layer].globalEndNode; n++)
            biases[n] = g_nodeBiases[n - widths[0]];              /// allocate enough space for a whole bias vector in the layer but only copy the one this core needs

        if (layer < OUTPUTLAYER)     /// set up for the next pass
        {
            coreIndex[layer + 1].globalNodeZeroForLayer = coreIndex[layer].globalNodeZeroForLayer + curLayerWidth; /// the length of the node bias array is the sum of the layer widths
            coreIndex[layer + 1].globalWgtZeroForLayer = coreIndex[layer].globalWgtZeroForLayer + (curLayerWidth * prevLayerWidth);
        }


    }
    (*p_d) = d;

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
    int d = 0;
    __private int   widths[] = INITWIDTHARRAY;
    __private idx   coreIndex[LAYERCOUNT];
    __private float derived[TOTALNODES]; /// derived[] and biases[] are maintained in parallel - derived[] contanins a copy of the input values g_inVals[] and biases are blank on those indexes
    __private float biases[TOTALNODES];
    __private float wgt[MAXWEIGHTSPERCORE];       /// space for local storage of weights ... is filled by the forward pass and used later to train


    copyIn(g_inVals, g_nodeBiases, biases, g_weights, wgt, derived, widths, coreIndex, &d, debug);
    forwardPass(biases, wgt, derived, widths, coreIndex);//, debug);

    /// Copy Out
    n0 = coreIndex[OUTPUTLAYER].globalStartNode - (TOTALNODES - widths[OUTPUTLAYER]);    /// convert the index of the final derived layer back to a zero base
    for(n=coreIndex[OUTPUTLAYER].globalStartNode; n<coreIndex[OUTPUTLAYER].globalEndNode; n++)
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
    int n;      /// indexes the global node array
    int w;
    int layerStartNode, layerNodeIterator;  /// indexes the node local to the layer
    int prevLayer_globalNodeIterator;
    int nextLayer_globalWgtZero;
    int layer;                                          /// counts from n to 1
    int curLayerWidth, nextLayerWidth, prevLayerWidth, firstWeight, lastWeight;

    int gid = get_global_id(0);
    int d = 0;

    float unmodifiedWeight;         /// local copies of the weight error and the weight
    float learningRate = g_learningRate;
    float outputError;       /// temporary storage before working out the delta for each node

    __private idx   coreIndex[LAYERCOUNT];
    __private int   widths[] = INITWIDTHARRAY;
    __private float derived[TOTALNODES];        // could restrict this to the width of the output layer
    __private float delta[LARGESTDERIVEDLAYER];        // could restrict this to the width of the output layer
    __private float wgt[MAXWEIGHTSPERCORE];                  /// space for local storage of weights ... is filled by the forward pass and used later to train
    __private float biases[TOTALNODES];

    unsigned int core[] = {core00, core01, core02, core03, core10, core11, core12, core13, core20, core21, core22, core23, core30, core31, core32, core33};

    copyIn(g_inVals, g_nodeBiases, biases, g_weights, wgt, derived, widths, coreIndex, &d, debug);
    forwardPass(biases, wgt, derived, widths, coreIndex);//, debug);

    for (layer = OUTPUTLAYER; layer > 0; layer--)
    {
        prevLayerWidth = widths[layer - 1];
        curLayerWidth = widths[layer];

        layerStartNode = coreIndex[layer].globalStartNode - coreIndex[layer].globalNodeZeroForLayer;   /// store this sot that it only has to be calculated once
        layerNodeIterator = layerStartNode;
        if (layer == OUTPUTLAYER)
        {
            /// calculate the OUTPUT layer error
            for (n = coreIndex[OUTPUTLAYER].globalStartNode; n < coreIndex[OUTPUTLAYER].globalEndNode; n++)
            {
                outputError = g_desiredVals[layerNodeIterator] - derived[n];      /// width of desired == width outputlayer
                /// if (lastTrainingSet)
                    g_error[layerNodeIterator] = outputError;                          /// pass the final deltas back
                delta[layerNodeIterator] = derived[n] * (1 - derived[n]) * outputError;      /// calculate the weight update delta for each output node first derivative of the sigmoid function [Read and Marks pg65]
                layerNodeIterator++;
            }
        }

        else
        {
            nextLayerWidth = widths[layer + 1];

            /// for each outbound weight - i.e. for each inbound weight of the next layer
            nextLayer_globalWgtZero = coreIndex[layer + 1].globalWgtZeroForLayer;

            for (n = coreIndex[layer].globalStartNode; n < coreIndex[layer].globalEndNode; n++)    // not sure about this
            {
                outputError = 0;
                for (w = 0; w < nextLayerWidth; w++)
                {
                    outputError += g_weightDeltas[nextLayer_globalWgtZero + ( w * curLayerWidth) + layerNodeIterator];   /// g_weightDeltas[] mirrors g_weights[] in that weightDeltas are organised around the INCOMING weights of the next layer
                }
                delta[layerNodeIterator] = derived[n] * (1 - derived[n]) * outputError;                                  /// therefore to pick out the deltas for the current layer you need to pick out the node's delta from each section of the array associated with each next layer node

                layerNodeIterator++;
            }
        }

        /// online learning for now
        /// for each inbound weight
        firstWeight = coreIndex[layer].globalStartWeight;              /// update the __global g_weights array for now
        lastWeight = firstWeight + prevLayerWidth;               /// the current node has one incoming weight for each node in the previous layer

        layerNodeIterator = layerStartNode;

        for (n = coreIndex[layer].globalStartNode; n < coreIndex[layer].globalEndNode; n++)
        {
            prevLayer_globalNodeIterator = coreIndex[layer-1].globalNodeZeroForLayer;     /// globalNodeZeroForLayer is the first node of the whole layer - layer zero (input layer) is also in derived[]
            for (w = firstWeight; w < lastWeight; w++)
            {
                unmodifiedWeight = g_weights[w];
                g_weights[w] = unmodifiedWeight + (learningRate * delta[layerNodeIterator] * derived[prevLayer_globalNodeIterator]); /// updated weight = LR * delta * PREVIOUS LAYER OUTPUT  (input layer is now the first part of derived[])
                /// Use g_weightDeltas to communication between cores for now
                g_weightDeltas[w] = (delta[layerNodeIterator] * unmodifiedWeight);      /// sotre the delta * un-updated weight in an array that is parallel to the weight array

                prevLayer_globalNodeIterator++;
            }

            /// update the node bias
            biases[n] += learningRate * delta[layerNodeIterator];
            g_nodeBiases[n - widths[0]] = biases[n];         /// return the updated node biases to the host -- one by one for now

            firstWeight = lastWeight;
            lastWeight += prevLayerWidth;
            layerNodeIterator++;
        }

        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);        /// pause for every core to catch up before going onto the next layer
    }
}
