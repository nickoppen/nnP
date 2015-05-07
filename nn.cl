#include "/home/parallella/Work/nnP/coreId16.inc"
//#include <e32_opencl_ext.h>
//#include <coprthr_device.h>

#include "/home/parallella/Work/nnP/cldefs.inc"
/// cldefs.inc contains #defines for all static variables
/// example contents of cldefs.inc
///#define CORECOUNT 16
///#define LAYERCOUNT 4
///#define OUTPUTLAYER 3                 // LAYERCOUNT -1
///#define MAXWEIGHTTOLAYER 1024
///#define LARGESTDERIVEDLAYER 32
///#define LARGESTINPUTLAYER 32          // max of all the layers that feed into other layers
///#define INITWIDTHARRAY {32,32,16,16}/


__kernel void k_forward(    __global float * inVals,
                            __global float * outVals,
                            __global float * debug,
                            __global float * nodeBiases,     // nodeBiases all in one big array
                            __global float * weights)        // weights for all layers in one big array
{
    int n, i, w;            /// node, input, weight
    int d = 0;              /// debug
    int gid = get_global_id(0);
    int layer;
    int firstNode, lastNode;                /// the index of the first and last nodes in the __global node array
    int localFirstNode, localLastNode;      /// the  index of the first and last nodes in the current layer
    int firstWeight, lastWeight;
    int nodeIndexOffset = 0;
    int wgtIndexOffset = 0;
    int destNodesPerCore, destNodesModulus;
    int curLayerWidth, prevLayerWidth;      /// convenience variables - saves having to do an array look up all the time
    float activationQuant;
    unsigned int core[] = {core00, core01, core02, core03, core10, core11, core12, core13, core20, core21, core22, core23, core30, core31, core32, core33};
    unsigned int coreI;
    unsigned int localCoreId = LOCAL_MEM_ADDRESS_BASE(gid);


    /// local storage
    __private int   widths[] = INITWIDTHARRAY;
    __private float wgt[MAXWEIGHTTOLAYER];
    __private float biases[LARGESTDERIVEDLAYER];
    __private float in[LARGESTINPUTLAYER];
    __private float derived[LARGESTDERIVEDLAYER];

    for(layer = 1; layer<LAYERCOUNT; layer++)
    {
        curLayerWidth = widths[layer];
        prevLayerWidth = widths[layer-1];

        destNodesPerCore = curLayerWidth / CORECOUNT;                   /// all cores get this many
        destNodesModulus = curLayerWidth % CORECOUNT;                   /// the remainder are assigned one per node starting from gid == 0

        firstNode = nodeIndexOffset + ((gid * destNodesPerCore) + min(gid, destNodesModulus)); /// all node biases are in one big array so nodeIndexOffset records where the current layer starts
        lastNode = firstNode + destNodesPerCore + ((gid < destNodesModulus) ? 1 : 0);
        localFirstNode = firstNode - nodeIndexOffset;                   /// firstNode - nodeIndexOffset is the node index within the current  layer
        localLastNode = lastNode - nodeIndexOffset;                     /// localFirstNode and localLastNode align with the derived value attay
        firstWeight = wgtIndexOffset + (localFirstNode * prevLayerWidth);
        lastWeight = firstWeight + ((lastNode - firstNode) * prevLayerWidth);

      ///memcopy(...);     /// only copy in the weights that are needed for this node
        w=0;
        for (i=firstWeight; i<lastWeight; i++)
            wgt[w++] = weights[i];

        /// memcopy(..);
        if (layer == 1)                             /// input layer to first hidden layer
            for (i=0; i<widths[0]; i++)
                in[i] = inVals[i];
        else                                        /// all other layers
            for (i=0; i<prevLayerWidth; i++)
                in[i] = derived[i];

        if (gid == 0)
        {
            for (i=0; i<prevLayerWidth; i++)
                debug[d++] = in[i];
            debug[d++] = 1000.0;
        }

            /// testing - inialise the derived layer to see what values have ben calculated
        for (i=0; i<LARGESTDERIVEDLAYER; i++)
            derived[i]= (float)1.0;

        ///memcopy(..);
        n = localFirstNode;
        for (i=firstNode; i<lastNode; i++)
            biases[n++] = nodeBiases[i];              /// allocate enough space for a whole bias vector in the layer but only copy the one this core needs


        firstWeight = 0;                            /// only the weights relevant to thse nodes have been copied into local memory
        lastWeight = prevLayerWidth;               /// check boundry condition on the very last weight into the output layer
        for (n=localFirstNode; n<localLastNode; n++)
        {
            activationQuant = 0.0;
            i=0;                                    /// i is the index into the input vector which starts for 0 for every node;
            for (w=firstWeight; w<lastWeight; w++)
            {
//                if (gid == 15)
//                {
//                    debug[d++] = in[i];
//                    debug[d++] = wgt[w];
//                    debug[d++] = in[i] * wgt[w];  /// test
//                }
                activationQuant += in[i++] * wgt[w];
            }

//            if (gid == 9) /// test
//            {
//                debug[d++] = activationQuant;
//                debug[d++] = biases[n];
//                debug[d++]  = (1.0 / (1.0 + (float)exp(-(biases[n] + activationQuant))));
//            }
            derived[n] = (1.0 / (1.0 + (float)exp(-(biases[n] + activationQuant))));      // sigmoid function f(t) = 1/(1 + e^(-t))

            firstWeight = lastWeight;
            lastWeight += prevLayerWidth;
        }

        if (layer < OUTPUTLAYER)
        {
            /// transmit the node values calculated here to all other cores.
            for (coreI = 0; coreI < CORECOUNT; coreI++)
            {
                if (core[coreI] != localCoreId)
                    for (n=localFirstNode; n < localLastNode; n++)
                        *(float *)NEIGHBOUR_LOC(core[coreI], derived,  n, (sizeof(float))) = derived[n];

//              debug - watch the derived values arriving at core 0 from the other nodes
//                if (gid == 0)
//                    for (i=0; i<curLayerWidth; i++)
//                        debug[d++] = derived[i];
            }
            /// make sure that every core has passed all values before proceeding onto the next layer
            barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
        }
        else
        {
            for(n=localFirstNode; n<localLastNode; n++)
                outVals[n] = derived[n];        /// put the last derived vector into outVals for transmission to the host
        }

        nodeIndexOffset += curLayerWidth; /// the length of the node bias array is the sum of the layer widths
        wgtIndexOffset += curLayerWidth * prevLayerWidth;
    }
}
