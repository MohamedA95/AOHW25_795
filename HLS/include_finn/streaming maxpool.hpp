#ifndef CUSTOM_MAXPOOL_H
#define CUSTOM_MAXPOOL_H

#include <limits>

#include "interpret.hpp"
#include "pool.hpp"
#include "maxpool.h"

/**
 * \brief 2D max pool function function
 *
 * The function performs a max pool operation and works in conjuction 
 * with a sliding window unit performing im2col on the input data, allowing 
 * generic kernel and stride values. Also it supports input parallelism and arbitrary input and output widths
 * Modified from the original implementation by Xilinx.
 *
 * \tparam IFMChannels 		Number of Input Feature Maps
 * \tparam IFMDim 			  Width and Height of the Input Feature Map (assumed square)
 * \tparam PE             Number of channels in the pool layer computed in parallel
 * \tparam PoolDim        Dimension of the Max Pool kernel
 * \tparam Stride         Stride of the max pool kernel
 * \tparam TSrcI          DataType of the input value (Slice)
 * \tparam ActType        DataType of the input value, without Slice (e.g. ap_fixed<32,16>)
 * \tparam InStreamW      Width, in number of bits, of the input stream
 * \tparam OutStreamW     Width, in number of bits, of the output stream
 *
 * \param in              Input stream
 * \param out             Output stream
 * \param reps            Number of time the function has to be repeatedly executed (e.g. number of images)
 */
template<
  unsigned int IFMChannels,
  unsigned int IFMDim,
  unsigned int PE,
  unsigned int PoolDim,
  unsigned int Stride,
  typename TSrcI,
  typename ActType,
  int InStreamW,
  int OutStreamW
>
void Streaming_Maxpool_batch2d(hls::stream<ap_uint<InStreamW>> &in,hls::stream<ap_uint<OutStreamW>> &out,int const  reps) {
    #pragma HLS INLINE
    const int FOLD = IFMChannels/PE;
    unsigned const InpPerImage = IFMDim * IFMDim * IFMChannels * TSrcI::width / InStreamW;
    hls::stream<ap_uint<PE*TSrcI::width> > input_generator_out("Streaming_Maxpool_batch2d.input_generator_out");
#pragma HLS STREAM variable=input_generator_out depth=8
    auto const mem_r = ap_resource_dflt();

    ConvolutionInputGenerator_dws<PoolDim, IFMChannels,TSrcI::width, IFMDim, IFMDim/PoolDim, PE,Stride>(in, input_generator_out, reps, mem_r);

    MaxPoolFunction<ActType,PoolDim> maxpool_fxn;
    Pool_batch<IFMChannels, PE, PoolDim*PoolDim,TSrcI, TSrcI>(input_generator_out, out, maxpool_fxn, (IFMDim/PoolDim)*(IFMDim/PoolDim)*reps);

}
#endif