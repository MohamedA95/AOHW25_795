/******************************************************************************
 *  Copyright (c) 2019, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *******************************************************************************/

/*******************************************************************************
 *
 *  Authors: Giulio Gambardella <giuliog@xilinx.com>
 *           Thomas B. Preusser <thomas.preusser@utexas.edu>
 *             Marie-Curie Fellow, Xilinx Ireland, Grant Agreement No. 751339
 *           Christoph Doehring <cdoehrin@xilinx.com>
 *
 *  \file mvau.hpp
 *
 *  This file lists a templated funtion used to implement
 *  Matrix-Vector-Activation Unit
 *
 *  This project has received funding from the European Union's Framework
 *  Programme for Research and Innovation Horizon 2020 (2014-2020) under
 *  the Marie Skłodowska-Curie Grant Agreement No. 751339.
 *
 *******************************************************************************/

#ifndef MVAU_HPP
#define MVAU_HPP

#include "hls_stream.h"

#include "mac.hpp"

/**
 * \brief Matrix vector activate function
 *
 * The function performs the multiplication between a weigth matrix and the input activation vector,
 * accumulating the results and then applying an activation function on the accumulated result.
 *
 *
 * \tparam MatrixW        Width of the input matrix
 * \tparam MatrixH        Heigth of the input matrix
 * \tparam SIMD           Number of input columns computed in parallel
 * \tparam PE             Number of output rows computed in parallel
 * \tparam Tw             DataType of a single weight
 * \tparam Ti             DataType of a single input activation
 * \tparam To             DataType of a single output activation
 * \tparam TI             DataType of the input stream - safely deducible from the paramaters !!!: safely deducible
 * \tparam TO             DataType of the output stream - safely deducible from the paramaters !!!: safely deducible
 * \tparam TW             DataType of the weights matrix - safely deducible from the paramaters !!!: safely deducible
 * \tparam TA             DataType of the activation class (e.g. thresholds) - safely deducible from the paramaters !!!: safely deducible
 * \tparam R              Datatype for the resource used for FPGA implementation of the MAC  - safely deducible from the paramaters !!!: safely deducible
 *
 * \param in              Input stream
 * \param out             Output stream
 * \param weights         Weights matrix (currently supports BinaryWeights or FixedPointWeights)
 * \param activation      Activation class
 * \param inp_zero_point  Zero point of the input data
 * \param reps            Number of time the function has to be repeatedly executed (e.g. number of images)
 * \param r               Resource type for the hardware implementation of the MAC block
 * \param flag            !!!: Used during debugging to print values from the selected instance
 */
template <
    unsigned MatrixW,
    unsigned MatrixH,
    unsigned SIMD,
    unsigned PE,
    typename Tw,
    typename Ti,
    typename To,
    typename TI, typename TO, typename TW, typename TA, typename R>
void Matrix_Vector_Activate_Batch(hls::stream<TI> &in,
                                  hls::stream<TO> &out,
                                  TW const &weights,
                                  TA const &activation,
                                  Ti const inp_zero_point,
                                  int const reps,
                                  R const &r, int flag = 0, Profiler *profiler = nullptr)
{
  // how many different rows each neuron will compute
  // alternatively: number of vertical matrix chunks
  constexpr unsigned int NF = MatrixH / PE;
  // how many synapse groups each row is split into
  // alternatively: number of horizontal matrix chunks
  constexpr unsigned int SF = MatrixW / SIMD;

  // input vector buffers
  TI inputBuf[SF];

  decltype(activation.init(0, 0)) accu[PE];
#pragma HLS ARRAY_PARTITION variable = accu complete dim = 0

  unsigned short nf = 0;
  unsigned short sf = 0;
  unsigned tile = 0; // invariant: tile = nf*SF + sf

  // everything merged into a common iteration space (one "big" loop instead
  // of smaller nested loops) to get the pipelinening the way we want
  constexpr unsigned TOTAL_FOLD = NF * SF;
  for (unsigned i = 0; i < reps * TOTAL_FOLD; i++)
  {
#pragma HLS PIPELINE style=flp II=1
    TI inElem;
    if (nf == 0)
    {
      // read input from stream
      inElem = in.read();
      // store in appropriate buffer for reuse
      inputBuf[sf] = inElem;
    }
    else
    {
      // reuse buffered input
      inElem = inputBuf[sf];
    }

    // Threshold Initialisation
    if (sf == 0)
    {
      for (unsigned pe = 0; pe < PE; pe++)
      {
#pragma HLS UNROLL
        accu[pe] = activation.init(nf, pe);
      }
    }
    // compute matrix-vector product for each processing element
    for (unsigned pe = 0; pe < PE; pe++)
    {
#pragma HLS UNROLL
      auto wgt = weights[pe][tile];
      accu[pe] = mac_<SIMD, Tw, Ti>(accu[pe], wgt, inElem, inp_zero_point, r, flag, profiler);
    }
    // keep track of which folded synapse/neuron we are processing
    ++tile;
    if (++sf == SF)
    {
      // produce output and clear accumulators
      TO outElem;
      for (unsigned pe = 0; pe < PE; pe++)
      {
#pragma HLS UNROLL
        outElem((pe + 1) * To::width - 1, pe * To::width) = activation.activate(nf, pe, accu[pe], flag, profiler);
      }
      out.write(outElem);
      // next folded neuron or image
      sf = 0;
      if (++nf == NF)
      {
        nf = 0;
        tile = 0;
      }
    }
  }
}

template <
    unsigned MatrixW,
    unsigned MatrixH,
    unsigned SIMD,
    unsigned PE,
    typename Tw,
    typename Ti,
    typename To,
    typename TI, typename TO, typename TA, typename R>
void Matrix_Vector_Activate_Batch_StreamWeights(hls::stream<TI> &in,
                                                hls::stream<TO> &out,
                                                hls::stream<ap_uint<PE * SIMD * Tw::width>> &weight,
                                                TA const &activation,
                                                int const reps,
                                                R const &r, int flag = 0, Profiler *profiler = nullptr)
{

  // how many different rows each neuron will compute
  // alternatively: number of vertical matrix chunks
  constexpr unsigned NF = MatrixH / PE;

  // how many synapse groups each row is split into
  // alternatively: number of horizontal matrix chunks
  constexpr unsigned SF = MatrixW / SIMD;

  // input vector buffers
  TI inputBuf[SF];

  // accumulators
  decltype(activation.init(0, 0)) accu[PE];
#pragma HLS ARRAY_PARTITION variable = accu complete dim = 0

  // ap_uint<PE*SIMD*Tw::width> w_pe_simd;
  // ap_uint<SIMD*Tw::width>  w_simd;

  unsigned nf = 0;
  unsigned sf = 0;
  unsigned tile = 0; // invariant: tile = nf*SF + sf

  // everything merged into a common iteration space (one "big" loop instead
  // of smaller nested loops) to get the pipelinening the way we want
  constexpr unsigned TOTAL_FOLD = NF * SF;
  for (unsigned i = 0; i < reps * TOTAL_FOLD; i++)
  {
#pragma HLS PIPELINE style=flp II=1
    TI inElem;

    if (nf == 0)
    {
      // read input from stream
      inElem = in.read();
      // store in appropriate buffer for reuse
      inputBuf[sf] = inElem;
    }
    else
    {
      // reuse buffered input
      inElem = inputBuf[sf];
    }

    // read from the parameter stream
    ap_uint<PE * SIMD * Tw::width> w_pe_simd = weight.read();

    // Threshold Initialisation
    if (sf == 0)
    {
      for (unsigned pe = 0; pe < PE; pe++)
      {
#pragma HLS UNROLL
        accu[pe] = activation.init(nf, pe);
      }
    }

    // compute matrix-vector product for each processing element
    for (unsigned pe = 0; pe < PE; pe++)
    {
#pragma HLS UNROLL
      ap_uint<SIMD * Tw::width> w_simd = w_pe_simd((pe + 1) * SIMD * Tw::width - 1, pe * SIMD * Tw::width);
      accu[pe] = mac_<SIMD, Tw, Ti>(accu[pe], w_simd, inElem, r, flag, profiler);
    }

    // keep track of which folded synapse/neuron we are processing
    ++tile;
    if (++sf == SF)
    {
      // produce output and clear accumulators
      TO outElem;
      for (unsigned pe = 0; pe < PE; pe++)
      {
#pragma HLS UNROLL
        outElem((pe + 1) * To::width - 1, pe * To::width) = activation.activate(nf, pe, accu[pe], flag, profiler);
      }
      out.write(outElem);
      // next folded neuron or image
      sf = 0;
      if (++nf == NF)
      {
        nf = 0;
        tile = 0;
      }
    }
  }
}

/** RPTU
 * \brief Matrix vector activate function for transposed convolution.
 *
 * The function performs the multiplication between a weight matrix and the input activation vector,
 * accumulating the results and then applying an activation function on the accumulated result.
 * This implementation is only valid for transposed convolution, hence it does not accumulate the results on a per kernel bases,
 * it accumulates the results on a channel wise bases, in other words it's like having a kernel of 1x1. Also, it assumes that the input,
 * is already expanded to the correct dimension so 1|2  should be inputted as 1|1|2|2 if the transpose kernel is 2x2 also it's currently only effective for Kernel of 2x2
 *                                                 3|4                        1|1|2|2
 *                                                                            3|3|4|4
 *                                                                            3|3|4|4
 *
 *
 * \tparam IFMChannels    Number of input feature map channels
 * \tparam OFMChannels    Number of output feature map channels
 * \tparam ConvKernelDim  Dimension of the convolution kernel (assumed square) currently only 2 is supported
 * \tparam IFMDim 		    Width and Height of the Input Feature Map (assumed square)
 * \tparam SIMD           Number of input columns computed in parallel
 * \tparam Tw             DataType of a single weight
 * \tparam Ti             DataType of a single input activation
 * \tparam To             DataType of a single output activation
 * \tparam TI             DataType of the input stream - safely deducible from the parameters
 * \tparam TO             DataType of the output stream - safely deducible from the parameters
 * \tparam TW             DataType of the weights matrix - safely deducible from the parameters
 * \tparam TA             DataType of the activation class (e.g. thresholds) - safely deducible from the parameters
 * \tparam R              Datatype for the resource used for FPGA implementation of the MAC  - safely deducible from the parameters
 *
 * \param in              Input stream
 * \param out             Output stream
 * \param weights         Weights matrix (currently supports BinaryWeights or FixedPointWeights)
 * \param activation      Activation class
 * \param inp_zero_point  Zero point of the input data
 * \param reps            Number of time the function has to be repeatedly executed (e.g. number of images)
 * \param r               Resource type for the hardware implementation of the MAC block
 */
template<
    unsigned IFMChannels,
    unsigned OFMChannels,
    unsigned ConvKernelDim,
    unsigned IFMDim,
    unsigned SIMD,
    unsigned PE,
    typename Tw,
    typename Ti,
    typename To,
    typename TI, typename TO, typename TW, typename TA, typename R>
void Matrix_Vector_Activate_Batch_Transpose(hls::stream<TI>& in,
                                            hls::stream<TO>& out,
                                            TW  const& weights,
                                            TA  const& activation,
                                            Ti const inp_zero_point,
                                            int const  reps,
                                            R const& r, int flag = 0, Profiler *profiler = nullptr) {
  static_assert(ConvKernelDim == 2, ""); //Code was only tested for 2x2 kernel, the logic the rests the tile needs modifications for other kernel sizes
  // how many different rows each neuron will compute
  // alternatively: number of vertical matrix chunks
  constexpr unsigned NF = OFMChannels / PE;

  // how many synapse groups each row is split into
  // alternatively: number of horizontal matrix chunks
  constexpr unsigned SF = IFMChannels / SIMD;
  // input vector buffers
  TI  inputBuf[SF];

  decltype(activation.init(0,0)) accu[PE] = {0};
#pragma HLS ARRAY_PARTITION variable=accu complete dim=0

  unsigned  nf = 0;
  unsigned  sf = 0;
  unsigned  tile = 0; // invariant: tile = nf*SF + sf

  // everything merged into a common iteration space (one "big" loop instead
  // of smaller nested loops) to get the pipelining the way we want
  constexpr unsigned TOTAL_FOLD = NF * SF;
  constexpr unsigned ONE_RAW = ConvKernelDim * OFMChannels * IFMChannels; // Number of iterations required to calculate just one row of the kernel, used to control the tile index
  constexpr unsigned TWO_RAW = ConvKernelDim * ConvKernelDim * OFMChannels * IFMChannels; // Number of iterations required to calculate just two row of the kernel, used to control the tile index
  constexpr unsigned ROW_ITERATIONS = OFMChannels * IFMChannels * IFMDim; // Number of iterations required to calculate whole row of the input, used to toggle Kernel_row
  unsigned TILE_BASE = 0; // For the first row of kernel it should be 0 otherwise it should be MAX_KER
  bool Kernel_row = 0; // 0:first row, 1:second row
  for (unsigned i = 0; i < reps * TOTAL_FOLD; i++) {
#pragma HLS pipeline style=flp II=1
    TI  inElem;
    if (nf == 0) {
      // read input from stream
      inElem = in.read();
      // store in appropriate buffer for reuse
      inputBuf[sf] = inElem;
    }
    else {
      // reuse buffered input
      inElem = inputBuf[sf];
    }

    // compute matrix-vector product for each processing element
    for (unsigned pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
      auto wgt = weights[pe][tile];
      accu[pe] = mac_<SIMD, Tw, Ti>(accu[pe], wgt, inElem, inp_zero_point, r, flag, profiler);
    }

    // keep track of which folded synapse/neuron we are processing
    ++tile;
    if (++sf == SF) {
      // produce output and clear accumulators
      TO outElem;
      for (unsigned pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
        outElem((pe + 1) * To::width - 1, pe * To::width) = activation.activate(nf, pe, accu[pe], flag, profiler);
        accu[pe] = 0;
      }
      out.write(outElem);
      // next folded neuron or image
      sf = 0;
      if (++nf == NF) {
        nf = 0;
      }
    }
    if ((i + 1) % ROW_ITERATIONS == 0) { //reset every whole row
      Kernel_row = !Kernel_row;
      TILE_BASE = Kernel_row * ONE_RAW;
    }
    if (tile == ONE_RAW || tile == TWO_RAW) {
      tile = TILE_BASE;
    }
  }
}
#endif
