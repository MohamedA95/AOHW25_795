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
 *****************************************************************************/
 
/*****************************************************************************
 *
 *  Authors: Giulio Gambardella <giuliog@xilinx.com>
 *           Thomas B. Preusser <thomas.preusser@utexas.edu>
 *             Marie-Curie Fellow, Xilinx Ireland, Grant Agreement No. 751339
 *           Christoph Doehring <cdoehrin@xilinx.com>
 *
 *  \file fclayer.h
 *
 *  Library of templated HLS functions for BNN deployment. 
 *  This file lists a set of convenience funtions used to implement fully 
 *  connected layers
 *
 *****************************************************************************/
 
#ifndef FCLAYER_H
#define FCLAYER_H

#include <ap_int.h>
#include <hls_stream.h>
#include "streamtools.h"
#include "mvau.hpp"

template
<
  unsigned int MatrixW,
  unsigned int MatrixH,
  unsigned int SIMD,
  unsigned int PE,
  typename Tw,
  typename TSrcI = Identity,
  typename TDstI = Identity,
  typename TWeightI = Identity,	
  // safely deducible (stream width must be int though!)  
  int InStreamW, int OutStreamW, typename TW, typename TA, typename R
>
void Linear_Batch(hls::stream<ap_uint<InStreamW>>  &in,
                  hls::stream<ap_uint<OutStreamW>> &out,
                  TW const & weights,
                  TA const & activation,
                  unsigned const   reps,
                  R const &r, int flag = 0, Profiler *profiler = nullptr)
{
	
#pragma HLS INLINE

  Matrix_Vector_Activate_Batch<MatrixW, MatrixH, SIMD, PE, 1, Tw, TSrcI, TDstI, TWeightI>
    (static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(in),
     static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (out),
     weights, activation, reps, r, flag, profiler);
}
template<
  unsigned int MatrixW,
  unsigned int MatrixH,  
  unsigned int SIMD,
  unsigned int PE,  
  typename Tw,
  typename TSrcI = Identity,
  typename TDstI = Identity,
  typename TWeightI = Identity,	
  // safely deducible (stream width must be int though!)  
  int InStreamW, int OutStreamW, typename TA, typename R
>
void Linear_Batch_StreamWeights(hls::stream<ap_uint<InStreamW>>  & in,
                                hls::stream<ap_uint<OutStreamW>> & out,
                                hls::stream<ap_uint<PE*SIMD*Tw::width>> & weights,
                                TA const & activation,
                                unsigned const reps,
                                R const &r, int flag = 0, Profiler *profiler = nullptr)
{
	
#pragma HLS INLINE
  unsigned const  InpPerImage = (MatrixW * TSrcI::width) / InStreamW;
  unsigned const  OutPerImage = MatrixH / PE;

  Matrix_Vector_Activate_Batch_StreamWeights<MatrixW, MatrixH, SIMD, PE, Tw, TSrcI, TDstI, TWeightI>
    (static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(in),
     static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (out),
     weights, activation, reps, r, flag, profiler);
}

template
<
  unsigned int MatrixW,
  unsigned int MatrixH,  
  unsigned int SIMD,
  unsigned int PE,  
  typename Tw,
  typename TSrcI = Identity,
  typename TDstI = Identity,
  typename TWeightI = Identity,	
  // safely deducible (stream width must be int though!)  
  int InStreamW, int OutStreamW, int WeightMemW, typename TA, typename R
>
void Linear_Batch_OffChipWeights(hls::stream<ap_uint<InStreamW>>  &in_stream,
                                hls::stream<ap_uint<OutStreamW>> &out_stream,
                                ap_uint<WeightMemW> *weight_mem,
                                TA const & activation,
                                unsigned const reps,
                                R const &r, int flag = 0, Profiler *profiler = nullptr)
{
	
#pragma HLS INLINE

  unsigned const ReadsPerWeightMem = (MatrixW * MatrixH * Tw::width + WeightMemW - 1) / WeightMemW;

  hls::stream<ap_uint<WeightMemW>> m2s_out("m2s_out");
  Mem2Stream_Batch<ReadsPerWeightMem>(weight_mem, m2s_out, reps);
  WidthAdjustedInputStream <WeightMemW, SIMD*PE*Tw::width, ReadsPerWeightMem>  weight_stream (m2s_out, reps);	

	Linear_Batch_StreamWeights
  <
  MatrixW,
  MatrixH,
  SIMD,
  PE,
  Tw,
  TSrcI,
  TDstI
  >
  (in_stream,
  out_stream,
  weight_stream,
  activation,
  reps,
  r, flag, profiler);
}

#endif
