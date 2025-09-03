#include "bnn-library.h"
#include "dnn_config.hpp"

void top_level_0(ap_uint<INPUT_MEM_WIDTH> *conv_layer_0_input_mem,
	ap_uint<OUTPUT_MEM_WIDTH> *conv_layer_10_output_mem)
{

#pragma HLS INTERFACE s_axilite port=return bundle=control

	unsigned const conv_layer_0_reads_per_input_mem = CONV_LAYER_0_IFM_BITS_ALIGNED * CONV_LAYER_0_IFM_CH * CONV_LAYER_0_IFM_DIM * CONV_LAYER_0_IFM_DIM  / INPUT_MEM_WIDTH;
#pragma HLS INTERFACE m_axi offset=slave port=conv_layer_0_input_mem bundle=conv_layer_0_input_mem depth=conv_layer_0_reads_per_input_mem * Reps
#pragma HLS INTERFACE s_axilite port=conv_layer_0_input_mem bundle=control
	unsigned const conv_layer_10_writes_per_output_mem = CONV_LAYER_10_OFM_BITS_ALIGNED * CONV_LAYER_10_OFM_CH * CONV_LAYER_10_OFM_DIM * CONV_LAYER_10_OFM_DIM  / OUTPUT_MEM_WIDTH;
	unsigned const conv_layer_10_writes_per_output_stream = CONV_LAYER_10_OUTPUT_ACCESSES;
#pragma HLS INTERFACE m_axi offset=slave port=conv_layer_10_output_mem bundle=conv_layer_10_output_mem depth=conv_layer_10_writes_per_output_mem * Reps
#pragma HLS INTERFACE s_axilite port=conv_layer_10_output_mem bundle=control


#pragma HLS DATAFLOW
auto const default_compute_resource = ap_resource_dflt();
#include "dnn_params_0.hpp"

	hls::stream<ap_uint<INPUT_MEM_WIDTH>> conv_layer_0_m2s;
#pragma HLS STREAM variable=conv_layer_0_m2s depth=128
	hls::stream<ap_uint<CONV_LAYER_0_INPUT_STREAM_WIDTH>> conv_layer_0_input_stream;
#pragma HLS STREAM variable=conv_layer_0_input_stream depth=128
	hls::stream<ap_uint<CONV_LAYER_1_INPUT_STREAM_WIDTH>> conv_layer_0_conv_layer_1_stream;
#pragma HLS STREAM variable=conv_layer_0_conv_layer_1_stream depth=128
	hls::stream<ap_uint<CONV_LAYER_1_OUTPUT_STREAM_WIDTH>> conv_layer_1_sdwc_0_stream;
#pragma HLS STREAM variable=conv_layer_1_sdwc_0_stream depth=128
	hls::stream<ap_uint<SPLIT_0_INPUT_STREAM_WIDTH>> sdwc_0_split_0_stream;
#pragma HLS STREAM variable=sdwc_0_split_0_stream depth=128
	hls::stream<ap_uint<MAXPOOL_0_INPUT_STREAM_WIDTH>> split_0_maxpool_0_stream;
#pragma HLS STREAM variable=split_0_maxpool_0_stream depth=128
	hls::stream<ap_uint<MAXPOOL_0_OUTPUT_STREAM_WIDTH>> maxpool_0_sdwc_1_stream;
#pragma HLS STREAM variable=maxpool_0_sdwc_1_stream depth=128
	hls::stream<ap_uint<CONV_LAYER_2_INPUT_STREAM_WIDTH>> sdwc_1_conv_layer_2_stream;
#pragma HLS STREAM variable=sdwc_1_conv_layer_2_stream depth=128
	hls::stream<ap_uint<CONV_LAYER_2_OUTPUT_STREAM_WIDTH>> conv_layer_2_sdwc_2_stream;
#pragma HLS STREAM variable=conv_layer_2_sdwc_2_stream depth=128
	hls::stream<ap_uint<CONV_LAYER_3_INPUT_STREAM_WIDTH>> sdwc_2_conv_layer_3_stream;
#pragma HLS STREAM variable=sdwc_2_conv_layer_3_stream depth=128
	hls::stream<ap_uint<CONV_LAYER_3_OUTPUT_STREAM_WIDTH>> conv_layer_3_sdwc_3_stream;
#pragma HLS STREAM variable=conv_layer_3_sdwc_3_stream depth=128
	hls::stream<ap_uint<SPLIT_1_INPUT_STREAM_WIDTH>> sdwc_3_split_1_stream;
#pragma HLS STREAM variable=sdwc_3_split_1_stream depth=128
	hls::stream<ap_uint<MAXPOOL_1_INPUT_STREAM_WIDTH>> split_1_maxpool_1_stream;
#pragma HLS STREAM variable=split_1_maxpool_1_stream depth=128
	hls::stream<ap_uint<MAXPOOL_1_OUTPUT_STREAM_WIDTH>> maxpool_1_sdwc_4_stream;
#pragma HLS STREAM variable=maxpool_1_sdwc_4_stream depth=128
	hls::stream<ap_uint<CONV_LAYER_4_INPUT_STREAM_WIDTH>> sdwc_4_conv_layer_4_stream;
#pragma HLS STREAM variable=sdwc_4_conv_layer_4_stream depth=128
	hls::stream<ap_uint<CONV_LAYER_4_OUTPUT_STREAM_WIDTH>> conv_layer_4_sdwc_5_stream;
#pragma HLS STREAM variable=conv_layer_4_sdwc_5_stream depth=128
	hls::stream<ap_uint<CONV_LAYER_5_INPUT_STREAM_WIDTH>> sdwc_5_conv_layer_5_stream;
#pragma HLS STREAM variable=sdwc_5_conv_layer_5_stream depth=128
	hls::stream<ap_uint<CONV_LAYER_5_OUTPUT_STREAM_WIDTH>> conv_layer_5_sdwc_6_stream;
#pragma HLS STREAM variable=conv_layer_5_sdwc_6_stream depth=128
	hls::stream<ap_uint<CONVTRANSPOSE_0_INPUT_STREAM_WIDTH>> sdwc_6_convtranspose_0_stream;
#pragma HLS STREAM variable=sdwc_6_convtranspose_0_stream depth=128
	hls::stream<ap_uint<CONVTRANSPOSE_0_OUTPUT_STREAM_WIDTH>> convtranspose_0_sdwc_7_stream;
#pragma HLS STREAM variable=convtranspose_0_sdwc_7_stream depth=128
	hls::stream<ap_uint<CONCAT_0_INPUT_STREAM_WIDTH>> split_1_concat_0_stream;
#pragma HLS STREAM variable=split_1_concat_0_stream depth=118092
	hls::stream<ap_uint<CONCAT_0_INPUT_STREAM_WIDTH>> sdwc_7_concat_0_stream;
#pragma HLS STREAM variable=sdwc_7_concat_0_stream depth=128
	hls::stream<ap_uint<CONCAT_0_OUTPUT_STREAM_WIDTH>> concat_0_sdwc_8_stream;
#pragma HLS STREAM variable=concat_0_sdwc_8_stream depth=128
	hls::stream<ap_uint<CONV_LAYER_6_INPUT_STREAM_WIDTH>> sdwc_8_conv_layer_6_stream;
#pragma HLS STREAM variable=sdwc_8_conv_layer_6_stream depth=128
	hls::stream<ap_uint<CONV_LAYER_7_INPUT_STREAM_WIDTH>> conv_layer_6_conv_layer_7_stream;
#pragma HLS STREAM variable=conv_layer_6_conv_layer_7_stream depth=128
	hls::stream<ap_uint<CONV_LAYER_7_OUTPUT_STREAM_WIDTH>> conv_layer_7_sdwc_9_stream;
#pragma HLS STREAM variable=conv_layer_7_sdwc_9_stream depth=128
	hls::stream<ap_uint<CONVTRANSPOSE_1_INPUT_STREAM_WIDTH>> sdwc_9_convtranspose_1_stream;
#pragma HLS STREAM variable=sdwc_9_convtranspose_1_stream depth=128
	hls::stream<ap_uint<CONVTRANSPOSE_1_OUTPUT_STREAM_WIDTH>> convtranspose_1_sdwc_10_stream;
#pragma HLS STREAM variable=convtranspose_1_sdwc_10_stream depth=128
	hls::stream<ap_uint<CONCAT_1_INPUT_STREAM_WIDTH>> split_0_concat_1_stream;
#pragma HLS STREAM variable=split_0_concat_1_stream depth=236057
	hls::stream<ap_uint<CONCAT_1_INPUT_STREAM_WIDTH>> sdwc_10_concat_1_stream;
#pragma HLS STREAM variable=sdwc_10_concat_1_stream depth=128
	hls::stream<ap_uint<CONCAT_1_OUTPUT_STREAM_WIDTH>> concat_1_sdwc_11_stream;
#pragma HLS STREAM variable=concat_1_sdwc_11_stream depth=128
	hls::stream<ap_uint<CONV_LAYER_8_INPUT_STREAM_WIDTH>> sdwc_11_conv_layer_8_stream;
#pragma HLS STREAM variable=sdwc_11_conv_layer_8_stream depth=128
	hls::stream<ap_uint<CONV_LAYER_9_INPUT_STREAM_WIDTH>> conv_layer_8_conv_layer_9_stream;
#pragma HLS STREAM variable=conv_layer_8_conv_layer_9_stream depth=128
	hls::stream<ap_uint<CONV_LAYER_9_OUTPUT_STREAM_WIDTH>> conv_layer_9_sdwc_12_stream;
#pragma HLS STREAM variable=conv_layer_9_sdwc_12_stream depth=128
	hls::stream<ap_uint<CONV_LAYER_10_INPUT_STREAM_WIDTH>> sdwc_12_conv_layer_10_stream;
#pragma HLS STREAM variable=sdwc_12_conv_layer_10_stream depth=128
	hls::stream<ap_uint<CONV_LAYER_10_OUTPUT_STREAM_WIDTH>> conv_layer_10_output_stream;
#pragma HLS STREAM variable=conv_layer_10_output_stream depth=128
	hls::stream<ap_uint<OUTPUT_MEM_WIDTH>> conv_layer_10_s2m;
#pragma HLS STREAM variable=conv_layer_10_s2m depth=128


	Mem2Stream_Batch<conv_layer_0_reads_per_input_mem>(conv_layer_0_input_mem, conv_layer_0_m2s, Reps);
	StreamingDataWidthConverter_Batch<INPUT_MEM_WIDTH, CONV_LAYER_0_INPUT_STREAM_WIDTH, conv_layer_0_reads_per_input_mem>(conv_layer_0_m2s, conv_layer_0_input_stream, Reps);


	std::cout << "Conv_layer_0" << std::endl;

	Conv_Padding_Batch
	<
	CONV_LAYER_0_K,
	CONV_LAYER_0_IFM_CH,
	CONV_LAYER_0_IFM_DIM,
	CONV_LAYER_0_OFM_CH,
	CONV_LAYER_0_OFM_DIM,
	CONV_LAYER_0_STRIDE,
	CONV_LAYER_0_PADDING,
	CONV_LAYER_0_SIMD,
	CONV_LAYER_0_PE,
	conv_layer_0_weight_dtype,
	conv_layer_0_input_dtype,
	conv_layer_0_output_dtype
	>
	(conv_layer_0_input_stream,
	conv_layer_0_conv_layer_1_stream,
	conv_layer_0_weights,
	conv_layer_0_activations,
	CONV_LAYER_0_INPUT_ZERO_POINT,
	Reps,
	default_compute_resource,
	0,
	&Conv_layer_0);

	std::cout << "Conv_layer_1" << std::endl;

	Conv_Padding_Batch
	<
	CONV_LAYER_1_K,
	CONV_LAYER_1_IFM_CH,
	CONV_LAYER_1_IFM_DIM,
	CONV_LAYER_1_OFM_CH,
	CONV_LAYER_1_OFM_DIM,
	CONV_LAYER_1_STRIDE,
	CONV_LAYER_1_PADDING,
	CONV_LAYER_1_SIMD,
	CONV_LAYER_1_PE,
	conv_layer_1_weight_dtype,
	conv_layer_1_input_dtype,
	conv_layer_1_output_dtype
	>
	(conv_layer_0_conv_layer_1_stream,
	conv_layer_1_sdwc_0_stream,
	conv_layer_1_weights,
	conv_layer_1_activations,
	CONV_LAYER_1_INPUT_ZERO_POINT,
	Reps,
	default_compute_resource,
	0,
	&Conv_layer_1);

	std::cout << "sdwc_0" << std::endl;

	StreamingDataWidthConverter_Batch<CONV_LAYER_1_OUTPUT_STREAM_WIDTH, MAXPOOL_0_INPUT_STREAM_WIDTH, CONV_LAYER_1_OUTPUT_ACCESSES>(conv_layer_1_sdwc_0_stream,  sdwc_0_split_0_stream, Reps);

	std::cout << "Split_0" << std::endl;

	DuplicateStreams_Batch
	<
	SPLIT_0_INPUT_ACCESSESS,
	split_0_input_dtype
	>
	(sdwc_0_split_0_stream,
	split_0_maxpool_0_stream,
	split_0_concat_1_stream,
	Reps,
	&Split_0);

	std::cout << "MaxPool_0" << std::endl;

	Streaming_Maxpool_batch2d
	<
	MAXPOOL_0_IFM_CH,
	MAXPOOL_0_IFM_DIM,
	MAXPOOL_0_PE,
	MAXPOOL_0_K,
	MAXPOOL_0_STRIDE,
	Slice<maxpool_0_input_dtype>,
	maxpool_0_input_dtype
	>
	(split_0_maxpool_0_stream,
	maxpool_0_sdwc_1_stream,
	Reps);

	std::cout << "sdwc_1" << std::endl;

	StreamingDataWidthConverter_Batch<MAXPOOL_0_OUTPUT_STREAM_WIDTH, CONV_LAYER_2_INPUT_STREAM_WIDTH, MAXPOOL_0_OUTPUT_ACCESSES>(maxpool_0_sdwc_1_stream,  sdwc_1_conv_layer_2_stream, Reps);

	std::cout << "Conv_layer_2" << std::endl;

	Conv_Padding_Batch
	<
	CONV_LAYER_2_K,
	CONV_LAYER_2_IFM_CH,
	CONV_LAYER_2_IFM_DIM,
	CONV_LAYER_2_OFM_CH,
	CONV_LAYER_2_OFM_DIM,
	CONV_LAYER_2_STRIDE,
	CONV_LAYER_2_PADDING,
	CONV_LAYER_2_SIMD,
	CONV_LAYER_2_PE,
	conv_layer_2_weight_dtype,
	conv_layer_2_input_dtype,
	conv_layer_2_output_dtype
	>
	(sdwc_1_conv_layer_2_stream,
	conv_layer_2_sdwc_2_stream,
	conv_layer_2_weights,
	conv_layer_2_activations,
	CONV_LAYER_2_INPUT_ZERO_POINT,
	Reps,
	default_compute_resource,
	0,
	&Conv_layer_2);

	std::cout << "sdwc_2" << std::endl;

	StreamingDataWidthConverter_Batch<CONV_LAYER_2_OUTPUT_STREAM_WIDTH, CONV_LAYER_3_INPUT_STREAM_WIDTH, CONV_LAYER_2_OUTPUT_ACCESSES>(conv_layer_2_sdwc_2_stream,  sdwc_2_conv_layer_3_stream, Reps);

	std::cout << "Conv_layer_3" << std::endl;

	Conv_Padding_Batch
	<
	CONV_LAYER_3_K,
	CONV_LAYER_3_IFM_CH,
	CONV_LAYER_3_IFM_DIM,
	CONV_LAYER_3_OFM_CH,
	CONV_LAYER_3_OFM_DIM,
	CONV_LAYER_3_STRIDE,
	CONV_LAYER_3_PADDING,
	CONV_LAYER_3_SIMD,
	CONV_LAYER_3_PE,
	conv_layer_3_weight_dtype,
	conv_layer_3_input_dtype,
	conv_layer_3_output_dtype
	>
	(sdwc_2_conv_layer_3_stream,
	conv_layer_3_sdwc_3_stream,
	conv_layer_3_weights,
	conv_layer_3_activations,
	CONV_LAYER_3_INPUT_ZERO_POINT,
	Reps,
	default_compute_resource,
	0,
	&Conv_layer_3);

	std::cout << "sdwc_3" << std::endl;

	StreamingDataWidthConverter_Batch<CONV_LAYER_3_OUTPUT_STREAM_WIDTH, MAXPOOL_1_INPUT_STREAM_WIDTH, CONV_LAYER_3_OUTPUT_ACCESSES>(conv_layer_3_sdwc_3_stream,  sdwc_3_split_1_stream, Reps);

	std::cout << "Split_1" << std::endl;

	DuplicateStreams_Batch
	<
	SPLIT_1_INPUT_ACCESSESS,
	split_1_input_dtype
	>
	(sdwc_3_split_1_stream,
	split_1_maxpool_1_stream,
	split_1_concat_0_stream,
	Reps,
	&Split_1);

	std::cout << "MaxPool_1" << std::endl;

	Streaming_Maxpool_batch2d
	<
	MAXPOOL_1_IFM_CH,
	MAXPOOL_1_IFM_DIM,
	MAXPOOL_1_PE,
	MAXPOOL_1_K,
	MAXPOOL_1_STRIDE,
	Slice<maxpool_1_input_dtype>,
	maxpool_1_input_dtype
	>
	(split_1_maxpool_1_stream,
	maxpool_1_sdwc_4_stream,
	Reps);

	std::cout << "sdwc_4" << std::endl;

	StreamingDataWidthConverter_Batch<MAXPOOL_1_OUTPUT_STREAM_WIDTH, CONV_LAYER_4_INPUT_STREAM_WIDTH, MAXPOOL_1_OUTPUT_ACCESSES>(maxpool_1_sdwc_4_stream,  sdwc_4_conv_layer_4_stream, Reps);

	std::cout << "Conv_layer_4" << std::endl;

	Conv_Padding_Batch
	<
	CONV_LAYER_4_K,
	CONV_LAYER_4_IFM_CH,
	CONV_LAYER_4_IFM_DIM,
	CONV_LAYER_4_OFM_CH,
	CONV_LAYER_4_OFM_DIM,
	CONV_LAYER_4_STRIDE,
	CONV_LAYER_4_PADDING,
	CONV_LAYER_4_SIMD,
	CONV_LAYER_4_PE,
	conv_layer_4_weight_dtype,
	conv_layer_4_input_dtype,
	conv_layer_4_output_dtype
	>
	(sdwc_4_conv_layer_4_stream,
	conv_layer_4_sdwc_5_stream,
	conv_layer_4_weights,
	conv_layer_4_activations,
	CONV_LAYER_4_INPUT_ZERO_POINT,
	Reps,
	default_compute_resource,
	0,
	&Conv_layer_4);

	std::cout << "sdwc_5" << std::endl;

	StreamingDataWidthConverter_Batch<CONV_LAYER_4_OUTPUT_STREAM_WIDTH, CONV_LAYER_5_INPUT_STREAM_WIDTH, CONV_LAYER_4_OUTPUT_ACCESSES>(conv_layer_4_sdwc_5_stream,  sdwc_5_conv_layer_5_stream, Reps);

	std::cout << "Conv_layer_5" << std::endl;

	Conv_Padding_Batch
	<
	CONV_LAYER_5_K,
	CONV_LAYER_5_IFM_CH,
	CONV_LAYER_5_IFM_DIM,
	CONV_LAYER_5_OFM_CH,
	CONV_LAYER_5_OFM_DIM,
	CONV_LAYER_5_STRIDE,
	CONV_LAYER_5_PADDING,
	CONV_LAYER_5_SIMD,
	CONV_LAYER_5_PE,
	conv_layer_5_weight_dtype,
	conv_layer_5_input_dtype,
	conv_layer_5_output_dtype
	>
	(sdwc_5_conv_layer_5_stream,
	conv_layer_5_sdwc_6_stream,
	conv_layer_5_weights,
	conv_layer_5_activations,
	CONV_LAYER_5_INPUT_ZERO_POINT,
	Reps,
	default_compute_resource,
	0,
	&Conv_layer_5);

	std::cout << "sdwc_6" << std::endl;

	StreamingDataWidthConverter_Batch<CONV_LAYER_5_OUTPUT_STREAM_WIDTH, CONVTRANSPOSE_0_INPUT_STREAM_WIDTH, CONV_LAYER_5_OUTPUT_ACCESSES>(conv_layer_5_sdwc_6_stream,  sdwc_6_convtranspose_0_stream, Reps);

	std::cout << "ConvTranspose_0" << std::endl;

	TransposeConv
	<
	CONVTRANSPOSE_0_K,
	CONVTRANSPOSE_0_IFM_CH,
	CONVTRANSPOSE_0_IFM_DIM,
	CONVTRANSPOSE_0_OFM_CH,
	CONVTRANSPOSE_0_SIMD,
	CONVTRANSPOSE_0_PE,
	convtranspose_0_weight_dtype,
	convtranspose_0_input_dtype,
	convtranspose_0_output_dtype
	>
	(sdwc_6_convtranspose_0_stream,
	convtranspose_0_sdwc_7_stream,
	convtranspose_0_weights,
	convtranspose_0_activations,
	CONVTRANSPOSE_0_INPUT_ZERO_POINT,
	Reps,
	default_compute_resource,
	0,
	&ConvTranspose_0);

	std::cout << "sdwc_7" << std::endl;

	StreamingDataWidthConverter_Batch<CONVTRANSPOSE_0_OUTPUT_STREAM_WIDTH, CONCAT_0_INPUT_STREAM_WIDTH, CONVTRANSPOSE_0_OUTPUT_ACCESSES>(convtranspose_0_sdwc_7_stream,  sdwc_7_concat_0_stream, Reps);

	std::cout << "Concat_0" << std::endl;

	ConcatStreams_Batch
	<
	CONCAT_0_IFM_CH/CONCAT_0_SIMD,
	CONCAT_0_IFM_DIM,
	concat_0_scale_dtype,
	concat_0_shift_dtype
	>
	(
	split_1_concat_0_stream,
	sdwc_7_concat_0_stream,
	concat_0_sdwc_8_stream,
	CONCAT_0_0_M0,
	CONCAT_0_0_INPUT_ZERO_POINT,
	CONCAT_0_1_M0,
	CONCAT_0_1_INPUT_ZERO_POINT,
	Reps);

	std::cout << "sdwc_8" << std::endl;

	StreamingDataWidthConverter_Batch<CONCAT_0_OUTPUT_STREAM_WIDTH, CONV_LAYER_6_INPUT_STREAM_WIDTH, CONCAT_0_OUTPUT_ACCESSES>(concat_0_sdwc_8_stream,  sdwc_8_conv_layer_6_stream, Reps);

	std::cout << "Conv_layer_6" << std::endl;

	Conv_Padding_Batch
	<
	CONV_LAYER_6_K,
	CONV_LAYER_6_IFM_CH,
	CONV_LAYER_6_IFM_DIM,
	CONV_LAYER_6_OFM_CH,
	CONV_LAYER_6_OFM_DIM,
	CONV_LAYER_6_STRIDE,
	CONV_LAYER_6_PADDING,
	CONV_LAYER_6_SIMD,
	CONV_LAYER_6_PE,
	conv_layer_6_weight_dtype,
	conv_layer_6_input_dtype,
	conv_layer_6_output_dtype
	>
	(sdwc_8_conv_layer_6_stream,
	conv_layer_6_conv_layer_7_stream,
	conv_layer_6_weights,
	conv_layer_6_activations,
	CONV_LAYER_6_INPUT_ZERO_POINT,
	Reps,
	default_compute_resource,
	0,
	&Conv_layer_6);

	std::cout << "Conv_layer_7" << std::endl;

	Conv_Padding_Batch
	<
	CONV_LAYER_7_K,
	CONV_LAYER_7_IFM_CH,
	CONV_LAYER_7_IFM_DIM,
	CONV_LAYER_7_OFM_CH,
	CONV_LAYER_7_OFM_DIM,
	CONV_LAYER_7_STRIDE,
	CONV_LAYER_7_PADDING,
	CONV_LAYER_7_SIMD,
	CONV_LAYER_7_PE,
	conv_layer_7_weight_dtype,
	conv_layer_7_input_dtype,
	conv_layer_7_output_dtype
	>
	(conv_layer_6_conv_layer_7_stream,
	conv_layer_7_sdwc_9_stream,
	conv_layer_7_weights,
	conv_layer_7_activations,
	CONV_LAYER_7_INPUT_ZERO_POINT,
	Reps,
	default_compute_resource,
	0,
	&Conv_layer_7);

	std::cout << "sdwc_9" << std::endl;

	StreamingDataWidthConverter_Batch<CONV_LAYER_7_OUTPUT_STREAM_WIDTH, CONVTRANSPOSE_1_INPUT_STREAM_WIDTH, CONV_LAYER_7_OUTPUT_ACCESSES>(conv_layer_7_sdwc_9_stream,  sdwc_9_convtranspose_1_stream, Reps);

	std::cout << "ConvTranspose_1" << std::endl;

	TransposeConv
	<
	CONVTRANSPOSE_1_K,
	CONVTRANSPOSE_1_IFM_CH,
	CONVTRANSPOSE_1_IFM_DIM,
	CONVTRANSPOSE_1_OFM_CH,
	CONVTRANSPOSE_1_SIMD,
	CONVTRANSPOSE_1_PE,
	convtranspose_1_weight_dtype,
	convtranspose_1_input_dtype,
	convtranspose_1_output_dtype
	>
	(sdwc_9_convtranspose_1_stream,
	convtranspose_1_sdwc_10_stream,
	convtranspose_1_weights,
	convtranspose_1_activations,
	CONVTRANSPOSE_1_INPUT_ZERO_POINT,
	Reps,
	default_compute_resource,
	0,
	&ConvTranspose_1);

	std::cout << "sdwc_10" << std::endl;

	StreamingDataWidthConverter_Batch<CONVTRANSPOSE_1_OUTPUT_STREAM_WIDTH, CONCAT_1_INPUT_STREAM_WIDTH, CONVTRANSPOSE_1_OUTPUT_ACCESSES>(convtranspose_1_sdwc_10_stream,  sdwc_10_concat_1_stream, Reps);

	std::cout << "Concat_1" << std::endl;

	ConcatStreams_Batch
	<
	CONCAT_1_IFM_CH/CONCAT_1_SIMD,
	CONCAT_1_IFM_DIM,
	concat_1_scale_dtype,
	concat_1_shift_dtype
	>
	(
	split_0_concat_1_stream,
	sdwc_10_concat_1_stream,
	concat_1_sdwc_11_stream,
	CONCAT_1_0_M0,
	CONCAT_1_0_INPUT_ZERO_POINT,
	CONCAT_1_1_M0,
	CONCAT_1_1_INPUT_ZERO_POINT,
	Reps);

	std::cout << "sdwc_11" << std::endl;

	StreamingDataWidthConverter_Batch<CONCAT_1_OUTPUT_STREAM_WIDTH, CONV_LAYER_8_INPUT_STREAM_WIDTH, CONCAT_1_OUTPUT_ACCESSES>(concat_1_sdwc_11_stream,  sdwc_11_conv_layer_8_stream, Reps);

	std::cout << "Conv_layer_8" << std::endl;

	Conv_Padding_Batch
	<
	CONV_LAYER_8_K,
	CONV_LAYER_8_IFM_CH,
	CONV_LAYER_8_IFM_DIM,
	CONV_LAYER_8_OFM_CH,
	CONV_LAYER_8_OFM_DIM,
	CONV_LAYER_8_STRIDE,
	CONV_LAYER_8_PADDING,
	CONV_LAYER_8_SIMD,
	CONV_LAYER_8_PE,
	conv_layer_8_weight_dtype,
	conv_layer_8_input_dtype,
	conv_layer_8_output_dtype
	>
	(sdwc_11_conv_layer_8_stream,
	conv_layer_8_conv_layer_9_stream,
	conv_layer_8_weights,
	conv_layer_8_activations,
	CONV_LAYER_8_INPUT_ZERO_POINT,
	Reps,
	default_compute_resource,
	0,
	&Conv_layer_8);

	std::cout << "Conv_layer_9" << std::endl;

	Conv_Padding_Batch
	<
	CONV_LAYER_9_K,
	CONV_LAYER_9_IFM_CH,
	CONV_LAYER_9_IFM_DIM,
	CONV_LAYER_9_OFM_CH,
	CONV_LAYER_9_OFM_DIM,
	CONV_LAYER_9_STRIDE,
	CONV_LAYER_9_PADDING,
	CONV_LAYER_9_SIMD,
	CONV_LAYER_9_PE,
	conv_layer_9_weight_dtype,
	conv_layer_9_input_dtype,
	conv_layer_9_output_dtype
	>
	(conv_layer_8_conv_layer_9_stream,
	conv_layer_9_sdwc_12_stream,
	conv_layer_9_weights,
	conv_layer_9_activations,
	CONV_LAYER_9_INPUT_ZERO_POINT,
	Reps,
	default_compute_resource,
	0,
	&Conv_layer_9);

	std::cout << "sdwc_12" << std::endl;

	StreamingDataWidthConverter_Batch<CONV_LAYER_9_OUTPUT_STREAM_WIDTH, CONV_LAYER_10_INPUT_STREAM_WIDTH, CONV_LAYER_9_OUTPUT_ACCESSES>(conv_layer_9_sdwc_12_stream,  sdwc_12_conv_layer_10_stream, Reps);

	std::cout << "Conv_layer_10" << std::endl;

	Conv_Pw_Batch
	<
	CONV_LAYER_10_K,
	CONV_LAYER_10_IFM_CH,
	CONV_LAYER_10_IFM_DIM,
	CONV_LAYER_10_OFM_CH,
	CONV_LAYER_10_OFM_DIM,
	CONV_LAYER_10_STRIDE,
	CONV_LAYER_10_PADDING,
	CONV_LAYER_10_SIMD,
	CONV_LAYER_10_PE,
	conv_layer_10_weight_dtype,
	conv_layer_10_input_dtype,
	conv_layer_10_output_dtype
	>
	(sdwc_12_conv_layer_10_stream,
	conv_layer_10_output_stream,
	conv_layer_10_weights,
	conv_layer_10_activations,
	CONV_LAYER_10_INPUT_ZERO_POINT,
	Reps,
	default_compute_resource,
	0,
	&Conv_layer_10);


	StreamingDataWidthConverter_Batch<CONV_LAYER_10_OUTPUT_STREAM_WIDTH, OUTPUT_MEM_WIDTH, conv_layer_10_writes_per_output_stream>(conv_layer_10_output_stream, conv_layer_10_s2m, Reps);
	Stream2Mem_Batch<conv_layer_10_writes_per_output_mem>(conv_layer_10_s2m, conv_layer_10_output_mem, Reps);

}
