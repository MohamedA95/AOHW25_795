#define main_translation_unit
#include <string> 
#include <iostream>
#include <sstream>
#include <fstream>
#include <random>
#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include "../include_hls/dnn_config.hpp"
#include "../include_hls/dnn_top_level_wrapper.hpp"

template<size_t NUM_BITS> using stream_type = ap_uint<NUM_BITS>;

int main()
{
	stream_type<INPUT_MEM_WIDTH> input_mem[num_packed_inputs * Reps];
	stream_type<OUTPUT_MEM_WIDTH> output_mem[num_packed_outputs * Reps];
	stream_type<INPUT_MEM_WIDTH> data = 0;
	INPUT_dtype input;
	float input_float;
	int input_mem_counter = 0;
	static std::default_random_engine generator;
	static std::uniform_real_distribution<float> distribution(0.0, 1.0);
	for (int sample = 0; sample < 1; ++sample){
		for (int i = 0; i < num_packed_inputs; ++i){
			for (int j = 0; j < INPUT_MEM_WIDTH / CONV_LAYER_0_IFM_BITS; j++){
				input_float = distribution(generator);
				input_float = nearbyint(input_float/INPUT_SCALE) + INPUT_ZERO_POINT;
				input_float = std::max(-128.0f, std::min(input_float, 127.0f));
				input = static_cast<INPUT_dtype>(input_float);
				data(j * CONV_LAYER_0_IFM_BITS + CONV_LAYER_0_IFM_BITS - 1, j * CONV_LAYER_0_IFM_BITS) = *reinterpret_cast<stream_type<CONV_LAYER_0_IFM_BITS> *>(&input);
			}
			input_mem[input_mem_counter] = data;
			input_mem_counter++;
			data = 0;
		}
	}

	top_level_wrapper(input_mem, output_mem);

	return 0;
}
