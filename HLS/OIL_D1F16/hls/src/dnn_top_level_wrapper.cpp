#include "bnn-library.h"
#include "dnn_config.hpp"
#include "dnn_top_level_wrapper.hpp"
#include "dnn_params_0.hpp"
#include "./dnn_top_level_0.cpp"

void top_level_wrapper(ap_uint<INPUT_MEM_WIDTH> *conv_layer_0_input_mem,
	ap_uint<OUTPUT_MEM_WIDTH> *conv_layer_6_output_mem)
{

	std::cout << "Top level: 0" << std::endl;
	top_level_0(conv_layer_0_input_mem,
		conv_layer_6_output_mem);

}
