#include "dnn_profilers.hpp"

void get_json_params(std::string profiled_activations_path)
{{
	std::ofstream act_json;
	act_json.open(profiled_activations_path);
	act_json << "[" << std::endl;
	act_json << Conv_layer_0.get_json_string("Conv_layer_0") << "," << std::endl;
	act_json << Conv_layer_1.get_json_string("Conv_layer_1") << "," << std::endl;
	act_json << sdwc_0.get_json_string("sdwc_0") << "," << std::endl;
	act_json << Split_0.get_json_string("Split_0") << "," << std::endl;
	act_json << MaxPool_0.get_json_string("MaxPool_0") << "," << std::endl;
	act_json << sdwc_1.get_json_string("sdwc_1") << "," << std::endl;
	act_json << Conv_layer_2.get_json_string("Conv_layer_2") << "," << std::endl;
	act_json << sdwc_2.get_json_string("sdwc_2") << "," << std::endl;
	act_json << Conv_layer_3.get_json_string("Conv_layer_3") << "," << std::endl;
	act_json << sdwc_3.get_json_string("sdwc_3") << "," << std::endl;
	act_json << ConvTranspose_0.get_json_string("ConvTranspose_0") << "," << std::endl;
	act_json << sdwc_4.get_json_string("sdwc_4") << "," << std::endl;
	act_json << Concat_0.get_json_string("Concat_0") << "," << std::endl;
	act_json << sdwc_5.get_json_string("sdwc_5") << "," << std::endl;
	act_json << Conv_layer_4.get_json_string("Conv_layer_4") << "," << std::endl;
	act_json << Conv_layer_5.get_json_string("Conv_layer_5") << "," << std::endl;
	act_json << sdwc_6.get_json_string("sdwc_6") << "," << std::endl;
	act_json << Conv_layer_6.get_json_string("Conv_layer_6") << std::endl;
	act_json << "]" << std::endl;
	act_json.close();
}}
