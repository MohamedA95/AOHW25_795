#pragma once

#ifdef __PRODEBUG__
#include "json.hpp"
#include <iostream>
#endif

#ifdef main_translation_unit
#define EXTERN
#else
#define EXTERN extern
#endif

#include <limits>
#include <cmath>
#include <cstdlib>
#include <fstream>

class Profiler_
{
private:
	float min_val;
	float max_val;

public:
	Profiler_() : min_val(std::numeric_limits<float>::max()), max_val(std::numeric_limits<float>::min()){}

	template
	<
	typename val_t
	>
	void update(val_t new_val)
	{
		float new_val_float = static_cast<float>(new_val);
		if(min_val > new_val_float){
			min_val = new_val_float;
		}
		if(max_val < new_val_float){
			max_val = new_val_float;
		}
	}

	float get_min(){
		return min_val;
	}

	float get_max(){
		return max_val;
	}

	unsigned int get_int()
	{
		int minimum = (int)min_val;
		int maximum = (int)max_val;
		minimum = (minimum != 0) ? (int)(std::log2((float)std::fabs(minimum))) + 1 : 0;
		maximum = (maximum != 0) ? (int)(std::log2((float)std::fabs(maximum))) + 1 : 0; 
		unsigned int int_bit_width = (minimum > maximum) ? minimum : maximum;
		return int_bit_width + 1;
	}
};

class Profiler
{	
private:
	Profiler_ input;
	Profiler_ input_dublicate;
	Profiler_ weight;
	Profiler_ acc;

	Profiler_ bias;	
	Profiler_ scale;
	Profiler_ shift;
	Profiler_ act_acc;
	Profiler_ output;
	
public:
	template<typename val_t> void update_input(val_t new_val){
		input.update(new_val);
	}

	template<typename val_t> void update_input_dublicate(val_t new_val){
		input_dublicate.update(new_val);
	}

	template<typename val_t> void update_weight(val_t new_val){
		weight.update(new_val);
	}
	
	template<typename val_t> void update_acc(val_t new_val){
		acc.update(new_val);
	}

	template<typename val_t> void update_bias(val_t new_val){
		bias.update(new_val);
	}

	template<typename val_t> void update_scale(val_t new_val){
		scale.update(new_val);
	}

	template<typename val_t> void update_shift(val_t new_val){
		shift.update(new_val);
	}

	template<typename val_t> void update_act_acc(val_t new_val){
		act_acc.update(new_val);
	}

	template<typename val_t> void update_output(val_t new_val){
		output.update(new_val);
	}

	#ifdef __PRODEBUG__
		std::string get_json_string(std::string layer_name)
		{
			nlohmann::json j;
			j["name"] = layer_name;

			j["ia_min"] = input.get_min();
			j["ia_max"] = input.get_max();
			j["ia_int_bits"] = input.get_int();

			j["fifo_ia_min"] = input_dublicate.get_min();
			j["fifo_ia_max"] = input_dublicate.get_max();
			j["fifo_ia_int_bits"] = input_dublicate.get_int();

			j["weight_min"] = weight.get_min();
			j["weight_max"] = weight.get_max();
			j["weight_int_bits"] = weight.get_int();

			j["bias_min"] = bias.get_min();
			j["bias_max"] = bias.get_max();
			j["bias_int_bits"] = bias.get_int();

			j["acc_min"] = acc.get_min();
			j["acc_max"] = acc.get_max();
			j["acc_int_bits"] = acc.get_int();

			j["scale_min"] = scale.get_min();
			j["scale_max"] = scale.get_max();
			j["scale_int_bits"] = scale.get_int();

			j["shift_min"] = shift.get_min();
			j["shift_max"] = shift.get_max();
			j["shift_int_bits"] = shift.get_int();

			j["act_acc_min"] = act_acc.get_min();
			j["act_acc_max"] = act_acc.get_max();
			j["act_acc_int_bits"]= act_acc.get_int();

			j["oa_min"] = output.get_min();
			j["oa_max"] = output.get_max();
			j["oa_int_bits"] = output.get_int();

			return("[ " +  j.dump(4) + " ]");
		}
	#endif

};

void get_json_params(std::string profilers_hpp_path);
