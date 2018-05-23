#include <utils.hpp>


void *alloc_1D(int i, size_t bytes) {
	void *ret = (void *)calloc(i, bytes);

	return ret;	
}


void *alloc_2D(int i, int j, size_t bytes) {
	void **ret = (void **)calloc(i, sizeof(void *));

	for (int i_idx = 0; i_idx < i; i_idx++)
		ret[i_idx] = calloc(j, bytes);
	return ret;	
}

void *alloc_3D(int i, int j, int k, size_t bytes) {
	void ***ret = (void ***)calloc(i, sizeof(void *));

	for (int i_idx = 0; i_idx < i; i_idx++) {
		ret[i_idx] = (void **)calloc(j, sizeof(void *));

		for (int j_idx = 0; j_idx < j; j_idx++)
			ret[i_idx][j_idx] = calloc(k, bytes);
	}

	// std::cout<<ret<<std::endl;
	if (ret == NULL)
		std::cout<<"Memory not allocated";
	return ret;
}

void *alloc_4D(int i, int j, int k, int l, size_t bytes) {
	void ****ret = (void ****)calloc(i, sizeof(void *));

	for (int i_idx = 0; i_idx < i; i_idx++) {
		ret[i_idx] = (void ***)calloc(j, sizeof(void *));

		for (int j_idx = 0; j_idx < j; j_idx++) {
			ret[i_idx][j_idx] = (void **)calloc(k, sizeof(void *));

			for (int k_idx = 0; k_idx < k; k_idx++)
				ret[i_idx][j_idx][k_idx] = calloc(l, bytes);
		}
	}

	// printf("%d\n", i * j * k * l * bytes);
	if (ret == NULL)
		std::cout<<"Memory not allocated";
	return ret;
}

void print_conf_cfg(Conv_conf cfg, Data_conf input_cfg, Data_conf output_cfg) {
	std::cout<<"Conv : ";
	std::cout<<"["<<output_cfg.c<<"x"<<cfg.h<<"x"<<cfg.w<<"x"<<input_cfg.c<<"]"<<std::endl;
	std::cout<<"Input : ";
	std::cout<<"["<<input_cfg.h<<"x"<<input_cfg.w<<"x"<<input_cfg.c<<std::endl;
}

void print_pool_cfg(Pool_conf pool_conf, Data_conf input_cfg) {
	std::cout<<"Pool : ";
	std::cout<<"["<<pool_conf.h<<"x"<<pool_conf.w<<"]"<<std::endl;
	std::cout<<"Input : ";
	std::cout<<"["<<input_cfg.h<<"x"<<input_cfg.w<<"x"<<input_cfg.c<<"]"<<std::endl;
	std::cout<<std::endl;
}

void print_linearize_conv_cfg(Data_conf input_cfg, int output_cfg) {
	std::cout<<"Input : "<<"["<<input_cfg.h<<"x"<<input_cfg.w<<"x"<<input_cfg.c<<"]"<<std::endl;
	std::cout<<"Output : "<<"["<<output_cfg<<"]"<<std::endl;
	std::cout<<std::endl;
}

void print_fc_cfg(int input_cfg, int output_cfg) {
	std::cout<<"Input : "<<"["<<input_cfg<<"]"<<std::endl;
	std::cout<<"Output : "<<"["<<output_cfg<<"]"<<std::endl;
	std::cout<<std::endl;
}

void free_mem(void *ptr) {
	free(ptr);
}

void replicate_across_cols(float *input, float *output, int rows, int cols) {
	for (int r = 0; r < rows; r++) {
		int val = input[r];
		for (int c = 0; c < cols; c++) {
			output[r * cols + c] += val;
		}
	}
}

void replicate_across_rows(float *input, float *output, int rows, int cols) {
	for (int r = 0; r < rows; r++) {
		// int val = input[r];
		for (int c = 0; c < cols; c++) {
			output[r * cols + c] += input[c];
		}
	}
}

unsigned int get_highest_prob(float *data, int data_size) {

	unsigned int max_idx;
	float max = -5.0f;
	for (int i = 0; i < data_size; i++) {
		if (max < data[i]) {
			max = data[i];
			max_idx = i;
		}
		// std::cout<<data[i]<<std::endl;
	}
	return (max_idx + 1);
}

//it load's part of the input in a tile

void load_tile(float *in, Data_conf input_conf, TILE_BASE tile_idx, int num_tiles,
					Conv_conf conv_cfg, float *out, Data_conf output_conf) {

	int h_base = tile_idx.h;
	int w_base = tile_idx.w;

	for (int out_h_idx = 0; out_h_idx < output_conf.h; out_h_idx++) {
    	for (int out_w_idx = 0; out_w_idx < output_conf.w; out_w_idx++) {
    		for (int c_idx = 0; c_idx < output_conf.c; c_idx++) {

    			int in_h_idx = h_base + out_h_idx - conv_cfg.pad;
    			int in_w_idx = w_base + out_w_idx - conv_cfg.pad;

    			int out_idx = (out_h_idx * output_conf.w + out_w_idx) * output_conf.c + c_idx;
	    		int in_idx = (in_h_idx * input_conf.w + in_w_idx) * input_conf.c + c_idx;

	    		if (in_h_idx >= 0 && in_h_idx < input_conf.h && 
	    						in_w_idx >= 0 && in_w_idx < input_conf.w)
    				out[out_idx] = in[in_idx];
    			else
    				out[out_idx] = 0;

    		}
    	}
    }

}

//save the tile when we have merge them to creatw the full output
void save_tile(float *in, Data_conf input_conf, TILE_BASE tile_base,
					float *out, Data_conf output_conf) {
	int h_base = tile_base.h;
	int w_base = tile_base.w;

	for (int h_idx = 0; h_idx < input_conf.h; h_idx++) {
    	for (int w_idx = 0; w_idx < input_conf.w; w_idx++) {
    		for (int c_idx = 0; c_idx < input_conf.c; c_idx++) {

    			int in_idx = (h_idx * input_conf.w + w_idx) * input_conf.c + c_idx;
    			int out_idx = ((h_idx + h_base) * output_conf.w + 
    							(w_idx + w_base)) * output_conf.c + c_idx;

    			out[out_idx] = in[in_idx];
    			in[in_idx] = 0;
    		}
    	}
    }
}


Data_conf get_input_dependency(Data_conf output_conf, Conv_conf conv_cfg) {

	Data_conf ret;

	ret.h = (output_conf.h - 1) * conv_cfg.stride - 2 * conv_cfg.pad + conv_cfg.h;
	ret.w = (output_conf.w - 1) * conv_cfg.stride - 2 * conv_cfg.pad + conv_cfg.w;

	return ret;
}

void alloc_patch(float *patch, Data_conf input_conf, int num_tiles, Conv_conf conv_cfg) {
	patch = (float *)mkl_calloc((input_conf.h/num_tiles) * (input_conf.w/num_tiles) * input_conf.c * 
						conv_cfg.h * conv_cfg.w, sizeof(float), sizeof(float) *8);
}

//fill an array with random array

void create_rand_array(float *in, int size) {
	srand(time(0));

	for (int i = 0; i < size; i++)
		in[i] = rand() % 255;
}