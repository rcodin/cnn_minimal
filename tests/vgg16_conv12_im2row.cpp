#include <stdio.h>
#include <cstdlib>
#include <stdint.h>
#include <layers.hpp>
#include <utils.hpp>
#include <mkl.h>
#include <cnpy.hpp>
#include <input.hpp>
#include <iomanip>
#include <fstream>
#include <tiling.hpp>
#include <utility>

using namespace std;

int main() {
	Conv_conf conv11_conf = {3, 3, 1, 1};
	Data_conf input11_conf = {224, 224, 3};
	Data_conf output11_conf = {224, 224, 64};

	Conv_conf conv12_conf = {3, 3, 1, 1};
	Data_conf input12_conf = {224, 224, 64};
	Data_conf output12_conf = {224, 224, 64};


	size_t bytes = sizeof(float);
	int alignment = bytes * 8;

	bool tiled = false;

	int h_num_tiles = 8;
	int w_num_tiles = 8;

	Conv_conf conv11_tiled_conf = {3, 3, 1, 0};

	Data_conf input11_tiled_conf = {input11_conf.h/h_num_tiles + (conv11_conf.h - 1),
						input11_conf.w/w_num_tiles  + (conv11_conf.w - 1), input11_conf.c};
	Data_conf output11_tiled_conf = {output11_conf.h/h_num_tiles, output11_conf.w/w_num_tiles, output11_conf.c};


	Conv_conf conv12_tiled_conf = {3, 3, 1, 0};
	Data_conf input12_tiled_conf = {input12_conf.h/h_num_tiles + (conv12_conf.h - 1),
					input12_conf.w/w_num_tiles  + (conv12_conf.w - 1), input12_conf.c};
	Data_conf output12_tiled_conf = {output12_conf.h/h_num_tiles, output12_conf.w/w_num_tiles, output12_conf.c};


	float *input11 = (float *)mkl_calloc(input11_conf.h * input11_conf.w *
		input11_conf.c, bytes, alignment);

	float *output11 = (float *)mkl_calloc(output11_conf.h * output11_conf.w *
		output11_conf.c, bytes, alignment);

	float *input11_tiled = (float *)mkl_calloc(input11_tiled_conf.h * input11_tiled_conf.w *
		input11_tiled_conf.c, bytes, alignment);

	float *output11_tiled = (float *)mkl_calloc(output11_tiled_conf.h * output11_tiled_conf.w *
		output11_tiled_conf.c, bytes, alignment);



	float *input12 = output11;

	float *output12 = (float *)mkl_calloc(output12_conf.h * output12_conf.w *
		output12_conf.c, bytes, alignment);

	float *input12_tiled = (float *)mkl_calloc(input12_tiled_conf.h * input12_tiled_conf.w *
		input12_tiled_conf.c, bytes, alignment);

	float *output12_tiled = (float *)mkl_calloc(output12_tiled_conf.h * output12_tiled_conf.w *
		output12_tiled_conf.c, bytes, alignment);


    std::string weight_dir = "/home/roni/project/files/vgg_16/tensorflow/weights_dir/";
    std::string image_file = "/home/roni/project/files/vgg_16/tensorflow/laska.png";

    Image_cfg input_cfg = {224, 224};
    // float *input = (float *)malloc(input_cfg.rows * input_cfg.cols * 3);

    int err = read_image_rgb(image_file, input_cfg, input11);

    ifstream im_file;

    im_file.open("/home/roni/project/files/vgg_16/tensorflow/image_out.log");

    for (int i = 0; i < (224 * 224 *3); i++) {
    	im_file>>input11[i];
    	// cout<<input11[i];
    }


	float *conv11_weights;
	float *conv11_biases;


	float *conv12_weights;
	float *conv12_biases;
	
	cnpy::NpyArray arr11 = cnpy::npy_load(weight_dir+"conv1_1_W.npy");
	conv11_weights = arr11.data<float>();

    cnpy::NpyArray arr11_biases = cnpy::npy_load(weight_dir+"conv1_1_b.npy");
	conv11_biases = arr11_biases.data<float>();


	cnpy::NpyArray arr12 = cnpy::npy_load(weight_dir+"conv1_2_W.npy");
	conv12_weights = arr12.data<float>();

    cnpy::NpyArray arr12_biases = cnpy::npy_load(weight_dir+"conv1_2_b.npy");
	conv12_biases = arr12_biases.data<float>();

	auto start = std::chrono::system_clock::now();
	auto end = std::chrono::system_clock::now();

	if (tiled) {
		// while (1)
		for (int h_tile = 0; h_tile < h_num_tiles; h_tile++) {
			for (int w_tile = 0; w_tile < w_num_tiles; w_tile++) {

				int h_base = h_tile * (input11_tiled_conf.h - (conv11_conf.h - 1));
				int w_base = w_tile * (input11_tiled_conf.w - (conv11_conf.w - 1));

				TILE_BASE tile_base = {h_base, w_base};

				load_tile(input11, input11_conf, tile_base, h_num_tiles, 
							conv11_conf, input11_tiled, input11_tiled_conf);

				conv_im2row(input11_tiled, output11_tiled, conv11_weights, conv11_biases, conv11_tiled_conf,
					input11_tiled_conf, output11_tiled_conf);

				save_tile(output11_tiled, output11_tiled_conf, tile_base, output11, output11_conf);

			}
		}
		start = std::chrono::system_clock::now();
		for (int h_tile = 0; h_tile < h_num_tiles; h_tile++) {
			for (int w_tile = 0; w_tile < w_num_tiles; w_tile++) {

				int h_base = h_tile * (input12_tiled_conf.h - (conv12_conf.h - 1));
				int w_base = w_tile * (input12_tiled_conf.w - (conv12_conf.w - 1));

				TILE_BASE tile_base = {h_base, w_base};

				load_tile(input12, input12_conf, tile_base, h_num_tiles, 
							conv12_conf, input12_tiled, input12_tiled_conf);

				conv_im2row(input12_tiled, output12_tiled, conv12_weights, conv12_biases, conv12_tiled_conf,
					input12_tiled_conf, output12_tiled_conf);

				save_tile(output12_tiled, output12_tiled_conf, tile_base, output12, output12_conf);

			}
		}
		end = std::chrono::system_clock::now();
	}
	else {
		conv_im2row(input11, output11, conv11_weights, conv11_biases, conv11_conf,
				input11_conf, output11_conf);
		
		start = std::chrono::system_clock::now();
		conv_im2row(input12, output12, conv12_weights, conv12_biases, conv12_conf,
				input12_conf, output12_conf);
		end = std::chrono::system_clock::now();
	}


	// std::chrono::duration<double> elapsed_time = end-start;

	// cout<<elapsed_time.count()<<endl;

	// for (int i = 0; i < output12_conf.h; i++) {
	// 	for (int j = 0; j < output12_conf.h; j++) {
	// 		for (int k = 0; k < output12_conf.c; k++) {
	// 			int idx = (i * output12_conf.w + j) * output12_conf.c + k;
	// 			cout<<fixed<<setprecision(10)<<output12[idx]<<endl;
	// 		}
	// 	}
	// }
}