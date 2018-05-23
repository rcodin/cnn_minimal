#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdlib>
#include <iostream>
#include <layers.hpp>
#include <float.h>
#include <layers.hpp>
#include <tiling.hpp>
#include <mkl.h>
#include <ctime>

//memory manager
void *alloc_1D(int i, size_t bytes);
void *alloc_2D(int i, int j, size_t bytes);
void *alloc_3D(int i, int j, int k, size_t bytes);
void *alloc_4D(int i, int j, int k, int l, size_t bytes);
void free_mem(void *ptr);

void print_conf_cfg(Conv_conf cfg, Data_conf input_cfg, Data_conf output_cfg);
void replicate_across_rows(float *input, float *output, int rows, int cols);
void replicate_across_cols(float *input, float *output, int rows, int cols);
unsigned int get_highest_prob(float *data, int data_size);
void load_tile(float *in, Data_conf input_conf, TILE_BASE tile_idx, int num_tiles,
					Conv_conf conv_cfg, float *out, Data_conf output_conf);

void save_tile(float *in, Data_conf input_conf, TILE_BASE tile_base,
					float *out, Data_conf output_conf);

Data_conf get_input_dependency(Data_conf output_conf, Conv_conf conv_cfg);

void create_rand_array(float *in, int size);

#endif