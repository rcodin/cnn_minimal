#include <iostream>
#include <cstdlib>
#include <im2col.hpp>
#include <stdio.h>

float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[channel + channels*(col + row*width)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    // std::cout<<height_col<<std::endl;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = (c / channels) % ksize;
        int h_offset = c / ksize / channels;
        int c_im = c % channels;
        
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;

                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

void im2col_cpu_mod(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) {

    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (h = 0; h < height_col; ++h) {
        for (w = 0; w < width_col; ++w) {

            for (c = 0; c < channels_col; ++c) {
                int w_offset = (c / channels) % ksize;
                int h_offset = c / ksize / channels;
                int c_im = c % channels;

                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (h * width_col + w) * channels_col +c;

                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

float im2col_get_pixel_patch(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    // return 0;
    if (row < 0 || col < 0||
        row >= height || col >= width) return 0;

    return im[channel + channels * (col + row * width)];
}

float im2col_cpu_patch(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float *data_col)
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = (c / channels) % ksize;
        int h_offset = c / ksize / channels;
        int c_im = c % channels;
        
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}