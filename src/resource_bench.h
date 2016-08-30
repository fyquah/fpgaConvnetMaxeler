#ifndef RESOURCE_BENCH_H
#define RESOURCE_BENCH_H 
#include <cstdio>
#include <cmath>
#include <cstdlib>

#include <string>
#include <vector>
#include <iostream>
#include <map>
#include <fstream>
#include <sstream>
#include <iomanip>

#include <sys/time.h>
#include "MaxSLiCInterface.h"

std::string make_file_name(int x);
std::string make_output_file_name(int x);
bool is_similar(float, float);
std::map<std::string, std::string> load_config_file(std::string directory);

template <typename action_t>
void resource_benchmark(max_file_t* max_file, void (*run_fnc)(max_engine_t*, action_t*), std::string out_file_name) {

    std::map<std::string, std::string> m = load_config_file("../resource_benchmark/config.properties");
    const int input_channels = atoi(m["inputChannels"].c_str());
    const int output_channels = atoi(m["outputChannels"].c_str());
    const int input_height = atoi(m["inputHeight"].c_str());
    const int input_width = atoi(m["inputWidth"].c_str());
    const int kernel_dim = atoi(m["kernelDim"].c_str());
    const int output_height = input_height - (kernel_dim - 1);
    const int output_width = input_width - (kernel_dim - 1);

#ifdef __SIM__
    const int test_cases = 20;
#else
    const int test_cases = atoi(m["testCases"].c_str());
#endif

     std::cerr << "input channels = " << input_channels << std::endl;
     std::cerr << "output channels = " << output_channels << std::endl;
     std::cerr << "input height = " << input_height << std::endl;
     std::cerr << "input width = " << input_width << std::endl;
     std::cerr << "kernel dim = " << kernel_dim << std::endl;
     std::cerr << "output height = " << output_height << std::endl;
     std::cerr << "output width =" << output_width << std::endl;

    float* x = new float[test_cases * input_height * input_width * input_channels];
    float* y = new float[test_cases * output_height * output_width * output_channels];

    // get input
    for (int t = 0 ; t < test_cases ; t++) {
        std::ifstream fin(make_file_name(t).c_str());
        for (int r = 0 ; r < input_height ; r++) {
            for (int c = 0 ; c < input_width ; c++) {
                for (int i = 0 ; i < input_channels ; i++) {
                    int idx = (input_height * input_width * input_channels) * t
                            + (input_width * input_channels) * r
                            + input_channels * c
                            + i;
                    fin >> x[idx];
                }
            }
        }
        fin.close();
    }

    std::cerr << "Done parsing input! putting into DFE now ..." << std::endl;
    std::cerr << "Starting job:!" << std::endl;

    timeval t_begin, t_end;
    max_engine_t *engine = max_load(max_file, "*");
    action_t actions;
    actions.param_N = test_cases;
    actions.instream_x = x;
    actions.outstream_y = y;

    gettimeofday(&t_begin, NULL);
    run_fnc(engine, &actions);
    gettimeofday(&t_end, NULL);

    double begin = t_begin.tv_sec * 1000000 + t_begin.tv_usec;
    double end = t_end.tv_sec * 1000000 + t_end.tv_usec;
    double delta = end - begin;

    std::ofstream fout(out_file_name.c_str());

    std::cerr << std::fixed;
    std::cerr << std::setprecision(7);
    std::cerr << "It took " << delta << "micro seconds" << std::endl;
    fout << delta << std::endl;
    fout.close();
    std::cerr << "DONE!" << std::endl;

    for (int t = 0 ; t < test_cases ; t++) {
        bool correct = true;
        int num_correct = 0;
        int total = output_height * output_width * output_channels;
        std::cerr << "Test case " << t << " ";
        std::ifstream fin (make_output_file_name(t).c_str());

        for (int r = 0 ; r < output_height ; r++) {
            for (int c = 0 ; c < output_width ; c++) {
                for (int i = 0 ; i < output_channels ; i++) {
                    int idx = (output_height * output_width * output_channels) * t
                            + (output_width * output_channels) * r
                            + output_channels * c
                            + i;
                    float v;

                    fin >> v;
                    std::cout << v << " vs " << y[idx] << std::endl;
                    if (!is_similar(v, y[idx])) {
                        correct = false;
                    } else {
                        num_correct += 1;
                    }
                }
            }
        }
        fin.close();

        if (correct) {
            std::cerr << "OKAY!" << std::endl;
        } else {
            std::cerr << "WRONG!" << std::endl;
        }
        std::cerr << "Correct proportion = " << ((float) num_correct / (float) total) << std::endl;
    }

    max_unload(engine);
}

#endif
