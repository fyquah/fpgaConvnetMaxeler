#include <iostream>
#include <cstdlib>
#include "scene.h"


template<typename T>
const int INPUT_SIZE = 230400;
const int OUTPUT_SIZE = 865536;


int main() {
#ifdef __SIM__
    int N = 1;
#else
    int N = 1000;
#endif

    std::vector<std::vector<double> > images;
    std::vector<int> labels;
    float *x = new float[N * INPUT_SIZE];
    float *conv_out = new float[N * OUTPUT_SIZE];

    try {
        std::cout << "Initializing input data ..." << std::endl;
        for (int i = 0 ; i < N * INPUT_SIZE ; i++) {
            x[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }

        std::cout << "Running CNN ... " << std::endl;
        max_file_t *max_file = scene_init();
        max_engine_t *dfe = max_load(max_file, "*");
        lenet_actions_t action;

        action.param_N = N;
        action.instream_x = x;
        action.outstream_y = conv_out;

        timeval t_begin, t_end;
        gettimeofday(&t_begin, NULL);
        lenet_run(dfe, &action);
        gettimeofday(&t_end, NULL);
	std::cout << "Completed feature extraction!" << std::endl;
        max_unload(dfe);

        /* Begin calculating the time taken and performance stuff. */
        double begin = double(t_begin.tv_sec) * 1000000 + double(t_begin.tv_usec);
        double end = double(t_end.tv_sec) * 1000000 + double(t_end.tv_usec);
        double delta = end - begin;
	double throughput = double(N) / delta * 1000000;
        /*
         * totalGOps calculated by
         * (234 * 314 * 49 * 16 * 3
         *  + 111 * 151 * 49 * 64 * 16
         *  + 49 * 69 * 49 * 256 * 64) * 2.0 / 1e9
         *  Hardcoding value here just for convenience.
         */
        double totalGOps = 7.5663;

        std::cout << "t_begin.tv_sec = " << t_begin.tv_sec << std::endl;
        std::cout << "t_begin.tv_usec = " << t_begin.tv_usec << std::endl;
        std::cout << "t_end.tv_sec = " << t_end.tv_sec << std::endl;
        std::cout << "t_end.tv_usec = " << t_end.tv_usec << std::endl;
        std::cout << "Total time = " << delta << std::endl;
        std::cout << "Throughput (images per second) = " << throughput << std::endl;
	std::cout << "GOps = " << throughput * 7.5663 << std::endl;

    } catch (const std::string & s) {
        std::cerr << "Caught an error!" << std::endl;
        std::cerr << s << std::endl;
    }

    return 0;
}
