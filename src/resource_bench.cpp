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
#include "resource_benchmark.h"
#include "MaxSLiCInterface.h"

using namespace std;

string make_file_name (int x) {
    stringstream ss;
    ss << "../resource_benchmark/inputs/input_";
    ss << x;
    ss << ".txt";
    return ss.str();
}

string make_output_file_name (int x) {
    stringstream ss;
    ss << "../resource_benchmark/outputs/output_";
    ss << x;
    ss << ".txt";
    return ss.str();
}

bool is_similar (float a, float b) {
    return fabs(a - b) < 0.0001;
}

map<string, string> load_config_file(string directory) {
    string line;
    map<string, string> m;
    ifstream fin(directory.c_str());

    while(getline(fin, line)) {
        if (line[0] == '#') {
            continue;
        }

        unsigned pos;

        for (pos = 0 ; pos < line.size() ; pos++) {
            if (line[pos] == '=') {
                break;
            }
        }

        if (pos == line.size()) {
            continue;
        }

        string key = line.substr(0, pos);
        string value = line.substr(pos + 1);

        m[key] = value;
    }

    fin.close();

    return m;
}

template <typename action_t>
void resource_benchmark(max_file_t* max_file, void (*run_fnc)(max_engine_t*, action_t*)) {

    map<string, string> m = load_config_file("../resource_benchmark/config.properties");
    const int input_channels = atoi(m["inputChannels"].c_str());
    const int output_channels = atoi(m["outputChannels"].c_str());
    const int input_height = atoi(m["inputHeight"].c_str());
    const int input_width = atoi(m["inputWidth"].c_str());
    const int kernel_dim = atoi(m["kernelDim"].c_str());
    const int output_height = input_height - (kernel_dim - 1);
    const int output_width = input_width - (kernel_dim - 1);

#ifdef __SIM__
    const int test_cases = 1;
#else
    const int test_cases = atoi(m["testCases"].c_str());
#endif

    /*
     * debugging output
     * cerr << input_channels << endl;
     * cerr << output_channels << endl;
     * cerr << input_height << endl;
     * cerr << input_width << endl;
     * cerr << kernel_dim << endl;
     * cerr << output_height << endl;
     * cerr << output_width << endl;
     * */

    float* x = new float[test_cases * input_height * input_width * input_channels];
    float* y = new float[test_cases * output_height * output_width * output_channels];

    // get input
    for (int t = 0 ; t < test_cases ; t++) {
        ifstream fin(make_file_name(t).c_str());
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

    cerr << "Done parsing input! putting into DFE now ..." << endl;
    cerr << "Starting job:!" << endl;

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

    cerr << std::fixed;
    cerr << std::setprecision(2);
    cerr << "It took " << delta << "micro seconds" << endl;
    cout << delta << endl;
    cerr << "DONE!" << endl;

    for (int t = 0 ; t < test_cases ; t++) {
        bool correct = true;
        cerr << "Test case " << t << " ";
        ifstream fin (make_output_file_name(t).c_str());

        for (int r = 0 ; r < output_height ; r++) {
            for (int c = 0 ; c < output_width ; c++) {
                for (int i = 0 ; i < output_channels ; i++) {
                    int idx = (output_height * output_width * output_channels) * t
                            + (output_width * output_channels) * r
                            + output_channels * c
                            + i;
                    float v;

                    fin >> v;
                    if (!is_similar(v, y[idx])) {
                        correct = false;
                        goto done;
                    }
                }
            }
        }
done:
        fin.close();

        if (correct) {
            cerr << "OKAY!" << endl;
        } else {
            cerr << "WRONG!" << endl;
        }
    }
}

int main(void) {
    max_file_t* max_file = resource_bench_1_3_init();
    resource_benchmark<resource_bench_1_3_actions_t>(max_file, resource_bench_1_3_run);
    resource_bench_1_3_free();
}
