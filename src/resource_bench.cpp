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
void resource_benchmark(max_file_t* max_file, void (*run_fnc)(max_engine_t*, action_t*), string out_file_name) {

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

    ofstream fout(out_file_name.c_str());

    cerr << std::fixed;
    cerr << std::setprecision(7);
    cerr << "It took " << delta << "micro seconds" << endl;
    fout << delta << endl;
    fout.close();
    cerr << "DONE!" << endl;

    for (int t = 0 ; t < test_cases ; t++) {
        bool correct = true;
        int num_correct = 0;
        int total = output_height * output_width * output_channels;
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
                    } else {
                        num_correct += 1;
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
        cerr << "Correct percentage = %.5f" << ((float) num_correct / (float) total) << endl;
    }
}

int main () {

    max_file_t* max_file_1_3 = resource_bench_1_3_init();
    resource_benchmark<resource_bench_1_3_actions_t>(max_file_1_3, resource_bench_1_3_run, "1_3.out");
    resource_bench_1_3_free();
    

    max_file_t* max_file_5_6 = resource_bench_5_6_init();
    resource_benchmark<resource_bench_5_6_actions_t>(max_file_5_6, resource_bench_5_6_run, "5_6.out");
    resource_bench_5_6_free();
    

    max_file_t* max_file_1_13 = resource_bench_1_13_init();
    resource_benchmark<resource_bench_1_13_actions_t>(max_file_1_13, resource_bench_1_13_run, "1_13.out");
    resource_bench_1_13_free();
    

    max_file_t* max_file_10_1 = resource_bench_10_1_init();
    resource_benchmark<resource_bench_10_1_actions_t>(max_file_10_1, resource_bench_10_1_run, "10_1.out");
    resource_bench_10_1_free();
    

    max_file_t* max_file_4_9 = resource_bench_4_9_init();
    resource_benchmark<resource_bench_4_9_actions_t>(max_file_4_9, resource_bench_4_9_run, "4_9.out");
    resource_bench_4_9_free();
    

    max_file_t* max_file_1_6 = resource_bench_1_6_init();
    resource_benchmark<resource_bench_1_6_actions_t>(max_file_1_6, resource_bench_1_6_run, "1_6.out");
    resource_bench_1_6_free();
    

    max_file_t* max_file_7_2 = resource_bench_7_2_init();
    resource_benchmark<resource_bench_7_2_actions_t>(max_file_7_2, resource_bench_7_2_run, "7_2.out");
    resource_bench_7_2_free();
    

    max_file_t* max_file_5_7 = resource_bench_5_7_init();
    resource_benchmark<resource_bench_5_7_actions_t>(max_file_5_7, resource_bench_5_7_run, "5_7.out");
    resource_bench_5_7_free();
    

    max_file_t* max_file_2_6 = resource_bench_2_6_init();
    resource_benchmark<resource_bench_2_6_actions_t>(max_file_2_6, resource_bench_2_6_run, "2_6.out");
    resource_bench_2_6_free();
    

    max_file_t* max_file_1_20 = resource_bench_1_20_init();
    resource_benchmark<resource_bench_1_20_actions_t>(max_file_1_20, resource_bench_1_20_run, "1_20.out");
    resource_bench_1_20_free();
    

    max_file_t* max_file_2_9 = resource_bench_2_9_init();
    resource_benchmark<resource_bench_2_9_actions_t>(max_file_2_9, resource_bench_2_9_run, "2_9.out");
    resource_bench_2_9_free();
    

    max_file_t* max_file_4_8 = resource_bench_4_8_init();
    resource_benchmark<resource_bench_4_8_actions_t>(max_file_4_8, resource_bench_4_8_run, "4_8.out");
    resource_bench_4_8_free();
    
}
