#include "resource_bench_helper.h"

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

int main () {
    run_resource_bench();
}
