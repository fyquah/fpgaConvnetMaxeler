#!/bin/bash

cd "$(dirname $0)/.."

python scripts/generate_resource_benchmarks_makefile.py
python scripts/generate_resource_benchmarks.py
cd build/
make $@

