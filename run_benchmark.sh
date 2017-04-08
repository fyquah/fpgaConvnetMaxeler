#!/bin/bash
MAXOS_HW=/opt/maxeler/maxeleros/lib/libmaxeleros.so
export SLIC_CONF="default_engine_resource = fyq14^192.168.0.1"
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $PROJECT_ROOT/projects/cifar10_quick_maxring_3/build
MAXOS_HW=$MAXOS_HW ./target_dfe cifar10_quick_maxring.optimized.prototxt

cd $PROJECT_ROOT/projects/lenet_maxring/build
MAXOS_HW=$MAXOS_HW ./target_dfe lenet_maxring.optimized.prototxt
