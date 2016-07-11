#ifndef MNIST_H
#define MNIST_H

#include <string>
#include <vector>

void read_mnist_images(std::vector<std::vector<double> > &arr, std::string filename);
void read_mnist_labels(std::vector<int> & arr, std::string filename);

#endif
