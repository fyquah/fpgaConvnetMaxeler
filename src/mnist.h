#ifndef MNIST_H
#define MNIST_H

void read_mnist_images(vector<vector<double> > &arr, std::string filename);
void read_mnist_labels(vector<int> & arr, std::string filename);

#endif
