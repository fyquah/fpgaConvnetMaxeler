import caffe
import numpy as np
import struct
import logging

# Maps from fpgaconvnet to caffe
layers_mapping = {
        "conv0": "conv1_1",
        "conv1": "conv1_2",
        "conv3": "conv2_1",
        "conv4": "conv2_2",
        "conv6": "conv3_1",
        "conv7": "conv3_2",
        "conv8": "conv3_3",
        "conv10": "conv4_1",
        "conv11": "conv4_2",
        "conv12": "conv4_3",
        "conv14": "conv5_1",
        "conv15": "conv5_2",
        "conv16": "conv5_3",
        }
last_layer_key = "pool5"

caffe.set_mode_cpu()


def save_to_file(fname, data):
    # logging.info("Saving %s to %s" % (str(data), fname))
    with open(fname, "wb") as f:
        f.write(struct.pack("%df" % len(data), *data))


def main():
    logging.getLogger().setLevel(logging.INFO)
    net = caffe.Net('deploy.prototxt', 'model.caffemodel', caffe.TEST)

    # weights
    for fpgaconvnet_key, caffe_key in layers_mapping.iteritems():
        save_to_file(
                "testdata/weights/" + fpgaconvnet_key + "_weights",
                net.params[caffe_key][0].data.flatten())
        save_to_file(
                "testdata/weights/" + fpgaconvnet_key + "_bias",
                net.params[caffe_key][1].data.flatten())
    logging.info("Saved weights and biases into weights/ directory")

    im_input = np.random.uniform(size=net.blobs['data'].data.shape)
    net.blobs['data'].data[...] = im_input
    net.forward()

    im_input  = np.transpose(im_input, (0, 2, 3, 1))
    im_output = np.transpose(net.blobs[last_layer_key].data, (0, 2, 3, 1))

    print np.transpose(net.blobs["pool5"].data, (0, 2, 3, 1))
    print np.transpose(net.blobs["pool4"].data, (0, 2, 3, 1))

    logging.info("Input shape = %s" % str(im_input.shape))
    logging.info("Output shape = %s" % str(im_output.shape))

    logging.info("Writing input data to data/input.bin")
    save_to_file("testdata/data/input.bin", im_input.flatten())
    logging.info("Writing input data to data/output.bin")
    save_to_file("testdata/data/output.bin", im_output.flatten())


if __name__ == "__main__":
    main()
