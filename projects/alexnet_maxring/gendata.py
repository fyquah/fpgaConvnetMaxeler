import caffe
import numpy as np
import struct
import logging

# Maps from fpgaconvnet to caffe
layers_mapping = {
        "conv0": "conv1",
        "lrn1" : "norm1",
        "pool2": "pool1",
        "conv3": "conv2",
        "lrn4" : "norm2",
        "pool5": "pool2",
        "conv6": "conv3",
        "conv7": "conv4",
        "conv8": "conv5",
        "pool9": "pool5",
        }
last_layer_key = "conv4"

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
        if "conv" not in fpgaconvnet_key:
            continue
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

    im_output = np.transpose(net.blobs["pool5"].data, (0, 2, 3, 1))
    logging.info("Input shape = %s" % str(im_input.shape))
    logging.info("Output shape = %s" % str(im_output.shape))

    logging.info("Writing input data to data/input.bin")
    save_to_file("testdata/data/input.bin", im_input.flatten())
    logging.info("Writing input data to data/output.bin")
    save_to_file("testdata/data/output.bin", im_output.flatten())

    for fpgaconvnet_key, caffe_key in layers_mapping.iteritems():
        data = np.transpose(net.blobs[caffe_key].data, (0, 2, 3, 1))
        fname = "testdata/data/%s.bin" % fpgaconvnet_key
        logging.info("Writing input data to %s" % fname)
        save_to_file(fname, data.flatten())


if __name__ == "__main__":
    main()
