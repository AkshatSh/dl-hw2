from uwnet import *

def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1, LRELU, 1),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1, LRELU, 1),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1, LRELU, 1),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1, LRELU, 1),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10, SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = 0.1 # (0.01) => 65 test (0.001) => 59 test (0.1) => 0.67 (1) => nan
momentum = .9
decay = .005

m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
#

'''
The conv net performs around 10% better. The conv net had a test accuracy of about 60%, where as the
feed forward network performs at about 50% accuracy on the test set.

I think the convnet performs better than a feedforward network is because the network design is able to capture spatial features by design which seems relevant to how people view images as well.
'''

'''
The highest accuracy I got 67.5 % on test with batch norm
'''

# 8*27*1024 + 16*72*256 + 32*144*64 + 64*288*16 + 256*10
# 1108480
# 221696
# 3072 input
# 72 out

