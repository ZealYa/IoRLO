from . import net_comp_performance as ncp
import numpy as np


# per-layer latency

def latency_conv(capability, input_size, output_size):
    return capability * (6.24e-5 * input_size * 1024 + 1.074e-4 * output_size * 1024 - 1.938)

def latency_relu(capability, input_size):
    return capability * (1.534e-5 * input_size * 1024 + 4.844e-1)

def latency_pool(capability, input_size, output_size):
    return capability * (1.136e-5 * input_size * 1024 + 1.131e-6 * output_size * 1024 - 1.695)

def latency_norm(capability, input_size):
    return capability * (5.182e-5 * input_size * 1024 + 6.497e-1)

def latency_fc(capability, input_size, output_size):
    return capability * (9.163e-5 * input_size * 1024 + 3.99e-4 * output_size * 1024 - 1.172)


# network latency

# AlexNet
def alexnet_latency(comp):
    alex_block_latency = [latency_conv(comp, 37.151, 265.72) + latency_relu(comp, 265.72),
                          latency_pool(comp, 327.255, 63.045) + latency_norm(comp, 63.045),
                          latency_conv(comp, 62.751, 158.56) + latency_relu(comp, 158.56),
                          latency_pool(comp, 199.864, 36.467) + latency_norm(comp, 36.467),
                          latency_conv(comp, 36.508, 43.472) + latency_relu(comp, 43.472),
                          latency_conv(comp, 54.248, 43.118) + latency_relu(comp, 43.118),
                          latency_conv(comp, 54.685, 29.877) + latency_relu(comp, 29.877),
                          latency_pool(comp, 36.999, 4.231),
                          latency_fc(comp, 4.231, 51.429),
                          latency_fc(comp, 64.688, 50.178) + latency_fc(comp, 64.362, 0.322)]

    alex_block_latency = np.array(alex_block_latency)
    alex_block_latency[alex_block_latency < 0] = 1
    return alex_block_latency
    # return tool.norm_array(alex_block_latency)


# VGGNet-16
def vgg16_latency(comp):
    vgg_block_latency = [latency_conv(comp, 37.151, 1.629) + latency_relu(comp, 1.629),
                        latency_conv(comp, 0.303, 3.691) + latency_relu(comp, 3.691) + latency_norm(comp, 0.309) + latency_pool(comp, 7.567, 2.458),
                        latency_conv(comp, 2.458, 1.907) + latency_relu(comp, 1.907),
                        latency_conv(comp, 0.277, 2.479) + latency_relu(comp, 2.479) + latency_norm(comp, 0.27) + latency_pool(comp, 4.884, 0.909),
                        latency_conv(comp, 0.909, 1.967) + latency_relu(comp, 1.967),
                        latency_conv(comp, 0.164, 3.372) + latency_relu(comp, 3.372),
                        latency_conv(comp, 0.226, 2.836) + latency_relu(comp, 2.836) + latency_norm(comp, 0.22) + latency_pool(comp, 2.107, 0.556),
                        latency_conv(comp, 0.556, 4.479) + latency_relu(comp, 4.479),
                        latency_conv(comp, 0.199, 7.078) + latency_relu(comp, 7.078),
                        latency_conv(comp, 0.172, 8.234) + latency_relu(comp, 8.234) + latency_norm(comp, 0.187) + latency_pool(comp, 1.054, 0.791),
                        latency_conv(comp, 0.791, 0.795) + latency_relu(comp, 0.795),
                        latency_conv(comp, 0.156, 12.045) + latency_relu(comp, 12.045),
                        latency_conv(comp, 0.201, 7.645) + latency_relu(comp, 7.645) + latency_norm(comp, 0.201) + latency_pool(comp, 0.555, 0.488),
                        latency_fc(comp, 0.488, 161.9) + latency_relu(comp, 161.9),
                        latency_fc(comp, 0.252, 1005.47) + latency_relu(comp, 1005.47) + latency_fc(comp, 0.811, 3.516)]

    vgg_block_latency = np.array(vgg_block_latency)
    vgg_block_latency[vgg_block_latency < 0] = 1
    return vgg_block_latency
    # return tool.norm_array(alex_block_latency)


# ResNet-32
def res32_latency(comp):
    res_block_latency = [latency_conv(comp, 37.151, 159.028) + latency_norm(comp, 159.028) + latency_relu(comp, 161.205),
                        latency_conv(comp, 180.582, 172.676) + latency_norm(comp, 172.676) + latency_conv(comp, 171.616, 163.914) + latency_norm(comp, 163.914),
                        latency_conv(comp, 165.25, 170.833) + latency_norm(comp, 170.833) + latency_conv(comp, 169.402, 164.717) + latency_norm(comp, 164.717),
                        latency_conv(comp, 165.222, 155.006) + latency_norm(comp, 155.006) + latency_conv(comp, 156.889, 166.765) + latency_norm(comp, 166.765),
                        latency_conv(comp, 165.575, 156.38) + latency_norm(comp, 156.38) + latency_conv(comp, 161.414, 167) + latency_norm(comp, 167),
                        latency_conv(comp, 171.773, 163.601) + latency_norm(comp, 163.601) + latency_conv(comp, 154.215, 160.697) + latency_norm(comp, 160.697),
                        latency_conv(comp, 159.431, 337.696) + latency_norm(comp, 337.696) + latency_conv(comp, 324.8, 323.249) + latency_norm(comp, 323.249),
                        latency_conv(comp, 326.306, 81.645) + latency_norm(comp, 81.645) + latency_conv(comp, 76.563, 80.02) + latency_norm(comp, 80.02),
                        latency_conv(comp, 81.152, 24.53) + latency_norm(comp, 24.53) + latency_conv(comp, 24.658, 23.358) + latency_norm(comp, 23.358),
                        latency_conv(comp, 23.862, 6.155) + latency_norm(comp, 6.155) + latency_conv(comp, 6.033, 5.853) + latency_norm(comp, 5.853),
                        latency_conv(comp, 5.849, 1.636) + latency_norm(comp, 1.636) + latency_conv(comp, 1.695, 1.732) + latency_norm(comp, 1.732),
                        latency_conv(comp, 1.692, 3.023) + latency_norm(comp, 3.023) + latency_conv(comp, 3.139, 3.162) + latency_norm(comp, 3.162),
                        latency_conv(comp, 2.991, 0.978) + latency_norm(comp, 0.978) + latency_conv(comp, 0.83, 0.927) + latency_norm(comp, 0.927),
                        latency_conv(comp, 0.882, 0.978) + latency_norm(comp, 0.978) + latency_conv(comp, 0.979, 0.986) + latency_norm(comp, 0.986),
                        latency_conv(comp, 0.925, 0.863) + latency_norm(comp, 0.863) + latency_conv(comp, 0.892, 0.951) + latency_norm(comp, 0.951),
                        latency_conv(comp, 0.954, 0.909) + latency_norm(comp, 0.909) + latency_conv(comp, 0.956, 0.929) + latency_norm(comp, 0.929),
                        latency_fc(comp, 0.94, 0.327)]

    res_block_latency = np.array(res_block_latency)
    res_block_latency[res_block_latency < 0] = 1
    return res_block_latency
    # return tool.norm_array(alex_block_latency)


# MobileNet-V1
def MobileNetV1_ConvBN_latency_T(comp, layer_1_input, layer_2_input, layer_3_input):
    return latency_conv(comp, layer_1_input, layer_2_input) + latency_norm(comp, layer_2_input) + latency_relu(comp, layer_3_input)

def MobileNetV1_ConvDW_latency_T(comp, layer_1_input, layer_2_input, layer_3_input, layer_4_input, layer_5_input, layer_6_input):
    ConvDW_1 = latency_conv(comp, layer_1_input, layer_2_input) + latency_norm(comp, layer_2_input) + latency_relu(comp, layer_3_input)
    ConvDW_2 = latency_conv(comp, layer_4_input, layer_5_input) + latency_norm(comp, layer_5_input) + latency_relu(comp, layer_6_input)
    return ConvDW_1 + ConvDW_2

def mobilenet_latency(comp):
    mobile_block_latency = [MobileNetV1_ConvBN_latency_T(comp, 37.151, 360.819, 358.695) + MobileNetV1_ConvDW_latency_T(comp, 400.633, 353.338, 369.152, 415.551, 710.224, 721.285),
                           MobileNetV1_ConvDW_latency_T(comp, 849.531, 727.631, 714.656, 823.222, 1418.259, 1435.540),
                           MobileNetV1_ConvDW_latency_T(comp, 1646.346, 1445.826, 1442.646, 1636.792, 1410.675, 1405.317) + MobileNetV1_ConvDW_latency_T(comp, 1714.335, 357.550, 363.434, 410.188, 731.600, 712.675),
                           MobileNetV1_ConvDW_latency_T(comp, 838.411, 715.715, 719.103, 829.064, 717.434, 718.336) + MobileNetV1_ConvDW_latency_T(comp, 837.796, 178.101, 179.481, 209.119, 359.812, 360.391),
                           MobileNetV1_ConvDW_latency_T(comp, 406.291, 357.623, 360.171, 407.018, 361.224, 357.101) + MobileNetV1_ConvDW_latency_T(comp, 416.107, 360.842, 355.682, 407.582, 357.030, 358.636) + MobileNetV1_ConvDW_latency_T(comp, 410.672, 359.303, 360.202, 410.496, 359.222, 357.455) + MobileNetV1_ConvDW_latency_T(comp, 416.839, 358.109, 358.989, 407.486, 358.522, 359.605) + MobileNetV1_ConvDW_latency_T(comp, 415.415, 357.375, 359.892, 407.386, 358.854, 358.981),
                           MobileNetV1_ConvDW_latency_T(comp, 413.797, 90.327, 90.939, 102.085, 180.121, 179.563) + MobileNetV1_ConvDW_latency_T(comp, 207.440, 44.696, 45.488, 51.513, 45.151, 44.788) + latency_fc(comp, 51.716, 0.316)]

    mobile_block_latency = np.array(mobile_block_latency)
    mobile_block_latency[mobile_block_latency < 0] = 1
    return mobile_block_latency
    # return tool.norm_array(mobile_block_latency)


# standard mobile capability = 0.38 * mobile computing capability + 0.065

# AlexNet

# Mobile device is 2.4 GHZ (2400 mHZ, CPU=1)
def alexnet_latency_T():
    return alexnet_latency_C() * (2.0 * 56) / (2.4 * 4 * 0.38 + 0.065)
    # return alexnet_latency(ncp.C_T)

# Edge server is 2.3 GHZ (2300 mHZ, CPU=16)
def alexnet_latency_E():
    return alexnet_latency_C() * (2.0 * 56) / (2.3 * 16)
    # return alexnet_latency(ncp.C_E)

# Cloud server is 2.0 GHz (2000 mHZ, CPU=56)
def alexnet_latency_C():  # ms
    alex_ms = np.array([2.986, 3.02, 5.315, 1.37, 5.567, 5.012, 3.994, 0.421, 5.018, 53.17])
    # return alex_ms / 1000  # s
    return alex_ms  # s

# Helper device is Xiaomi 8 (2.8 * 4 + 1.8 * 4)
def alexnet_latency_H_h():
    return alexnet_latency_C() * (2.0 * 56) / ((2.8 * 4 + 1.8 * 4) * 0.38 + 0.065)

# Helper device is 2.4 GHZ (2400 mHZ, CPU=4) * 0.6
def alexnet_latency_H_l():
    return alexnet_latency_C() * (2.0 * 56) / (2.4 * 4 * 0.6 * 0.38 + 0.065)


# VGGNet-16

# Mobile device is 2.4 GHZ (2400 mHZ, CPU=4)
def vgg16_latency_T():
    return vgg16_latency_C() * (2.0 * 56) / (2.4 * 4 * 0.38 + 0.065)
    # return alexnet_latency(ncp.C_T)

# Edge server is 2.3 GHZ (2300 mHZ, CPU=16)
def vgg16_latency_E():
    return vgg16_latency_C() * (2.0 * 56) / (2.3 * 16)
    # return alexnet_latency(ncp.C_E)

# Cloud server is 2.0 GHz (2000 mHZ, CPU=56)
def vgg16_latency_C():  # ms
    vgg_ms = np.array([2.021, 15.829, 1.87, 7.969, 2.2, 4.516, 6.877, 5.324, 9.476, 11.467, 0.921, 11.215, 10.744, 138.897, 934.91])
    # return vgg_ms / 1000  # s
    return vgg_ms  # ms

# Helper device is Xiaomi 8 (2.8 * 4 + 1.8 * 4)
def vgg16_latency_H_h():
    return vgg16_latency_C() * (2.0 * 56) / ((2.8 * 4 + 1.8 * 4) * 0.38 + 0.065)

# Helper device is 2.4 GHZ (2400 mHZ, CPU=4) * 0.6
def vgg16_latency_H_l():
    return vgg16_latency_C() * (2.0 * 56) / (2.4 * 4 * 0.6 * 0.38 + 0.065)


# ResNet-32

# Mobile device is 2.4 GHZ (2400 mHZ, CPU=4)
def res32_latency_T():
    return res32_latency_C() * (2.0 * 56) / (2.4 * 4 * 0.38 + 0.065)
    # return alexnet_latency(ncp.C_T)

# Edge server is 2.3 GHZ (2300 mHZ, CPU=16)
def res32_latency_E():
    return res32_latency_C() * (2.0 * 56) / (2.3 * 16)
    # return alexnet_latency(ncp.C_E)

# Cloud server is 2.0 GHz (2000 mHZ, CPU=56)
def res32_latency_C():  # ms
    res32_ms = np.array([3.627, 6.785, 6.284, 6.119, 5.953, 5.966, 9.534, 4.138, 2.519, 1.904, 1.686, 1.881, 2.512, 1.861, 1.951, 2.031, 1.311])
    # return res32_ms / 1000  # s
    return res32_ms  # ms

# Helper device is Xiaomi 8 (2.8 * 4 + 1.8 * 4)
def res32_latency_H_h():
    return res32_latency_C() * (2.0 * 56) / ((2.8 * 4 + 1.8 * 4) * 0.38 + 0.065)

# Helper device is 2.4 GHZ (2400 mHZ, CPU=4) * 0.6
def res32_latency_H_l():
    return res32_latency_C() * (2.0 * 56) / (2.4 * 4 * 0.6 * 0.38 + 0.065)


# MobileNet-V1

# Mobile device is 2.4 GHZ (2400 mHZ, CPU=4)
def mobilenet_latency_T():
    return mobilenet_latency_C() * (2.0 * 56) / (2.4 * 4 * 0.38 + 0.065)
    # return alexnet_latency(ncp.C_T)

# Edge server is 2.3 GHZ (2300 mHZ, CPU=16)
def mobilenet_latency_E():
    return mobilenet_latency_C() * (2.0 * 56) / (2.3 * 16)
    # return alexnet_latency(ncp.C_E)

# Cloud server is 2.0 GHz (2000 mHZ, CPU=56)
def mobilenet_latency_C():  # ms
    mobilenet_ms = np.array([9.413, 17.232, 20.221, 14.420, 28.877, 8.187])
    # return mobilenet_ms / 1000  # s
    return mobilenet_ms  # ms

# Helper device is Xiaomi 8 (2.8 * 4 + 1.8 * 4)
def mobilenet_latency_H_h():
    return mobilenet_latency_C() * (2.0 * 56) / ((2.8 * 4 + 1.8 * 4) * 0.38 + 0.065)

# Helper device is 2.4 GHZ (2400 mHZ, CPU=4) * 0.6
def mobilenet_latency_H_l():
    return mobilenet_latency_C() * (2.0 * 56) / (2.4 * 4 * 0.6 * 0.38 + 0.065)
