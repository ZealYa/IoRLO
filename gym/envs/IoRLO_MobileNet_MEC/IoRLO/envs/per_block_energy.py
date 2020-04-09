from . import net_comp_performance as ncp
import numpy as np


# per-layer energy
# return J

def energy_conv(capability, input, output): # (KB -> B)
    return capability * (9.24e-7 * input * 1024 + 1.874e-6 * output * 1024 + 3.81e-2)

def energy_relu(capability, input):
    return capability * (1.435e-6 * input * 1024 + 2.881e-1)

def energy_pool(capability, input, output):
    return capability * (1.41e-6 * input * 1024 + 1.312e-7 * output * 1024 + 3.572e-1)

def energy_norm(capability, input):
    return capability * (5.187e-6 * input * 1024 + 5.991e-1)

def energy_fc(capability, input, output):
    return capability * (9.213e-6 * input * 1024 + 4.012e-5 * output * 1024 + 1.125)


# network energy consumption

# AlexNet
def alexnet_energy(comp):
    alex_block_energy = [energy_conv(comp, 37.151, 265.72) + energy_relu(comp, 265.72),
                         energy_pool(comp, 327.255, 63.045) + energy_norm(comp, 63.045),
                         energy_conv(comp, 62.751, 158.56) + energy_relu(comp, 158.56),
                         energy_pool(comp, 199.864, 36.467) + energy_norm(comp, 36.467),
                         energy_conv(comp, 36.508, 43.472) + energy_relu(comp, 43.472),
                         energy_conv(comp, 54.248, 43.118) + energy_relu(comp, 43.118),
                         energy_conv(comp, 54.685, 29.877) + energy_relu(comp, 29.877),
                         energy_pool(comp, 36.999, 4.231),
                         energy_fc(comp, 4.231, 51.429),
                         energy_fc(comp, 64.688, 50.178) + energy_fc(comp, 64.362, 0.322)]

    alex_block_energy = np.array(alex_block_energy)
    alex_block_energy[alex_block_energy < 0] = 1
    return alex_block_energy
    # return tool.norm_array(alex_block_energy)

# VGGNet-16
def vgg16_energy(comp):
    vgg_block_energy = [energy_conv(comp, 37.151, 1.629) + energy_relu(comp, 1.629),
                        energy_conv(comp, 0.303, 3.691) + energy_relu(comp, 3.691) + energy_norm(comp, 0.309) + energy_pool(comp, 7.567, 2.458),
                        energy_conv(comp, 2.458, 1.907) + energy_relu(comp, 1.907),
                        energy_conv(comp, 0.277, 2.479) + energy_relu(comp, 2.479) + energy_norm(comp, 0.27) + energy_pool(comp, 4.884, 0.909),
                        energy_conv(comp, 0.909, 1.967) + energy_relu(comp, 1.967),
                        energy_conv(comp, 0.164, 3.372) + energy_relu(comp, 3.372),
                        energy_conv(comp, 0.226, 2.836) + energy_relu(comp, 2.836) + energy_norm(comp, 0.22) + energy_pool(comp, 2.107, 0.556),
                        energy_conv(comp, 0.556, 4.479) + energy_relu(comp, 4.479),
                        energy_conv(comp, 0.199, 7.078) + energy_relu(comp, 7.078),
                        energy_conv(comp, 0.172, 8.234) + energy_relu(comp, 8.234) + energy_norm(comp, 0.187) + energy_pool(comp, 1.054, 0.791),
                        energy_conv(comp, 0.791, 0.795) + energy_relu(comp, 0.795),
                        energy_conv(comp, 0.156, 12.045) + energy_relu(comp, 12.045),
                        energy_conv(comp, 0.201, 7.645) + energy_relu(comp, 7.645) + energy_norm(comp, 0.201) + energy_pool(comp, 0.555, 0.488),
                        energy_fc(comp, 0.488, 161.9) + energy_relu(comp, 161.9),
                        energy_fc(comp, 0.252, 1005.47) + energy_relu(comp, 1005.47) + energy_fc(comp, 0.811, 3.516)]

    vgg_block_energy = np.array(vgg_block_energy)
    vgg_block_energy[vgg_block_energy < 0] = 1
    return vgg_block_energy
    # return tool.norm_array(alex_block_energy)

# ResNet-32
def res32_energy(comp):
    res_block_energy = [energy_conv(comp, 37.151, 159.028) + energy_norm(comp, 159.028) + energy_relu(comp, 161.205),
                        energy_conv(comp, 180.582, 172.676) + energy_norm(comp, 172.676) + energy_conv(comp, 171.616, 163.914) + energy_norm(comp, 163.914),
                        energy_conv(comp, 165.25, 170.833) + energy_norm(comp, 170.833) + energy_conv(comp, 169.402, 164.717) + energy_norm(comp, 164.717),
                        energy_conv(comp, 165.222, 155.006) + energy_norm(comp, 155.006) + energy_conv(comp, 156.889, 166.765) + energy_norm(comp, 166.765),
                        energy_conv(comp, 165.575, 156.38) + energy_norm(comp, 156.38) + energy_conv(comp, 161.414, 167) + energy_norm(comp, 167),
                        energy_conv(comp, 171.773, 163.601) + energy_norm(comp, 163.601) + energy_conv(comp, 154.215, 160.697) + energy_norm(comp, 160.697),
                        energy_conv(comp, 159.431, 337.696) + energy_norm(comp, 337.696) + energy_conv(comp, 324.8, 323.249) + energy_norm(comp, 323.249),
                        energy_conv(comp, 326.306, 81.645) + energy_norm(comp, 81.645) + energy_conv(comp, 76.563, 80.02) + energy_norm(comp, 80.02),
                        energy_conv(comp, 81.152, 24.53) + energy_norm(comp, 24.53) + energy_conv(comp, 24.658, 23.358) + energy_norm(comp, 23.358),
                        energy_conv(comp, 23.862, 6.155) + energy_norm(comp, 6.155) + energy_conv(comp, 6.033, 5.853) + energy_norm(comp, 5.853),
                        energy_conv(comp, 5.849, 1.636) + energy_norm(comp, 1.636) + energy_conv(comp, 1.695, 1.732) + energy_norm(comp, 1.732),
                        energy_conv(comp, 1.692, 3.023) + energy_norm(comp, 3.023) + energy_conv(comp, 3.139, 3.162) + energy_norm(comp, 3.162),
                        energy_conv(comp, 2.991, 0.978) + energy_norm(comp, 0.978) + energy_conv(comp, 0.83, 0.927) + energy_norm(comp, 0.927),
                        energy_conv(comp, 0.882, 0.978) + energy_norm(comp, 0.978) + energy_conv(comp, 0.979, 0.986) + energy_norm(comp, 0.986),
                        energy_conv(comp, 0.925, 0.863) + energy_norm(comp, 0.863) + energy_conv(comp, 0.892, 0.951) + energy_norm(comp, 0.951),
                        energy_conv(comp, 0.954, 0.909) + energy_norm(comp, 0.909) + energy_conv(comp, 0.956, 0.929) + energy_norm(comp, 0.929),
                        energy_fc(comp, 0.94, 0.327)]

    res_block_energy = np.array(res_block_energy)
    res_block_energy[res_block_energy < 0] = 1
    return res_block_energy
    # return tool.norm_array(alex_block_energy)

# MobileNet-V1
def MobileNetV1_ConvBN_energy_T(comp, layer_1_input, layer_2_input, layer_3_input):
    return energy_conv(comp, layer_1_input, layer_2_input) + energy_norm(comp, layer_2_input) + energy_relu(comp, layer_3_input)

def MobileNetV1_ConvDW_energy_T(comp, layer_1_input, layer_2_input, layer_3_input, layer_4_input, layer_5_input, layer_6_input):
    ConvDW_1 = energy_conv(comp, layer_1_input, layer_2_input) + energy_norm(comp, layer_2_input) + energy_relu(comp, layer_3_input)
    ConvDW_2 = energy_conv(comp, layer_4_input, layer_5_input) + energy_norm(comp, layer_5_input) + energy_relu(comp, layer_6_input)
    return ConvDW_1 + ConvDW_2

def mobilenet_energy(comp):
    mobile_block_energy = [MobileNetV1_ConvBN_energy_T(comp, 37.151, 360.819, 358.695) + MobileNetV1_ConvDW_energy_T(comp, 400.633, 353.338, 369.152, 415.551, 710.224, 721.285),
                           MobileNetV1_ConvDW_energy_T(comp, 849.531, 727.631, 714.656, 823.222, 1418.259, 1435.540),
                           MobileNetV1_ConvDW_energy_T(comp, 1646.346, 1445.826, 1442.646, 1636.792, 1410.675, 1405.317) + MobileNetV1_ConvDW_energy_T(comp, 1714.335, 357.550, 363.434, 410.188, 731.600, 712.675),
                           MobileNetV1_ConvDW_energy_T(comp, 838.411, 715.715, 719.103, 829.064, 717.434, 718.336) + MobileNetV1_ConvDW_energy_T(comp, 837.796, 178.101, 179.481, 209.119, 359.812, 360.391),
                           MobileNetV1_ConvDW_energy_T(comp, 406.291, 357.623, 360.171, 407.018, 361.224, 357.101) + MobileNetV1_ConvDW_energy_T(comp, 416.107, 360.842, 355.682, 407.582, 357.030, 358.636) + MobileNetV1_ConvDW_energy_T(comp, 410.672, 359.303, 360.202, 410.496, 359.222, 357.455) + MobileNetV1_ConvDW_energy_T(comp, 416.839, 358.109, 358.989, 407.486, 358.522, 359.605) + MobileNetV1_ConvDW_energy_T(comp, 415.415, 357.375, 359.892, 407.386, 358.854, 358.981),
                           MobileNetV1_ConvDW_energy_T(comp, 413.797, 90.327, 90.939, 102.085, 180.121, 179.563) + MobileNetV1_ConvDW_energy_T(comp, 207.440, 44.696, 45.488, 51.513, 45.151, 44.788) + energy_fc(comp, 51.716, 0.316)]

    mobile_block_energy = np.array(mobile_block_energy)
    mobile_block_energy[mobile_block_energy < 0] = 1
    return mobile_block_energy
    # return tool.norm_array(mobile_block_energy)


def alexnet_energy_T():
    return alexnet_energy(ncp.C_T)

def alexnet_energy_H_h():
    return alexnet_energy(ncp.C_H_h)

def alexnet_energy_H_l():
    return alexnet_energy(ncp.C_H_l)


def vgg16_energy_T():
    return vgg16_energy(ncp.C_T)

def vgg16_energy_H_h():
    return vgg16_energy(ncp.C_H_h)

def vgg16_energy_H_l():
    return vgg16_energy(ncp.C_H_l)


def res32_energy_T():
    return res32_energy(ncp.C_T)

def res32_energy_H_h():
    return res32_energy(ncp.C_H_h)

def res32_energy_H_l():
    return res32_energy(ncp.C_H_l)


def mobilenet_energy_T():
    return mobilenet_energy(ncp.C_T)

def mobilenet_energy_H_h():
    return mobilenet_energy(ncp.C_H_h)

def mobilenet_energy_H_l():
    return mobilenet_energy(ncp.C_H_l)
