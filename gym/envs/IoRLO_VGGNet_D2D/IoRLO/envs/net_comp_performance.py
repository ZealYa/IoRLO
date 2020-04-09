# computing capability performance

# MEC-based computing scenario
C_T = 1    # 2.4 GHZ (2400 mHZ, CPU=4)
C_E = 40.0 # 2.3 GHZ (2300 mHZ, CPU=16)
C_C = 10.0 # 2.0 GHz (2000 mHZ, CPU=56)

# D2D-based computing scenario
C_U = 1   # 2.4 GHZ (2400 mHZ, CPU=1)
C_H = 1.5
# For the two scenarios
C_H_h = (2.4 * 4) / (2.8 * 4 + 1.8 * 4)
C_H_l = (2.4 * 4) / (2.4 * 4 * 0.6)

# network performance

# MEC-based communication scenario
B_TE_U = 76.1 * 1024 / 8 # (KB/s)
B_TE_D = 382.4 * 1024 / 8
B_EC_U = 73.3 * 1024 / 8
B_EC_D = 542.3 * 1024 / 8
L_TE = 8.76 / 1000 # (s)
L_EC = 27.04 / 1000

# D2D-based communication scenario
B_D2D = 87.6 * 1024 / 8 # (KB/s)
L_D2D = 23.14 # (ms)

# Energy consumption for data transmission
a_u_5G = 65 # (mW/Mbps)
a_d_5G = 6.5
b_5G = 11475.97 # (mW)

a_d2d = 283.17
b_d2d = 132.86

def energy_u_TE_5G():
    return (a_u_5G * B_TE_U * 8 / 1024 + b_5G) / 1000 # 1 W = 1 J/s

def energy_d_TE_5G():
    return (a_d_5G * B_TE_D * 8 / 1024 + b_5G) / 1000

def energy_u_EC_5G():
    return (a_u_5G * B_EC_U * 8 / 1024 + b_5G) / 1000

def energy_d_EC_5G():
    return (a_d_5G * B_EC_D * 8 / 1024 + b_5G) / 1000

def energy_d2d():
    return (a_d2d * B_D2D * 8 / 1024 + b_d2d) / 1000


# DNN network performance (KB)

# 37.151 is the image size

# AlexNet
alexnet_input = [37.151, 327.255, 62.751, 199.864, 36.508, 54.248, 54.685, 36.999, 4.231, 51.429]
alexnet_output = 0.322

# VGGNet-16
vgg16_input = [37.151, 1011.715, 258.466, 469.883, 96.573, 211.345, 213.85, 54.752, 131.096, 117.655, 27.559, 25.257, 22.865, 6.697, 59.107]
vgg16_output = 0.328

# ResNet-32
res32_input = [37.151, 180.582, 165.25, 165.222, 165.575, 171.773, 159.431, 326.306, 81.152, 23.862, 5.849, 1.692, 2.991, 0.882, 0.925, 0.954, 0.94]
res32_output = 0.327

# MobileNet-V1
mobilenet_input = [37.151, 816.354, 1656.833, 816.989, 406.454, 414.231]
mobilenet_output = 0.404
