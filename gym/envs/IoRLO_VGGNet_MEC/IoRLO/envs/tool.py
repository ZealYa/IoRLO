import numpy as np
from . import net_comp_performance as ncp


def norm_array(array):
    min, max = array.min(), array.max()
    return (array - min) / (max - min)


def apply_action_MEC(action, n_out, thresa, thresb):
    """
    :param action: the offloading decision
    :param n_out: the dimension of the out put
    :param thresa:
    :param thresb:
    :return: from sigmoid (0,1) to int according to the Intent requirement from the third-party
    """
    for i in range(n_out):
        if action[i] < thresa:
            action[i] = 1
        elif action[i] > thresb:
            action[i] = 3
        else:
            action[i] = 2


def apply_action_D2D(action, n_out, thresc):
    for i in range(n_out):
        if action[i] < thresc:
            action[i] = 0
        else:
            action[i] = 1


def comm_cost_alexnet_MEC(action, n_out):
    per_block_r_l = np.zeros(n_out) # per block latency for receiving input (for T & E & C) (s)
    per_block_r_e = np.zeros(n_out) # per block energy consumption for receiving input (only for T)

    for block in range(n_out):

        if action[block] == 1:
            if block == 0 or action[block - 1] == 1:
                per_block_r_l[block] = 0
                per_block_r_e[block] = 0
            elif action[block - 1] == 2:
                per_block_r_l[block] = ncp.alexnet_input[block] / ncp.B_TE_D + ncp.L_TE / 2 # (s)
                per_block_r_e[block] = per_block_r_l[block] * ncp.energy_d_TE_5G() # (J)
            else: # i.e., action[block-1] == 3
                per_block_r_l[block] = ncp.alexnet_input[block] / ncp.B_TE_D + ncp.L_TE / 2 + ncp.alexnet_input[block] / ncp.B_EC_D + ncp.L_EC / 2
                per_block_r_e[block] = (ncp.alexnet_input[block] / ncp.B_TE_D + ncp.L_TE / 2) * ncp.energy_d_TE_5G() + (ncp.alexnet_input[block] / ncp.B_EC_D + ncp.L_EC / 2) * ncp.energy_d_EC_5G()

        elif action[block] == 2:
            if action[block-1] == 2:
                per_block_r_l[block] = 0
            elif block == 0 or action[block-1] == 1:
                per_block_r_l[block] = ncp.alexnet_input[block] / ncp.B_TE_U + ncp.L_TE / 2
            else: # i.e., action[block-1] == 3
                per_block_r_l[block] = ncp.alexnet_input[block] / ncp.B_EC_D + ncp.L_EC / 2

        else: # action[block] == 3
            if block == 0 or action[block-1] == 1:
                per_block_r_l[block] = ncp.alexnet_input[block] / ncp.B_TE_U + ncp.L_TE / 2 + ncp.alexnet_input[block] / ncp.B_EC_U + ncp.L_EC / 2
            elif action[block-1] == 2:
                per_block_r_l[block] = ncp.alexnet_input[block] / ncp.B_EC_U + ncp.L_EC / 2
            else: # action[block-1] = 3
                per_block_r_l[block] = 0

    per_block_s_l = np.zeros(n_out) # per block latency for sending output (only for T)
    per_block_s_e = np.zeros(n_out) # per block energy consumption for sending output (only for T)

    for block in range(n_out-1):
        if action[block] == 1:
            if block == n_out-1:
                per_block_s_l[block] = 0
                per_block_s_e[block] = 0
            if action[block+1] == 1:
                per_block_s_l[block] = 0
                per_block_s_e[block] = 0
            elif action[block+1] == 2:
                per_block_s_l[block] = ncp.alexnet_input[block+1] / ncp.B_TE_U + ncp.L_TE / 2
                per_block_s_e[block] = per_block_s_l[block] * ncp.energy_u_TE_5G()
            else: # action[block+1] == 3:
                per_block_s_l[block] = ncp.alexnet_input[block+1] / ncp.B_TE_U + ncp.L_TE / 2 + ncp.alexnet_input[block+1] / ncp.B_EC_U + ncp.L_EC / 2
                per_block_s_e[block] = (ncp.alexnet_input[block+1] / ncp.B_TE_U + ncp.L_TE / 2) * ncp.energy_u_TE_5G() + (ncp.alexnet_input[block+1] / ncp.B_EC_U + ncp.L_EC / 2) * ncp.energy_u_EC_5G()
        # else:
        #     print('edge and cloud will not consume energy of mobile device!')

    # receive result form T or E or C to T
    # i.e., block = n_out
    # receive_result_l  # the latency for mobile device receive result
    # receive_result_e  # the energy consumption for mobile device receive result

    if action[n_out-1] == 1:
        receive_result_l = 0
        receive_result_e = 0
    elif action[n_out-1] == 2:
        receive_result_l = ncp.alexnet_output / ncp.B_TE_D + ncp.L_TE / 2
        receive_result_e = receive_result_l * ncp.energy_d_TE_5G()
    else: # action[n_out]  == 3
        receive_result_l = ncp.alexnet_output / ncp.B_TE_D + ncp.L_TE / 2 + ncp.alexnet_output / ncp.B_EC_D + ncp.L_EC / 2
        receive_result_e = (ncp.alexnet_output / ncp.B_TE_D + ncp.L_TE / 2) * ncp.energy_d_TE_5G() + (ncp.alexnet_output / ncp.B_EC_D + ncp.L_EC / 2) * ncp.energy_d_EC_5G()

    # np.sum(per_block_s_l) has been sumed in comm_cost()
    return np.sum(per_block_r_l) + np.sum(receive_result_l), np.sum(per_block_r_e) + np.sum(per_block_s_e) + np.sum(receive_result_e)


def comm_cost_vgg16_MEC(action, n_out):
    per_block_r_l = np.zeros(n_out) # per block latency for receiving input (for T & E & C) (s)
    per_block_r_e = np.zeros(n_out) # per block energy consumption for receiving input (only for T)

    for block in range(n_out):

        if action[block] == 1:
            if block == 0 or action[block - 1] == 1:
                per_block_r_l[block] = 0
                per_block_r_e[block] = 0
            elif action[block - 1] == 2:
                per_block_r_l[block] = ncp.vgg16_input[block] / ncp.B_TE_D + ncp.L_TE / 2 # (s)
                per_block_r_e[block] = per_block_r_l[block] * ncp.energy_d_TE_5G() # (J)
            else: # i.e., action[block-1] == 3
                per_block_r_l[block] = ncp.vgg16_input[block] / ncp.B_TE_D + ncp.L_TE / 2 + ncp.vgg16_input[block] / ncp.B_EC_D + ncp.L_EC / 2
                per_block_r_e[block] = (ncp.vgg16_input[block] / ncp.B_TE_D + ncp.L_TE / 2) * ncp.energy_d_TE_5G() + (ncp.vgg16_input[block] / ncp.B_EC_D + ncp.L_EC / 2) * ncp.energy_d_EC_5G()

        elif action[block] == 2:
            if action[block-1] == 2:
                per_block_r_l[block] = 0
            elif block == 0 or action[block-1] == 1:
                per_block_r_l[block] = ncp.vgg16_input[block] / ncp.B_TE_U + ncp.L_TE / 2
            else: # i.e., action[block-1] == 3
                per_block_r_l[block] = ncp.vgg16_input[block] / ncp.B_EC_D + ncp.L_EC / 2

        else: # action[block] == 3
            if block == 0 or action[block-1] == 1:
                per_block_r_l[block] = ncp.vgg16_input[block] / ncp.B_TE_U + ncp.L_TE / 2 + ncp.vgg16_input[block] / ncp.B_EC_U + ncp.L_EC / 2
            elif action[block-1] == 2:
                per_block_r_l[block] = ncp.vgg16_input[block] / ncp.B_EC_U + ncp.L_EC / 2
            else: # action[block-1] = 3
                per_block_r_l[block] = 0

    per_block_s_l = np.zeros(n_out) # per block latency for sending output (only for T)
    per_block_s_e = np.zeros(n_out) # per block energy consumption for sending output (only for T)

    for block in range(n_out-1):
        if action[block] == 1:
            if block == n_out-1:
                per_block_s_l[block] = 0
                per_block_s_e[block] = 0
            if action[block+1] == 1:
                per_block_s_l[block] = 0
                per_block_s_e[block] = 0
            elif action[block+1] == 2:
                per_block_s_l[block] = ncp.vgg16_input[block+1] / ncp.B_TE_U + ncp.L_TE / 2
                per_block_s_e[block] = per_block_s_l[block] * ncp.energy_u_TE_5G()
            else: # action[block+1] == 3:
                per_block_s_l[block] = ncp.vgg16_input[block+1] / ncp.B_TE_U + ncp.L_TE / 2 + ncp.vgg16_input[block+1] / ncp.B_EC_U + ncp.L_EC / 2
                per_block_s_e[block] = (ncp.vgg16_input[block+1] / ncp.B_TE_U + ncp.L_TE / 2) * ncp.energy_u_TE_5G() + (ncp.vgg16_input[block+1] / ncp.B_EC_U + ncp.L_EC / 2) * ncp.energy_u_EC_5G()
        # else:
        #     print('edge and cloud will not consume energy of mobile device!')

    # receive result form T or E or C to T
    # i.e., block = n_out
    # receive_result_l  # the latency for mobile device receive result
    # receive_result_e  # the energy consumption for mobile device receive result

    if action[n_out-1] == 1:
        receive_result_l = 0
        receive_result_e = 0
    elif action[n_out-1] == 2:
        receive_result_l = ncp.vgg16_output / ncp.B_TE_D + ncp.L_TE / 2
        receive_result_e = receive_result_l * ncp.energy_d_TE_5G()
    else: # action[n_out]  == 3
        receive_result_l = ncp.vgg16_output / ncp.B_TE_D + ncp.L_TE / 2 + ncp.vgg16_output / ncp.B_EC_D + ncp.L_EC / 2
        receive_result_e = (ncp.vgg16_output / ncp.B_TE_D + ncp.L_TE / 2) * ncp.energy_d_TE_5G() + (ncp.vgg16_output / ncp.B_EC_D + ncp.L_EC / 2) * ncp.energy_d_EC_5G()

    # np.sum(per_block_s_l) has been sumed in comm_cost()
    return np.sum(per_block_r_l) + np.sum(receive_result_l), np.sum(per_block_r_e) + np.sum(per_block_s_e) + np.sum(receive_result_e)


def comm_cost_res32_MEC(action, n_out):
    per_block_r_l = np.zeros(n_out) # per block latency for receiving input (for T & E & C) (time is s)
    per_block_r_e = np.zeros(n_out) # per block energy consumption for receiving input (only for T)

    for block in range(n_out):

        if action[block] == 1:
            if block == 0 or action[block - 1] == 1:
                per_block_r_l[block] = 0
                per_block_r_e[block] = 0
            elif action[block - 1] == 2:
                per_block_r_l[block] = ncp.res32_input[block] / ncp.B_TE_D + ncp.L_TE / 2 # (s)
                per_block_r_e[block] = per_block_r_l[block] * ncp.energy_d_TE_5G() # (J)
            else: # i.e., action[block-1] == 3
                per_block_r_l[block] = ncp.res32_input[block] / ncp.B_TE_D + ncp.L_TE / 2 + ncp.res32_input[block] / ncp.B_EC_D + ncp.L_EC / 2
                per_block_r_e[block] = (ncp.res32_input[block] / ncp.B_TE_D + ncp.L_TE / 2) * ncp.energy_d_TE_5G() + (ncp.res32_input[block] / ncp.B_EC_D + ncp.L_EC / 2) * ncp.energy_d_EC_5G()

        elif action[block] == 2:
            if action[block-1] == 2:
                per_block_r_l[block] = 0
            elif block == 0 or action[block-1] == 1:
                per_block_r_l[block] = ncp.res32_input[block] / ncp.B_TE_U + ncp.L_TE / 2
            else: # i.e., action[block-1] == 3
                per_block_r_l[block] = ncp.res32_input[block] / ncp.B_EC_D + ncp.L_EC / 2

        else: # action[block] == 3
            if block == 0 or action[block-1] == 1:
                per_block_r_l[block] = ncp.res32_input[block] / ncp.B_TE_U + ncp.L_TE / 2 + ncp.res32_input[block] / ncp.B_EC_U + ncp.L_EC / 2
            elif action[block-1] == 2:
                per_block_r_l[block] = ncp.res32_input[block] / ncp.B_EC_U + ncp.L_EC / 2
            else:  # action[block-1] = 3
                per_block_r_l[block] = 0

    per_block_s_l = np.zeros(n_out) # per block latency for sending output (only for T)
    per_block_s_e = np.zeros(n_out) # per block energy consumption for sending output (only for T)

    for block in range(n_out-1):
        if action[block] == 1:
            if block == n_out-1:
                per_block_s_l[block] = 0
                per_block_s_e[block] = 0
            if action[block+1] == 1:
                per_block_s_l[block] = 0
                per_block_s_e[block] = 0
            elif action[block+1] == 2:
                per_block_s_l[block] = ncp.res32_input[block+1] / ncp.B_TE_U + ncp.L_TE / 2
                per_block_s_e[block] = per_block_s_l[block] * ncp.energy_u_TE_5G()
            else: # action[block+1] == 3:
                per_block_s_l[block] = ncp.res32_input[block+1] / ncp.B_TE_U + ncp.L_TE / 2 + ncp.res32_input[block+1] / ncp.B_EC_U + ncp.L_EC / 2
                per_block_s_e[block] = (ncp.res32_input[block+1] / ncp.B_TE_U + ncp.L_TE / 2) * ncp.energy_u_TE_5G() + (ncp.res32_input[block+1] / ncp.B_EC_U + ncp.L_EC / 2) * ncp.energy_u_EC_5G()
        # else:
        #     print('edge and cloud will not consume energy of mobile device!')

    # receive result form T or E or C to T
    # i.e., block = n_out
    # receive_result_l  # the latency for mobile device receive result
    # receive_result_e  # the energy consumption for mobile device receive result

    if action[n_out-1] == 1:
        receive_result_l = 0
        receive_result_e = 0
    elif action[n_out-1] == 2:
        receive_result_l = ncp.res32_output / ncp.B_TE_D + ncp.L_TE / 2
        receive_result_e = receive_result_l * ncp.energy_d_TE_5G()
    else: # action[n_out]  == 3
        receive_result_l = ncp.res32_output / ncp.B_TE_D + ncp.L_TE / 2 + ncp.res32_output / ncp.B_EC_D + ncp.L_EC / 2
        receive_result_e = (ncp.res32_output / ncp.B_TE_D + ncp.L_TE / 2) * ncp.energy_d_TE_5G() + (ncp.res32_output / ncp.B_EC_D + ncp.L_EC / 2) * ncp.energy_d_EC_5G()

    # np.sum(per_block_s_l) has been sumed in comm_cost()
    return np.sum(per_block_r_l) + np.sum(receive_result_l), np.sum(per_block_r_e) + np.sum(per_block_s_e) + np.sum(receive_result_e)


def comm_cost_mobilenet_MEC(action, n_out):
    per_block_r_l = np.zeros(n_out)  # per block latency for receiving input (for T & E & C) # (s)
    per_block_r_e = np.zeros(n_out)  # per block energy consumption for receiving input (only for T)

    for block in range(n_out):

        if action[block] == 1:
            if block == 0 or action[block - 1] == 1:
                per_block_r_l[block] = 0
                per_block_r_e[block] = 0
            elif action[block - 1] == 2:
                per_block_r_l[block] = ncp.mobilenet_input[block] / ncp.B_TE_D + ncp.L_TE / 2 # (s)
                per_block_r_e[block] = per_block_r_l[block] * ncp.energy_d_TE_5G() # (J)
            else: # i.e., action[block-1] == 3
                per_block_r_l[block] = ncp.mobilenet_input[block] / ncp.B_TE_D + ncp.L_TE / 2 + ncp.mobilenet_input[block] / ncp.B_EC_D + ncp.L_EC / 2
                per_block_r_e[block] = (ncp.mobilenet_input[block] / ncp.B_TE_D + ncp.L_TE / 2) * ncp.energy_d_TE_5G() + (ncp.mobilenet_input[block] / ncp.B_EC_D + ncp.L_EC / 2) * ncp.energy_d_EC_5G()

        elif action[block] == 2:
            if action[block-1] == 2:
                per_block_r_l[block] = 0
            elif block == 0 or action[block-1] == 1:
                per_block_r_l[block] = ncp.mobilenet_input[block] / ncp.B_TE_U + ncp.L_TE / 2
            else: # i.e., action[block-1] == 3
                per_block_r_l[block] = ncp.mobilenet_input[block] / ncp.B_EC_D + ncp.L_EC / 2

        else:  # action[block] == 3
            if block == 0 or action[block-1] == 1:
                per_block_r_l[block] = ncp.mobilenet_input[block] / ncp.B_TE_U + ncp.L_TE / 2 + ncp.mobilenet_input[block] / ncp.B_EC_U + ncp.L_EC / 2
            elif action[block-1] == 2:
                per_block_r_l[block] = ncp.mobilenet_input[block] / ncp.B_EC_U + ncp.L_EC / 2
            else: # action[block-1] = 3
                per_block_r_l[block] = 0

    per_block_s_l = np.zeros(n_out) # per block latency for sending output (only for T)
    per_block_s_e = np.zeros(n_out) # per block energy consumption for sending output (only for T)

    for block in range(n_out-1):
        if action[block] == 1:
            if block == n_out-1:
                per_block_s_l[block] = 0
                per_block_s_e[block] = 0
            if action[block+1] == 1:
                per_block_s_l[block] = 0
                per_block_s_e[block] = 0
            elif action[block+1] == 2:
                per_block_s_l[block] = ncp.mobilenet_input[block+1] / ncp.B_TE_U + ncp.L_TE / 2
                per_block_s_e[block] = per_block_s_l[block] * ncp.energy_u_TE_5G()
            else: # action[block+1] == 3:
                per_block_s_l[block] = ncp.mobilenet_input[block+1] / ncp.B_TE_U + ncp.L_TE / 2 + ncp.mobilenet_input[block+1] / ncp.B_EC_U + ncp.L_EC / 2
                per_block_s_e[block] = (ncp.mobilenet_input[block+1] / ncp.B_TE_U + ncp.L_TE / 2) * ncp.energy_u_TE_5G() + (ncp.mobilenet_input[block+1] / ncp.B_EC_U + ncp.L_EC / 2) * ncp.energy_u_EC_5G()
        # else:
        #     print('edge and cloud will not consume energy of mobile device!')

    # receive result form T or E or C to T
    # i.e., block = n_out
    # receive_result_l  # the latency for mobile device receive result
    # receive_result_e  # the energy consumption for mobile device receive result

    if action[n_out-1] == 1:
        receive_result_l = 0
        receive_result_e = 0
    elif action[n_out-1] == 2:
        receive_result_l = ncp.mobilenet_output / ncp.B_TE_D + ncp.L_TE / 2
        receive_result_e = receive_result_l * ncp.energy_d_TE_5G()
    else: # action[n_out]  == 3
        receive_result_l = ncp.mobilenet_output / ncp.B_TE_D + ncp.L_TE / 2 + ncp.mobilenet_output / ncp.B_EC_D + ncp.L_EC / 2
        receive_result_e = (ncp.mobilenet_output / ncp.B_TE_D + ncp.L_TE / 2) * ncp.energy_d_TE_5G() + (ncp.mobilenet_output / ncp.B_EC_D + ncp.L_EC / 2) * ncp.energy_d_EC_5G()

    # np.sum(per_block_s_l) has been sumed in comm_cost()
    return np.sum(per_block_r_l) + np.sum(receive_result_l), np.sum(per_block_r_e) + np.sum(per_block_s_e) + np.sum(receive_result_e)


def comm_cost_alexnet_D2D(action, n_out):
    per_block_r_l = np.zeros(n_out) # per block latency for receiving input (for T & H) (time is s)
    per_block_r_e = np.zeros(n_out) # per block energy consumption for receiving input (only for T)

    for block in range(n_out):

        if action[block] == 0:
            if block == 0 or action[block - 1] == 0:
                per_block_r_l[block] = 0
                per_block_r_e[block] = 0
            else:
                per_block_r_l[block] = ncp.alexnet_input[block] / ncp.B_D2D + ncp.L_D2D / 2  # s
                per_block_r_e[block] = per_block_r_l[block] * ncp.energy_d2d()  # (J)

        elif action[block] == 1:
            if action[block-1] == 1:
                per_block_r_l[block] = 0
            else: # action[block-1] == 0
                per_block_r_l[block] = ncp.alexnet_input[block] / ncp.B_D2D + ncp.L_D2D / 2  # s

    per_block_s_l = np.zeros(n_out) # per block latency for sending output (only for T)
    per_block_s_e = np.zeros(n_out) # per block energy consumption for sending output (only for T)

    for block in range(n_out-1):
        if action[block] == 0:
            if block == n_out-1:
                per_block_s_l[block] = 0
                per_block_s_e[block] = 0
            if action[block+1] == 0:
                per_block_s_l[block] = 0
                per_block_s_e[block] = 0
            else: # action[block+1] == 1:
                per_block_s_l[block] = ncp.alexnet_input[block+1] / ncp.B_D2D + ncp.L_D2D / 2  # s
                per_block_s_e[block] = per_block_s_l[block] * ncp.energy_d2d()

    # receive result form T or H to T
    # i.e., block = n_out
    # receive_result_l  # the latency for mobile device receive result
    # receive_result_e  # the energy consumption for mobile device receive result

    if action[n_out-1] == 0:
        receive_result_l = 0
        receive_result_e = 0
    else: # action[n_out-1] == 1:
        receive_result_l = ncp.alexnet_output / ncp.B_D2D + ncp.L_D2D / 2  # s
        receive_result_e = receive_result_l * ncp.energy_d2d()

    # np.sum(per_block_s_l) has been sumed in comm_cost()
    return np.sum(per_block_r_l) + np.sum(receive_result_l), np.sum(per_block_r_e) + np.sum(per_block_s_e) + np.sum(receive_result_e)


def comm_cost_vgg16_D2D(action, n_out):
    per_block_r_l = np.zeros(n_out) # per block latency for receiving input (for T & H) (s)
    per_block_r_e = np.zeros(n_out) # per block energy consumption for receiving input (only for T)

    for block in range(n_out):

        if action[block] == 0:
            if block == 0 or action[block - 1] == 0:
                per_block_r_l[block] = 0
                per_block_r_e[block] = 0
            else:
                per_block_r_l[block] = ncp.vgg16_input[block] / ncp.B_D2D + ncp.L_D2D / 2  # s
                per_block_r_e[block] = per_block_r_l[block] * ncp.energy_d2d()  # (J)

        elif action[block] == 1:
            if action[block-1] == 1:
                per_block_r_l[block] = 0
            else: # action[block-1] == 0
                per_block_r_l[block] = ncp.vgg16_input[block] / ncp.B_D2D + ncp.L_D2D / 2  # s

    per_block_s_l = np.zeros(n_out) # per block latency for sending output (only for T)
    per_block_s_e = np.zeros(n_out) # per block energy consumption for sending output (only for T)

    for block in range(n_out-1):
        if action[block] == 0:
            if block == n_out-1:
                per_block_s_l[block] = 0
                per_block_s_e[block] = 0
            if action[block+1] == 0:
                per_block_s_l[block] = 0
                per_block_s_e[block] = 0
            else: # action[block+1] == 1:
                per_block_s_l[block] = ncp.vgg16_input[block+1] / ncp.B_D2D + ncp.L_D2D / 2  # s
                per_block_s_e[block] = per_block_s_l[block] * ncp.energy_d2d()

    # receive result form T or H to T
    # i.e., block = n_out
    # receive_result_l  # the latency for mobile device receive result
    # receive_result_e  # the energy consumption for mobile device receive result

    if action[n_out-1] == 0:
        receive_result_l = 0
        receive_result_e = 0
    else: # action[n_out-1] == 1:
        receive_result_l = ncp.vgg16_output / ncp.B_D2D + ncp.L_D2D / 2  # s
        receive_result_e = receive_result_l * ncp.energy_d2d()

    # np.sum(per_block_s_l) has been sumed in comm_cost()
    return np.sum(per_block_r_l) + np.sum(receive_result_l), np.sum(per_block_r_e) + np.sum(per_block_s_e) + np.sum(receive_result_e)


def comm_cost_res32_D2D(action, n_out):
    per_block_r_l = np.zeros(n_out) # per block latency for receiving input (for T & H) (time is s)
    per_block_r_e = np.zeros(n_out) # per block energy consumption for receiving input (only for T)

    for block in range(n_out):

        if action[block] == 0:
            if block == 0 or action[block - 1] == 0:
                per_block_r_l[block] = 0
                per_block_r_e[block] = 0
            else:
                per_block_r_l[block] = ncp.res32_input[block] / ncp.B_D2D + ncp.L_D2D / 2  # s
                per_block_r_e[block] = per_block_r_l[block] * ncp.energy_d2d()  # (J)

        elif action[block] == 1:
            if action[block-1] == 1:
                per_block_r_l[block] = 0
            else: # action[block-1] == 0
                per_block_r_l[block] = ncp.res32_input[block] / ncp.B_D2D + ncp.L_D2D / 2  # s

    per_block_s_l = np.zeros(n_out) # per block latency for sending output (only for T)
    per_block_s_e = np.zeros(n_out) # per block energy consumption for sending output (only for T)

    for block in range(n_out-1):
        if action[block] == 0:
            if block == n_out-1:
                per_block_s_l[block] = 0
                per_block_s_e[block] = 0
            if action[block+1] == 0:
                per_block_s_l[block] = 0
                per_block_s_e[block] = 0
            else: # action[block+1] == 1:
                per_block_s_l[block] = ncp.res32_input[block+1] / ncp.B_D2D + ncp.L_D2D / 2  # s
                per_block_s_e[block] = per_block_s_l[block] * ncp.energy_d2d()

    # receive result form T or H to T
    # i.e., block = n_out
    # receive_result_l  # the latency for mobile device receive result
    # receive_result_e  # the energy consumption for mobile device receive result

    if action[n_out-1] == 0:
        receive_result_l = 0
        receive_result_e = 0
    else: # action[n_out-1] == 1:
        receive_result_l = ncp.res32_output / ncp.B_D2D + ncp.L_D2D / 2  # s
        receive_result_e = receive_result_l * ncp.energy_d2d()

    # np.sum(per_block_s_l) has been sumed in comm_cost()
    return np.sum(per_block_r_l) + np.sum(receive_result_l), np.sum(per_block_r_e) + np.sum(per_block_s_e) + np.sum(receive_result_e)


def comm_cost_mobilenet_D2D(action, n_out):
    per_block_r_l = np.zeros(n_out) # per block latency for receiving input (for T & H) (time is s)
    per_block_r_e = np.zeros(n_out) # per block energy consumption for receiving input (only for T)

    for block in range(n_out):

        if action[block] == 0:
            if block == 0 or action[block - 1] == 0:
                per_block_r_l[block] = 0
                per_block_r_e[block] = 0
            else:
                per_block_r_l[block] = ncp.mobilenet_input[block] / ncp.B_D2D + ncp.L_D2D / 2 # s
                per_block_r_e[block] = per_block_r_l[block] * ncp.energy_d2d() # (J)

        elif action[block] == 1:
            if action[block-1] == 1:
                per_block_r_l[block] = 0
            else: # action[block-1] == 0
                per_block_r_l[block] = ncp.mobilenet_input[block] / ncp.B_D2D + ncp.L_D2D / 2  # s

    per_block_s_l = np.zeros(n_out) # per block latency for sending output (only for T)
    per_block_s_e = np.zeros(n_out) # per block energy consumption for sending output (only for T)

    for block in range(n_out-1):
        if action[block] == 0:
            if block == n_out-1:
                per_block_s_l[block] = 0
                per_block_s_e[block] = 0
            if action[block+1] == 0:
                per_block_s_l[block] = 0
                per_block_s_e[block] = 0
            else: # action[block+1] == 1:
                per_block_s_l[block] = ncp.mobilenet_input[block+1] / ncp.B_D2D + ncp.L_D2D / 2  # s
                per_block_s_e[block] = per_block_s_l[block] * ncp.energy_d2d()

    # receive result form T or H to T
    # i.e., block = n_out
    # receive_result_l  # the latency for mobile device receive result
    # receive_result_e  # the energy consumption for mobile device receive result

    if action[n_out-1] == 0:
        receive_result_l = 0
        receive_result_e = 0
    else: # action[n_out-1] == 1:
        receive_result_l = ncp.mobilenet_output / ncp.B_D2D + ncp.L_D2D / 2  # s
        receive_result_e = receive_result_l * ncp.energy_d2d()

    # np.sum(per_block_s_l) has been sumed in comm_cost()
    return np.sum(per_block_r_l) + np.sum(receive_result_l), np.sum(per_block_r_e) + np.sum(per_block_s_e) + np.sum(receive_result_e)
