def HW_out(N, C_in, H_in, W_in, kernel_size, padding=[0, 0], stride=[1, 1], dilation=[1, 1]):
    H_out = (H_in + 2*padding[0] - dilation[0] * (kernel_size[0]-1)-1)/stride[0] + 1
    W_out = (W_in + 2*padding[1] - dilation[1] * (kernel_size[1]-1)-1)/stride[1] + 1
    return H_out, W_out


c1 = HW_out(32)