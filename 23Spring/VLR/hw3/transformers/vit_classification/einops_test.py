import matplotlib.pyplot as plt
import numpy as np
# import einops
from einops import rearrange, reduce, repeat


image = np.random.randn(30, 40)

# change it to RGB format by repeating in each channel
img_repeat = repeat(image, 'h w -> h w c', c=3).shape
# (30, 40, 3)
