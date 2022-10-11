import matplotlib.pyplot as plt
import numpy as np

image = np.array([i*0.2 for i in range(-10, 10)])
print(image.shape)
plt.imshow(image.reshape(1, 20))
plt.show()