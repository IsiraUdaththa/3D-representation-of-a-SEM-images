import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage as ndi

# Load the image using Pillow
image_path = "img/test.tif"
image = Image.open(image_path).convert('L')  # Convert to grayscale

# Convert the image to a NumPy array
image_array = np.array(image)

# Define the size of the region to calculate the mean depth
region_size = 10  # Adjust this value to change the region size

# Use a Gaussian filter to smooth the image (optional)
smoothed_image = ndi.gaussian_filter(image_array, sigma=1)  # Adjust sigma as needed


# Calculate the mean depth in each region
def calculate_mean_depth(image, region_size):
    height, width = image.shape
    mean_depth = np.zeros((height, width))

    for y in range(0, height, region_size):
        for x in range(0, width, region_size):
            region = image[y:y + region_size, x:x + region_size]
            mean = np.mean(region)
            mean_depth[y:y + region_size, x:x + region_size] = mean

    return mean_depth


mean_depth = calculate_mean_depth(smoothed_image, region_size)

# Create a meshgrid based on the image dimensions
x, y = np.meshgrid(np.arange(image_array.shape[1]), np.arange(image_array.shape[0]))

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the 3D depth surface plot with colormap
cmap = plt.get_cmap('viridis')  # Choose your preferred colormap
surface = ax.plot_surface(x, y, mean_depth, cmap=cmap, rstride=1, cstride=1, linewidth=0, antialiased=False)

# Add colorbar
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Depth')
ax.set_title('3D Depth Surface Map from Black and White Image')

# Show the plot
plt.show()
