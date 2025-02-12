import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Load the image
image = cv2.imread("F:\programming\computer_vision_nanodegree\projects\Store_heatmap\images\store.jpg")
# convert the image from BGR ot RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define selected Regions
regions = {
    "sitting" : ((1517,313) , (1633,708) , 30),
    "entrance": ((1052,832), (1316,991), 45),
    "Ailse": ((716,193) , (909,639), 120),
    "checkout": ((220, 728) , (646, 844), 60)
}

# Create a blank heatmap 
heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

# Define four rectangular areas with intensity values
# areas = [
#     ((332, 281), (530, 370), 45), # Entrance  # (top-left corner), (bottom-right corner), intensity
#     ((472, 191), (790, 247), 120), # Ailse
#     ((6, 390), (367, 571), 30), # sitting
#     ((653, 253), (742, 375), 60),
# ]




# Assign intensity values to the rectangular regions
for (top_left, bottom_right, intensity) in regions.values():
    x1, y1 = top_left
    x2, y2 = bottom_right
    heatmap[y1:y2, x1:x2] = intensity  # Assign the intensity inside the rectangle

# Smooth the heatmap using Gaussian blur
heatmap = gaussian_filter(heatmap, sigma=50)

# Normalize the heatmap (convert to 0-255)
heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
heatmap = np.uint8(heatmap)

# Apply a colormap 
heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

# overlay the heatmap onto the image
alpha = 0.5  # Transparency factor
overlay = cv2.addWeighted(heatmap_colored, alpha, image, 1 - alpha, 0)

# Show the final image
plt.figure(figsize=(10, 5))
plt.imshow(overlay)
plt.axis("off")
plt.savefig("outputs/store.jpg")
plt.show()