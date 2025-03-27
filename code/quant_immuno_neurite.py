import cv2
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import pandas as pd
from stardist.models import StarDist2D
from csbdeep.utils import normalize

# Load the multi-channel TIFF image
def load_tiff(filename):
	img = tiff.imread(filename)
	tubb3 = img[0]  # TUBB3 channel (neurites)
	islet1 = img[1]  # ISLET1 channel (specific nuclei)
	dapi = img[2]  # DAPI channel (nuclei)
	return dapi, islet1, tubb3

# Display raw channels
def display_raw_channels(dapi, islet1, tubb3):
	fig, axes = plt.subplots(1, 3, figsize=(15, 5))
	axes[0].imshow(dapi, cmap='gray')
	axes[0].set_title("DAPI (Nuclei)")
	axes[1].imshow(islet1, cmap='gray')
	axes[1].set_title("ISLET1 (Nuclei)")
	axes[2].imshow(tubb3, cmap='gray')
	axes[2].set_title("TUBB3 (Neurites)")
	for ax in axes:
		ax.axis("off")
	plt.show()

# Enhance contrast
def enhance_contrast(image): #design in order to improve visibility of detils
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	if image.dtype.itemsize == 1:
		enhanced = clahe.apply(image.astype(np.uint8))
	elif image.dtype.itemsize == 2:
		enhanced = clahe.apply(image.astype(np.uint16))
	else:
		return image
	return enhanced

# Segment nuclei using StarDist
def segment_nuclei_stardist(image):
	model = StarDist2D.from_pretrained("2D_versatile_fluo")
	image_norm = normalize(image, 1, 99.8)
	labels, _ = model.predict_instances(image_norm)
	return labels

# Filter small DAPI nuclei (remove debris)
def filter_dapi_nuclei(dapi_mask):
	dapi_regions = regionprops(dapi_mask)
	avg_dapi_size = np.mean([r.area for r in dapi_regions])
	min_nuclei_size = avg_dapi_size * 0.5  # Remove objects smaller than 50% of the average size

	filtered_mask = np.zeros_like(dapi_mask)
	for r in dapi_regions:
		if r.area >= min_nuclei_size:
			filtered_mask[r.coords[:, 0], r.coords[:, 1]] = dapi_mask[r.coords[:, 0], r.coords[:, 1]]

	return filtered_mask

# Filter ISLET1 nuclei based on DAPI nuclei size
def filter_islet1_by_size(islet1_mask, dapi_mask):
	dapi_regions = regionprops(dapi_mask)
	islet1_regions = regionprops(islet1_mask)

	avg_dapi_size = np.mean([r.area for r in dapi_regions])
	min_nuclei_size = avg_dapi_size * 0.5  # Allow only nuclei that are at least 50% of the DAPI size

	filtered_mask = np.zeros_like(islet1_mask)
	for r in islet1_regions:
		if r.area >= min_nuclei_size and np.any(dapi_mask[r.coords[:, 0], r.coords[:, 1]] > 0):
			filtered_mask[r.coords[:, 0], r.coords[:, 1]] = islet1_mask[r.coords[:, 0], r.coords[:, 1]]

	return filtered_mask

# Segment ISLET1-positive nuclei using StarDist
def segment_islet1_stardist(islet1, dapi_mask):
	islet1_enhanced = enhance_contrast(islet1)
	islet1_mask = segment_nuclei_stardist(islet1_enhanced)
	return filter_islet1_by_size(islet1_mask, dapi_mask)

# Segment neurites using GaussianBlur on X ex. TUBB3 staining
def segment_neurites(tubb3):
	nbytes = tubb3.dtype.itemsize # number of bytes used by channel (uint8 -> 1, uint16 -> 2)
	bps = nbytes*8                # bps
	max_val = 2**bps - 1          # 255 for 8-bit and 65535 for 16-bit channels.
	_, tubb3_trunc = cv2.threshold(tubb3, int(max_val*0.05), max_val, cv2.THRESH_TOZERO) # treat darkest 5% of colour range as fully black pixels.
	blurred = cv2.GaussianBlur(tubb3_trunc, (5, 5), 0)
	_, binary = cv2.threshold(blurred, 0, max_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return binary

# Compute measurements
def compute_measurements(nuclei_mask, islet1_mask, neurite_mask):
	nuclei_count = len(regionprops(nuclei_mask))
	islet1_count = len(regionprops(islet1_mask))
	neurite_area = np.sum(neurite_mask > 0)
	return nuclei_count, islet1_count, neurite_area

# Display masks
def display_masks(dapi, nuclei_mask, islet1_mask, neurite_mask):
	fig, axes = plt.subplots(1, 4, figsize=(20, 5))
	axes[0].imshow(dapi, cmap='gray')
	axes[0].set_title("DAPI (Nuclei)")
	axes[1].imshow(nuclei_mask, cmap='nipy_spectral')
	axes[1].set_title("Segmented Nuclei (Filtered)")
	axes[2].imshow(islet1_mask, cmap='nipy_spectral')
	axes[2].set_title("ISLET1+ Nuclei (Filtered)")
	axes[3].imshow(neurite_mask, cmap='gray')
	axes[3].set_title("Neurites (TUBB3)")
	for ax in axes:
		ax.axis("off")
	plt.show()

# Main function that call all other functions
def analyze_tiff(filename, output_csv="results.csv"):
	dapi, islet1, tubb3 = load_tiff(filename)
	display_raw_channels(dapi, islet1, tubb3)

	nuclei_mask = segment_nuclei_stardist(dapi)
	nuclei_mask = filter_dapi_nuclei(nuclei_mask)  # Remove small debris
	islet1_mask = segment_islet1_stardist(islet1, nuclei_mask)
	neurite_mask = segment_neurites(tubb3)

	nuclei_count, islet1_count, neurite_area = compute_measurements(nuclei_mask, islet1_mask, neurite_mask)

	results = pd.DataFrame({
		"Nuclei Count": [nuclei_count],
		"ISLET1 Count": [islet1_count],
		"Neurite Area": [neurite_area]
	})
	results.to_csv(output_csv, index=False)
	print("Analysis complete. Results saved in", output_csv)

	display_masks(dapi, nuclei_mask, islet1_mask, neurite_mask)
	return results

# Example usage
# analyze_tiff("path_to_your_image.tif")
