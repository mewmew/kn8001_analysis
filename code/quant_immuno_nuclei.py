import cv2
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import regionprops
from stardist.models import StarDist2D
from csbdeep.utils import normalize

# Load the multi-channel TIFF image
def load_tiff(filename):
	"""
	Load a multi-channel TIFF image with DAPI and two nuclear markers.
	"""
	img = tiff.imread(filename)
	marker1 = img[0]  # First marker (e.g., ISLET1, HOXB4, etc.)
	marker2 = img[1]  # Second marker (e.g., PAX6, OLIG2, etc.)
	dapi = img[2]  # DAPI (all nuclei)
	return dapi, marker1, marker2

# Display raw images
def display_raw_channels(dapi, marker1, marker2):
	"""
	Display raw images for verification.
	"""
	fig, axes = plt.subplots(1, 3, figsize=(15, 5))
	axes[0].imshow(dapi, cmap='gray')
	axes[0].set_title("DAPI (All Nuclei)")
	axes[1].imshow(marker1, cmap='gray')
	axes[1].set_title("Marker 1")
	axes[2].imshow(marker2, cmap='gray')
	axes[2].set_title("Marker 2")
	for ax in axes:
		ax.axis("off")
	plt.show()

# Enhance contrast using CLAHE
def enhance_contrast(image):
	"""
	Improve visibility of structures in the image using CLAHE.
	"""
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
	enhanced = clahe.apply(image.astype(np.uint8))
	return enhanced

# Segment nuclei using StarDist
def segment_nuclei_stardist(image):
	"""
	Use StarDist deep learning model to segment nuclei.
	"""
	model = StarDist2D.from_pretrained("2D_versatile_fluo")
	image_norm = normalize(image, 1, 99.8)
	labels, _ = model.predict_instances(image_norm)
	return labels

# Filter small DAPI nuclei (remove debris)
def filter_dapi_nuclei(dapi_mask):
	"""
	Remove small debris from the DAPI-stained nuclei segmentation.
	"""
	dapi_regions = regionprops(dapi_mask)
	avg_dapi_size = np.mean([r.area for r in dapi_regions])
	min_nuclei_size = avg_dapi_size * 0.5  # Remove objects smaller than 50% of the average size

	filtered_mask = np.zeros_like(dapi_mask)
	for r in dapi_regions:
		if r.area >= min_nuclei_size:
			filtered_mask[r.coords[:, 0], r.coords[:, 1]] = dapi_mask[r.coords[:, 0], r.coords[:, 1]]

	return filtered_mask

# General function to filter marker-positive nuclei based on DAPI size
def filter_marker_by_size(marker_mask, dapi_mask):
	"""
	Ensure marker-positive nuclei:
	- Are large enough (≥50% of the average DAPI nucleus size).
	- Overlap at least slightly with a DAPI-stained nucleus.
	"""
	dapi_regions = regionprops(dapi_mask)
	marker_regions = regionprops(marker_mask)

	avg_dapi_size = np.mean([r.area for r in dapi_regions])
	min_nuclei_size = avg_dapi_size * 0.5

	filtered_mask = np.zeros_like(marker_mask)
	for r in marker_regions:
		if r.area >= min_nuclei_size and np.any(dapi_mask[r.coords[:, 0], r.coords[:, 1]] > 0):
			filtered_mask[r.coords[:, 0], r.coords[:, 1]] = marker_mask[r.coords[:, 0], r.coords[:, 1]]

	return filtered_mask

# Segment marker-positive nuclei using StarDist
def segment_marker_stardist(marker, dapi_mask):
	"""
	Enhance contrast of marker images, segment using StarDist,
	and filter based on DAPI segmentation.
	"""
	marker_enhanced = enhance_contrast(marker)
	marker_mask = segment_nuclei_stardist(marker_enhanced)
	return filter_marker_by_size(marker_mask, dapi_mask)

# Compute measurements
def compute_measurements(nuclei_mask, marker1_mask, marker2_mask):
	"""
	Count:
	- Total DAPI nuclei
	- Marker 1+ nuclei
	- Marker 2+ nuclei
	"""
	nuclei_count = len(regionprops(nuclei_mask))
	marker1_count = len(regionprops(marker1_mask))
	marker2_count = len(regionprops(marker2_mask))
	return nuclei_count, marker1_count, marker2_count

# Display segmented masks
def display_masks(dapi, nuclei_mask, marker1_mask, marker2_mask):
	"""
	Show the final segmented images for validation.
	"""
	fig, axes = plt.subplots(1, 4, figsize=(20, 5))
	axes[0].imshow(dapi, cmap='gray')
	axes[0].set_title("DAPI (All Nuclei)")
	axes[1].imshow(nuclei_mask, cmap='nipy_spectral')
	axes[1].set_title("Segmented Nuclei (Filtered)")
	axes[2].imshow(marker1_mask, cmap='nipy_spectral')
	axes[2].set_title("Marker 1+ Nuclei (Filtered)")
	axes[3].imshow(marker2_mask, cmap='nipy_spectral')
	axes[3].set_title("Marker 2+ Nuclei (Filtered)")
	for ax in axes:
		ax.axis("off")
	plt.show()

# Main function
def analyze_tiff(filename, output_csv="results.csv"):
	"""
	Full pipeline:
	1. Load image channels
	2. Display raw images
	3. Segment DAPI nuclei (StarDist) and remove debris
	4. Segment Marker 1+ nuclei (StarDist) and filter based on DAPI
	5. Segment Marker 2+ nuclei (StarDist) and filter based on DAPI
	6. Compute cell counts
	7. Save results
	8. Display segmented masks
	"""
	dapi, marker1, marker2 = load_tiff(filename)
	display_raw_channels(dapi, marker1, marker2)

	nuclei_mask = segment_nuclei_stardist(dapi)
	nuclei_mask = filter_dapi_nuclei(nuclei_mask)
	marker1_mask = segment_marker_stardist(marker1, nuclei_mask)
	marker2_mask = segment_marker_stardist(marker2, nuclei_mask)

	nuclei_count, marker1_count, marker2_count = compute_measurements(nuclei_mask, marker1_mask, marker2_mask)

	results = pd.DataFrame({
		"Total Nuclei Count (DAPI)": [nuclei_count],
		"Marker 1+ Nuclei Count": [marker1_count],
		"Marker 2+ Nuclei Count": [marker2_count]
	})
	results.to_csv(output_csv, index=False)
	print("✅ Analysis complete. Results saved in", output_csv)

	display_masks(dapi, nuclei_mask, marker1_mask, marker2_mask)
	return results

# Example usage
# analyze_tiff("path_to_your_image.tif")
