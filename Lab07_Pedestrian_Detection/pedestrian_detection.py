"""
Lab 07 — Pedestrian Detection using OpenCV
============================================
Detect pedestrians using OpenCV's HOG + SVM method.

Key Concepts:
- HOG (Histogram of Oriented Gradients) Descriptors
- SVM Classifier
- Non-Maximum Suppression
- OpenCV's built-in Pedestrian Detector
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

# ============================================================
# 1. Introduction & Setup
# ============================================================
print("=" * 60)
print("LAB 07 — Pedestrian Detection using OpenCV (HOG + SVM)")
print("=" * 60)

# Initialize HOG descriptor with pre-trained pedestrian detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

print("\n[OK] HOG Descriptor initialized")
print(f"   Window Size: {hog.winSize}")
print(f"   Block Size:  {hog.blockSize}")
print(f"   Cell Size:   {hog.cellSize}")
print(f"   Block Stride: {hog.blockStride}")
print(f"   Nbins:       {hog.nbins}")

# ============================================================
# 2. Create Synthetic Test Images
# ============================================================
print("\n" + "-" * 40)
print("STEP 2: Creating Synthetic Test Images")
print("-" * 40)


def create_pedestrian_scene(width=640, height=480, num_figures=3):
    """Create a synthetic scene with pedestrian-like figures."""
    # Create a background (road-like scene)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky gradient
    for y in range(height // 2):
        blue_val = int(200 - y * 0.5)
        img[y, :] = [blue_val, int(blue_val * 0.8), int(blue_val * 0.5)]

    # Ground
    img[height // 2:, :] = [80, 80, 80]

    # Road markings
    for x in range(0, width, 80):
        cv2.rectangle(img, (x, height // 2 + 50), (x + 40, height // 2 + 55),
                      (200, 200, 200), -1)

    # Draw pedestrian-like stick figures
    positions = []
    for i in range(num_figures):
        cx = int(width * (i + 1) / (num_figures + 1))
        cy = height // 2 + 20

        # Head
        cv2.circle(img, (cx, cy - 60), 12, (200, 180, 160), -1)
        # Body
        cv2.line(img, (cx, cy - 48), (cx, cy), (100, 100, 200), 3)
        # Arms
        cv2.line(img, (cx - 20, cy - 30), (cx + 20, cy - 30), (100, 100, 200), 3)
        # Legs
        cv2.line(img, (cx, cy), (cx - 15, cy + 40), (100, 100, 200), 3)
        cv2.line(img, (cx, cy), (cx + 15, cy + 40), (100, 100, 200), 3)

        positions.append((cx - 30, cy - 75, 60, 120))

    return img, positions


def non_max_suppression(boxes, scores, threshold=0.3):
    """Apply Non-Maximum Suppression to reduce overlapping detections."""
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / areas[order[1:]]

        inds = np.where(overlap <= threshold)[0]
        order = order[inds + 1]

    return keep


# Generate test scenes
scenes = []
for i in range(3):
    img, positions = create_pedestrian_scene(num_figures=i + 2)
    scenes.append((img, positions))
    print(f"  Scene {i + 1}: {img.shape} with {len(positions)} figures")

# ============================================================
# 3. HOG Feature Visualization
# ============================================================
print("\n" + "-" * 40)
print("STEP 3: HOG Feature Visualization")
print("-" * 40)

# Compute HOG features on a sample region
sample_img = cv2.resize(scenes[0][0], (128, 256))
gray_sample = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)

# Compute HOG features
win_size = (128, 256)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
nbins = 9

hog_descriptor = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
features = hog_descriptor.compute(gray_sample)
print(f"\n[MATH] HOG Feature Vector Length: {len(features)}")

# Visualize using gradient magnitude (simplified HOG visualization)
gx = cv2.Sobel(gray_sample, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(gray_sample, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(gx ** 2 + gy ** 2)
orientation = np.arctan2(gy, gx) * 180 / np.pi

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
axes[0].imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image', fontweight='bold')
axes[0].axis('off')

axes[1].imshow(magnitude, cmap='hot')
axes[1].set_title('Gradient Magnitude', fontweight='bold')
axes[1].axis('off')

axes[2].imshow(orientation, cmap='hsv')
axes[2].set_title('Gradient Orientation', fontweight='bold')
axes[2].axis('off')

plt.suptitle('HOG Feature Visualization', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('hog_visualization.png', dpi=150)
plt.show()
print("[OK] HOG visualization saved")

# ============================================================
# 4. Pedestrian Detection
# ============================================================
print("\n" + "-" * 40)
print("STEP 4: Pedestrian Detection")
print("-" * 40)

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

for i, (img, ground_truth) in enumerate(scenes):
    # Detect pedestrians
    boxes, weights = hog.detectMultiScale(
        img,
        winStride=(8, 8),
        padding=(4, 4),
        scale=1.05
    )

    print(f"\n  Scene {i + 1}:")
    print(f"    Ground truth figures: {len(ground_truth)}")
    print(f"    Raw detections:       {len(boxes)}")

    # Draw results
    result_img = img.copy()

    # Draw ground truth in green
    for (x, y, w, h) in ground_truth:
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Apply NMS if detections exist
    if len(boxes) > 0:
        keep = non_max_suppression(boxes, weights.flatten())
        print(f"    After NMS:            {len(keep)}")

        for idx in keep:
            (x, y, w, h) = boxes[idx]
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(result_img, f'{weights[idx][0]:.2f}',
                        (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    axes[i].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    axes[i].set_title(f'Scene {i + 1} ({len(ground_truth)} figures)', fontweight='bold')
    axes[i].axis('off')

plt.suptitle('Pedestrian Detection Results\n(Green=Ground Truth, Red=Detections)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('detection_results.png', dpi=150)
plt.show()
print("\n[OK] Detection results saved as 'detection_results.png'")

# ============================================================
# 5. Summary
# ============================================================
print("\n" + "-" * 40)
print("STEP 5: Summary")
print("-" * 40)

print("""
[PIN] HOG + SVM Pedestrian Detection Summary:
   -----------------------------------------
   - HOG (Histogram of Oriented Gradients) extracts gradient features
   - A pre-trained linear SVM classifies windows as pedestrian/not
   - Sliding window scans the image at multiple scales
   - NMS (Non-Maximum Suppression) removes duplicate detections

   [INFO] Limitations:
   - Works best on upright, fully visible pedestrians
   - Sensitive to occlusion and unusual poses
   - Slower than deep learning methods (YOLO, SSD)

   [ROCKET] For production use, consider:
   - YOLO v8 for real-time detection
   - SSD MobileNet for mobile/embedded systems
""")

print("=" * 60)
print("[OK] Lab 07 Complete!")
print("=" * 60)
