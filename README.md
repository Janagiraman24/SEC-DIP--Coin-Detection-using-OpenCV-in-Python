# Coin-Detection-using-OpenCV-in-Python

## AIM

To detect and count the total number of coins present in an image using OpenCV morphological operations, thresholding, and SimpleBlobDetector.

## ALGORITHM

- Step 1 — Read the Image

- Step 2 — Convert to Grayscale

- Step 3 — Split into B, G and R Channels

- Step 4 — Perform Thresholding

- Step 5 — Perform Morphological Operations

## PROGRAM

```
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Read image
image = cv2.imread("CoinsA.png")

# Display original image
imageCopy = image.copy()
plt.imshow(image[:, :, ::-1])
plt.title("Original Image")
plt.show()


# Step 2: Convert image to grayscale
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12, 12))
plt.subplot(121)
plt.imshow(image[:, :, ::-1])
plt.title("Original Image")

plt.subplot(122)
plt.imshow(imageGray, cmap="gray")
plt.title("Grayscale Image")
plt.show()


# Step 3: Split image into B, G and R channels
imageB, imageG, imageR = cv2.split(image)

plt.figure(figsize=(20, 12))
plt.subplot(141)
plt.imshow(image[:, :, ::-1])
plt.title("Original Image")

plt.subplot(142)
plt.imshow(imageB, cmap="gray")
plt.title("Blue Channel")

plt.subplot(143)
plt.imshow(imageG, cmap="gray")
plt.title("Green Channel")

plt.subplot(144)
plt.imshow(imageR, cmap="gray")
plt.title("Red Channel")

plt.show()


# Step 4: Thresholding
_, imageThreshold = cv2.threshold(
    imageG,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

plt.imshow(imageThreshold, cmap="gray")
plt.title("Thresholded Image")
plt.show()


# Step 5: Morphological operations
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3, 3)
)

imageDilated = cv2.dilate(
    imageThreshold,
    kernel,
    iterations=1
)

imageDilated2 = cv2.dilate(
    imageThreshold,
    kernel,
    iterations=2
)

plt.imshow(imageDilated2, cmap="gray")
plt.title("Dilated Image Iteration 2")
plt.show()


imageEroded = cv2.erode(
    imageDilated2,
    kernel,
    iterations=1
)

plt.imshow(imageEroded, cmap="gray")
plt.title("Eroded Image")
plt.show()


# Step 6: Create SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()

params.blobColor = 0
params.minDistBetweenBlobs = 2

# Filter by Area
params.filterByArea = False

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.8

detector = cv2.SimpleBlobDetector_create(params)


# Step 7: Detect blobs
keypoints = detector.detect(imageEroded)

# Draw detected coins
output = cv2.drawKeypoints(
    image,
    keypoints,
    np.array([]),
    (0, 0, 255),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(10, 10))
plt.imshow(output[:, :, ::-1])
plt.title("Final Coin Detection")
plt.axis("off")
plt.show()


# Print number of detected coins
print(f"Number of coins detected: {len(keypoints)}")
```

## OUTPUT

<img width="452" height="362" alt="image" src="https://github.com/user-attachments/assets/0de8fefa-72c2-4aef-8b37-b6011d3f7a38" />

<img width="890" height="437" alt="image" src="https://github.com/user-attachments/assets/0804ec40-3718-4786-af82-86803080ef08" />

<img width="925" height="256" alt="image" src="https://github.com/user-attachments/assets/d3dc9830-e802-4eb8-993a-f40833a4cc96" />

<img width="372" height="347" alt="image" src="https://github.com/user-attachments/assets/e72724b9-ef98-4cc4-bc28-e4c755215abb" />

<img width="903" height="327" alt="image" src="https://github.com/user-attachments/assets/cb3a5ee3-bdd9-4acb-a47c-398658d2dd18" />

<img width="568" height="357" alt="image" src="https://github.com/user-attachments/assets/b97895d4-904a-475f-bd1d-5fc3a725b0ee" />

<img width="637" height="382" alt="image" src="https://github.com/user-attachments/assets/3a7bf9dc-58eb-4750-91aa-f4bd5664db21" />

<img width="691" height="292" alt="image" src="https://github.com/user-attachments/assets/d67579f9-4ae5-4275-a585-99c8ce5beed2" />

## RESULT

Thus,  Coin Detection using OpenCV in Python is executed successfully.





