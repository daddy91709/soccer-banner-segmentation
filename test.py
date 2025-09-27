import cv2
import matplotlib.pyplot as plt
from utils import find_matching_image_for_mask

IMAGES_DIR = "data/Tagged_Images"
MASKS_DIR = "data/Masks"

# prendi la prima maschera
mask_path = f"{MASKS_DIR}/mask0.png"
img_path = find_matching_image_for_mask(mask_path, IMAGES_DIR)

if img_path is None:
    raise RuntimeError("Nessuna immagine trovata per mask0.png")

# leggi immagine e maschera
img = cv2.imread(img_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# binarizza maschera
_, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# trova contorni e bounding box
contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)  # verde

# salva invece di mostrare
plt.figure(figsize=(10,6))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis("off")
plt.title("Bounding box calcolate dalla maschera")
plt.savefig("test_result.png", bbox_inches='tight', dpi=150)
print("Immagine salvata come test_result.png")
