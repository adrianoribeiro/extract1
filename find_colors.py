"""
Encontrar regiões coloridas (as imagens têm cores, o papel é cinza).
"""
import cv2
import numpy as np
from PIL import Image

img = cv2.imread("photo1.jpeg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

print(f"Image shape: {img.shape}")

# Pixels com saturação alta = coloridos
saturation = hsv[:, :, 1]
colored_mask = saturation > 30  # Pixels com saturação > 30

# Encontrar contornos das regiões coloridas
colored_binary = colored_mask.astype(np.uint8) * 255
contours, _ = cv2.findContours(colored_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"\nRegiões coloridas encontradas: {len(contours)}")

# Filtrar por tamanho
boxes = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    if 1000 < area < 100000:  # Tamanho razoável para ícones
        boxes.append((x, y, w, h, area))

boxes.sort(key=lambda b: b[1])  # Ordenar por Y (de cima para baixo)

print(f"\nRegiões de tamanho adequado: {len(boxes)}")
print("\nBounding boxes (ordenados por Y):")
for i, (x, y, w, h, area) in enumerate(boxes[:20]):
    print(f"  {i+1}. x={x}, y={y}, w={w}, h={h}, area={area}")
    print(f"      bbox=[{x}, {y}, {x+w}, {y+h}]")

# Salvar imagem de debug
debug = img.copy()
for x, y, w, h, _ in boxes[:20]:
    cv2.rectangle(debug, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imwrite("debug_colors.png", debug)
print("\nDebug salvo em: debug_colors.png")
