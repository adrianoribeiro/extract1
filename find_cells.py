"""
Encontrar as coordenadas exatas das células de imagem.
"""
import cv2
import numpy as np

img = cv2.imread("photo1.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(f"Dimensões da imagem: {img.shape}")

# Encontrar onde estão os pixels não-brancos (conteúdo)
# Inverter: pixels escuros (conteúdo) ficam brancos
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

# Encontrar contornos
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"\nContornos encontrados: {len(contours)}")

# Filtrar por tamanho e mostrar os maiores
boxes = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    if area > 5000:  # Ignorar ruído pequeno
        boxes.append((x, y, w, h, area))

# Ordenar por área
boxes.sort(key=lambda b: b[4], reverse=True)

print("\nMaiores regiões de conteúdo:")
for i, (x, y, w, h, area) in enumerate(boxes[:20]):
    print(f"  {i+1}. x={x}, y={y}, w={w}, h={h}, area={area}")

# Também vamos verificar onde estão as linhas verticais (bordas das células)
print("\n--- Analisando coluna esquerda (x < 400) ---")
left_boxes = [(x, y, w, h) for x, y, w, h, _ in boxes if x < 400]
left_boxes.sort(key=lambda b: b[1])  # Ordenar por Y

for i, (x, y, w, h) in enumerate(left_boxes[:15]):
    print(f"  {i+1}. bbox=[{x}, {y}, {x+w}, {y+h}]  (w={w}, h={h})")
