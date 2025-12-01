"""
Analisar regiões específicas da imagem para encontrar as células.
"""
import cv2
import numpy as np
from PIL import Image

img = Image.open("photo1.jpeg")
print(f"PIL size (w x h): {img.size}")

# Converter para array numpy
arr = np.array(img)
print(f"Array shape (h, w, c): {arr.shape}")

# Verificar valores em diferentes posições horizontais na linha Y=250 (primeira célula aprox)
print("\n--- Verificando linha Y=250 (primeira célula) ---")
y = 250
for x in range(50, 300, 20):
    pixel = arr[y, x]
    brightness = np.mean(pixel)
    print(f"  x={x}: RGB={tuple(pixel)}, brightness={brightness:.0f}")

# Verificar valores em diferentes posições verticais na coluna X=130
print("\n--- Verificando coluna X=130 ---")
x = 130
for y in range(150, 500, 30):
    pixel = arr[y, x]
    brightness = np.mean(pixel)
    print(f"  y={y}: RGB={tuple(pixel)}, brightness={brightness:.0f}")

# Vamos tentar encontrar os "quadrados" pretos das bordas das células
print("\n--- Procurando bordas escuras ---")
gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

# Encontrar pixels escuros (bordas pretas)
dark_pixels = np.where(gray < 100)
if len(dark_pixels[0]) > 0:
    y_coords = dark_pixels[0]
    x_coords = dark_pixels[1]
    print(f"  Pixels escuros encontrados: {len(y_coords)}")
    print(f"  X range: {x_coords.min()} - {x_coords.max()}")
    print(f"  Y range: {y_coords.min()} - {y_coords.max()}")

    # Histograma de posições X dos pixels escuros
    x_hist = np.bincount(x_coords, minlength=1200)
    peaks_x = np.where(x_hist > 100)[0]
    print(f"  Picos em X (bordas verticais): {peaks_x[:20]}...")

    # Histograma de posições Y
    y_hist = np.bincount(y_coords, minlength=1600)
    peaks_y = np.where(y_hist > 50)[0]
    print(f"  Picos em Y (bordas horizontais): {peaks_y[:30]}...")
