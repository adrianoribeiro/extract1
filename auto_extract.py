"""
Extração automática de imagens usando OpenCV para detecção de contornos.
"""

import cv2
import numpy as np
from PIL import Image
import os


def detect_image_cells(image_path: str, output_dir: str = "auto_extracted"):
    """
    Detecta automaticamente células de imagem em uma página de vocabulário.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Carregar imagem
    img = cv2.imread(image_path)
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar threshold para binarizar
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos por tamanho (queremos células pequenas, não a página toda)
    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / h if h > 0 else 0

        # Filtrar: células são aproximadamente quadradas e de tamanho médio
        if 3000 < area < 50000 and 0.5 < aspect_ratio < 2.0:
            cells.append((x, y, w, h))

    # Ordenar por posição Y (de cima para baixo)
    cells.sort(key=lambda c: c[1])

    print(f"Detectadas {len(cells)} células potenciais")

    # Extrair cada célula
    for i, (x, y, w, h) in enumerate(cells):
        # Adicionar margem
        margin = 5
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img.shape[1], x + w + margin)
        y2 = min(img.shape[0], y + h + margin)

        cell_img = original[y1:y2, x1:x2]

        # Salvar
        filename = f"cell_{i+1:02d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, cell_img)
        print(f"  Salvo: {filename} ({w}x{h} em posição {x},{y})")

    # Salvar imagem com contornos para debug
    debug_img = original.copy()
    for x, y, w, h in cells:
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, "_debug_contours.png"), debug_img)
    print(f"\nImagem de debug salva: _debug_contours.png")

    return cells


def detect_with_grid(image_path: str, output_dir: str = "grid_extracted"):
    """
    Abordagem alternativa: detectar linhas horizontais e verticais para encontrar a grade.
    """
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detectar bordas
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detectar linhas
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is not None:
        print(f"Detectadas {len(lines)} linhas")

        # Desenhar linhas para debug
        debug_img = original.copy()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imwrite(os.path.join(output_dir, "_debug_lines.png"), debug_img)
        print(f"Debug de linhas salvo")

    return lines


if __name__ == "__main__":
    print("=== Método 1: Detecção de contornos ===")
    detect_image_cells("photo1.jpeg", "auto_extracted")

    print("\n=== Método 2: Detecção de grade ===")
    detect_with_grid("photo1.jpeg", "grid_extracted")
