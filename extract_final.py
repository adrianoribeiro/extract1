"""
Extração final - usando detecção de regiões coloridas + filtros.
"""
import cv2
import numpy as np
from PIL import Image
import json
import os


def extract_vocabulary_images(image_path: str, output_dir: str = "final_extracted"):
    """
    Extrai imagens de vocabulário de uma página fotografada.
    """
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, w = img.shape[:2]
    print(f"Imagem: {image_path} ({w}x{h})")

    # Detectar pixels coloridos (saturação > threshold)
    saturation = hsv[:, :, 1]
    colored_mask = saturation > 25

    # Operações morfológicas para limpar ruído e conectar regiões
    kernel = np.ones((5, 5), np.uint8)
    colored_mask = cv2.morphologyEx(colored_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    colored_mask = cv2.morphologyEx(colored_mask, cv2.MORPH_OPEN, kernel)

    # Encontrar contornos
    contours, _ = cv2.findContours(colored_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos
    candidates = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        aspect = bw / bh if bh > 0 else 0

        # Critérios para ícones de vocabulário:
        # - Coluna de ícones (x < 370)
        # - Tamanho razoável
        # - Aproximadamente quadrado
        # - Abaixo do título (y > 200)
        if (x < 370 and
            2000 < area < 30000 and
            0.4 < aspect < 2.5 and
            y > 200):
            candidates.append({
                'x': x, 'y': y, 'w': bw, 'h': bh,
                'area': area, 'aspect': aspect
            })

    # Ordenar por Y (de cima para baixo)
    candidates.sort(key=lambda c: c['y'])

    print(f"\nCandidatos encontrados: {len(candidates)}")

    # Extrair cada candidato
    pil_img = Image.open(image_path)
    results = []
    debug_img = img.copy()

    for i, c in enumerate(candidates):
        x, y, bw, bh = c['x'], c['y'], c['w'], c['h']

        # Adicionar margem
        margin = 10
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w, x + bw + margin)
        y2 = min(h, y + bh + margin)

        # Recortar
        cropped = pil_img.crop((x1, y1, x2, y2))

        # Salvar
        filename = f"image_{i+1:02d}.png"
        filepath = os.path.join(output_dir, filename)
        cropped.save(filepath)

        results.append({
            'id': i + 1,
            'image_file': filename,
            'bbox': [x1, y1, x2, y2],
            'original_bbox': [x, y, x + bw, y + bh]
        })

        # Debug: desenhar retângulo
        cv2.rectangle(debug_img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(debug_img, str(i+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f"  {i+1}. bbox=[{x}, {y}, {x+bw}, {y+bh}]")

    # Salvar metadados
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Salvar debug
    cv2.imwrite(os.path.join(output_dir, "_debug.png"), debug_img)

    print(f"\n✓ {len(results)} imagens extraídas para {output_dir}/")
    return results


if __name__ == "__main__":
    extract_vocabulary_images("photo1.jpeg", "final_photo1")
    print("\n" + "="*50)
    extract_vocabulary_images("photo2.jpeg", "final_photo2")
