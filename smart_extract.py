"""
Abordagem Híbrida: LLM identifica regiões + refinamento automático

Para produção, você chamaria a API do Claude para analisar a imagem.
Aqui simulo o que a LLM retornaria após análise visual.
"""

import cv2
import numpy as np
from PIL import Image
import json
import os


def refine_bbox_with_edges(img, initial_bbox, search_margin=30):
    """
    Dado um bbox aproximado da LLM, refina usando detecção de bordas.
    """
    x1, y1, x2, y2 = initial_bbox
    h, w = img.shape[:2]

    # Expandir área de busca
    sx1 = max(0, x1 - search_margin)
    sy1 = max(0, y1 - search_margin)
    sx2 = min(w, x2 + search_margin)
    sy2 = min(h, y2 + search_margin)

    # Recortar região de busca
    region = img[sy1:sy2, sx1:sx2]

    # Converter para escala de cinza e detectar bordas
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Encontrar contornos na região
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Encontrar o maior contorno retangular
        best_cnt = max(contours, key=cv2.contourArea)
        rx, ry, rw, rh = cv2.boundingRect(best_cnt)

        # Converter de volta para coordenadas globais
        return [sx1 + rx, sy1 + ry, sx1 + rx + rw, sy1 + ry + rh]

    return initial_bbox


def extract_with_llm_guidance(image_path: str, llm_data: list, output_dir: str = "smart_extracted"):
    """
    Extrai imagens usando dados da LLM com refinamento opcional.
    """
    os.makedirs(output_dir, exist_ok=True)

    img_cv = cv2.imread(image_path)
    img_pil = Image.open(image_path)

    print(f"Processando: {image_path}")
    print(f"Dimensões: {img_pil.size}")

    results = []
    debug_img = img_cv.copy()

    for i, item in enumerate(llm_data):
        word = item["word"]
        bbox = item["bbox"]

        # Desenhar bbox original (vermelho)
        cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

        # Recortar e salvar
        cropped = img_pil.crop(bbox)

        safe_name = word.replace(" ", "_").replace("/", "_")
        filename = f"{i+1:02d}_{safe_name}.png"
        filepath = os.path.join(output_dir, filename)
        cropped.save(filepath)

        results.append({
            "id": i + 1,
            "word": word,
            "description": item.get("description", ""),
            "image_file": filename,
            "bbox": bbox
        })

        print(f"  ✓ {word}")

    # Salvar metadados
    with open(os.path.join(output_dir, "quiz_data.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Salvar debug
    cv2.imwrite(os.path.join(output_dir, "_debug.png"), debug_img)

    print(f"\nExtraídas {len(results)} imagens")
    print(f"Dados do quiz salvos em: quiz_data.json")

    return results


# ============================================================
# SIMULAÇÃO: Dados que a LLM (Claude) retornaria via API
# Em produção, isso viria de: claude.analyze_image(image, prompt)
# ============================================================

# Analisando photo1.jpeg com mais cuidado:
# - Imagem 1200x1600
# - Coluna de imagens na esquerda, ~100px de largura
# - 9 linhas de conteúdo
# - Cada célula ~110px de altura

LLM_ANALYSIS_PHOTO1 = [
    {"word": "das Wasser", "bbox": [78, 193, 185, 300], "description": "torneira com água corrente"},
    {"word": "das Wort", "bbox": [78, 308, 185, 415], "description": "pilha de livros"},
    {"word": "das Wetter", "bbox": [78, 423, 185, 530], "description": "cena de tempo/clima"},
    {"word": "die Wassermelone", "bbox": [78, 560, 185, 667], "description": "fatia de melancia"},
    {"word": "die Waffel", "bbox": [78, 680, 185, 787], "description": "waffle em formato de flor"},
    {"word": "die Wolke", "bbox": [78, 800, 185, 907], "description": "nuvem azul"},
    {"word": "die Krawatte", "bbox": [78, 920, 185, 1027], "description": "gravata vermelha"},
    {"word": "die Weste", "bbox": [78, 1045, 185, 1152], "description": "colete verde"},
    {"word": "die Wolle", "bbox": [78, 1165, 185, 1272], "description": "novelo de lã azul"},
]


if __name__ == "__main__":
    extract_with_llm_guidance("photo1.jpeg", LLM_ANALYSIS_PHOTO1, "smart_extracted")
