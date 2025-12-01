"""
Prova de Conceito: Extração de imagens usando LLM + PIL

Este script demonstra a viabilidade de:
1. Usar uma LLM (Claude) para analisar uma imagem e identificar elementos
2. Extrair cada elemento individual para criar jogos/quizzes
"""

from PIL import Image
import json
import os

# Dados extraídos pela LLM (Claude) ao analisar photo1.jpeg
# Em produção, isso viria de uma chamada à API do Claude
# Coordenadas ajustadas após análise mais precisa [x1, y1, x2, y2]
EXTRACTED_DATA_PHOTO1 = [
    {"word": "das Wasser", "bbox": [85, 200, 195, 310], "description": "torneira com água"},
    {"word": "das Wort", "bbox": [85, 315, 195, 425], "description": "livros empilhados"},
    {"word": "das Wetter", "bbox": [85, 430, 195, 540], "description": "pessoas no clima/chuva"},
    {"word": "die Wassermelone", "bbox": [85, 570, 195, 680], "description": "melancia"},
    {"word": "die Waffel", "bbox": [85, 690, 195, 800], "description": "waffle"},
    {"word": "die Wolke", "bbox": [85, 810, 195, 920], "description": "nuvem"},
    {"word": "die Krawatte", "bbox": [85, 930, 195, 1040], "description": "gravata"},
    {"word": "die Weste", "bbox": [85, 1055, 195, 1165], "description": "colete"},
    {"word": "die Wolle", "bbox": [85, 1175, 195, 1285], "description": "novelo de lã"},
]


def extract_images(image_path: str, extracted_data: list, output_dir: str = "extracted"):
    """
    Extrai imagens individuais baseado nas coordenadas fornecidas pela LLM.

    Args:
        image_path: Caminho da imagem original
        extracted_data: Lista de dicts com {word, bbox, description}
        output_dir: Diretório de saída
    """
    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(image_path)
    print(f"Processando: {image_path} ({img.size[0]}x{img.size[1]})")

    results = []

    for i, item in enumerate(extracted_data):
        word = item["word"]
        bbox = item["bbox"]  # [x1, y1, x2, y2]
        desc = item["description"]

        # Recorta a região
        cropped = img.crop(bbox)

        # Salva a imagem
        safe_name = word.replace(" ", "_").replace("/", "_")
        filename = f"{i+1:02d}_{safe_name}.png"
        filepath = os.path.join(output_dir, filename)
        cropped.save(filepath)

        results.append({
            "id": i + 1,
            "word": word,
            "description": desc,
            "image_file": filename,
            "original_bbox": bbox
        })

        print(f"  ✓ Extraído: {word} -> {filename}")

    # Salva metadados para uso no quiz
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nTotal: {len(results)} imagens extraídas")
    print(f"Metadados salvos em: {metadata_path}")

    return results


if __name__ == "__main__":
    # Teste com photo1.jpeg
    extract_images("photo1.jpeg", EXTRACTED_DATA_PHOTO1, "extracted_photo1")
