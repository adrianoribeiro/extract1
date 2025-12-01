"""
Extração de imagens usando Claude API com visão.

Uso:
    export ANTHROPIC_API_KEY="sua-chave-aqui"
    python extract_with_claude.py photo1.jpeg
"""

import anthropic
import base64
import json
import sys
import os
from PIL import Image


def encode_image(image_path: str) -> tuple[str, str]:
    """Codifica imagem em base64."""
    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")

    ext = image_path.lower().split(".")[-1]
    media_type = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp"
    }.get(ext, "image/jpeg")

    return data, media_type


def analyze_image_with_claude(image_path: str) -> list[dict]:
    """
    Usa Claude API para analisar a imagem e identificar elementos.
    Retorna lista de elementos com coordenadas.
    """
    client = anthropic.Anthropic()

    # Obter dimensões da imagem
    img = Image.open(image_path)
    width, height = img.size

    # Codificar imagem
    image_data, media_type = encode_image(image_path)

    prompt = f"""Analise esta imagem de uma página de vocabulário alemão.

A imagem tem dimensões {width}x{height} pixels.

Para CADA ícone/desenho de vocabulário visível, identifique:
1. A palavra alemã associada (com artigo: der/die/das)
2. A posição do CENTRO do ícone como PORCENTAGEM da imagem

IMPORTANTE:
- Identifique APENAS os ícones/desenhos ilustrativos (NÃO texto)
- Use porcentagens (0-100) para x e y do CENTRO do ícone
- Exemplo: um ícone no canto superior esquerdo seria x=10, y=10
- Exemplo: um ícone no centro da imagem seria x=50, y=50

Retorne APENAS um JSON válido no formato:
{{
  "items": [
    {{"word": "die Wassermelone", "center_x": 15, "center_y": 25, "description": "melancia"}},
    ...
  ]
}}

Liste TODOS os ícones de vocabulário que você consegue ver."""

    print(f"Analisando {image_path} ({width}x{height})...")
    print("Enviando para Claude API...")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )

    # Extrair JSON da resposta
    response_text = response.content[0].text

    # Tentar extrair JSON
    try:
        # Se a resposta contém ```json ... ```
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0]
        else:
            json_str = response_text

        data = json.loads(json_str.strip())
        return data.get("items", [])
    except json.JSONDecodeError as e:
        print(f"Erro ao parsear JSON: {e}")
        print(f"Resposta: {response_text}")
        return []


def extract_images(image_path: str, items: list[dict], output_dir: str = "claude_extracted", box_size: int = 120):
    """
    Extrai as imagens baseado nas coordenadas (porcentagem) do Claude.

    Args:
        box_size: tamanho do quadrado a extrair ao redor do centro (em pixels)
    """
    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(image_path)
    width, height = img.size
    results = []

    for i, item in enumerate(items):
        # Converter porcentagem para pixels
        center_x_pct = item.get("center_x", 0)
        center_y_pct = item.get("center_y", 0)

        if center_x_pct == 0 or center_y_pct == 0:
            # Fallback para formato antigo (bbox)
            bbox = item.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
            else:
                continue
        else:
            # Converter porcentagem para pixels
            center_x = int(width * center_x_pct / 100)
            center_y = int(height * center_y_pct / 100)

            # Criar bounding box ao redor do centro
            half_size = box_size // 2
            x1 = center_x - half_size
            y1 = center_y - half_size
            x2 = center_x + half_size
            y2 = center_y + half_size

        # Validar coordenadas
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(width, int(x2))
        y2 = min(height, int(y2))

        if x2 <= x1 or y2 <= y1:
            continue

        # Recortar
        cropped = img.crop((x1, y1, x2, y2))

        # Salvar
        word = item.get("word", f"item_{i+1}")
        safe_name = word.replace(" ", "_").replace("/", "_")
        filename = f"{i+1:02d}_{safe_name}.png"
        filepath = os.path.join(output_dir, filename)
        cropped.save(filepath)

        results.append({
            "id": i + 1,
            "word": word,
            "description": item.get("description", ""),
            "filename": filename,
            "bbox": [x1, y1, x2, y2]
        })

        print(f"  ✓ {word} -> {filename}")

    # Salvar metadados
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ {len(results)} imagens extraídas para {output_dir}/")
    return results


def main():
    if len(sys.argv) < 2:
        print("Uso: python extract_with_claude.py <imagem>")
        print("Exemplo: python extract_with_claude.py photo1.jpeg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Erro: arquivo não encontrado: {image_path}")
        sys.exit(1)

    # Verificar API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Erro: ANTHROPIC_API_KEY não configurada")
        print("Execute: export ANTHROPIC_API_KEY='sua-chave-aqui'")
        sys.exit(1)

    # Analisar com Claude
    items = analyze_image_with_claude(image_path)

    if not items:
        print("Nenhum item encontrado.")
        sys.exit(1)

    print(f"\nEncontrados {len(items)} itens:")
    for item in items:
        cx = item.get('center_x', '?')
        cy = item.get('center_y', '?')
        print(f"  - {item.get('word')}: centro=({cx}%, {cy}%)")

    # Extrair imagens
    output_dir = f"claude_extracted_{os.path.splitext(os.path.basename(image_path))[0]}"
    extract_images(image_path, items, output_dir)


if __name__ == "__main__":
    main()
