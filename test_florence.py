"""
Teste de extração com Florence-2 via Replicate.

Uso:
    export REPLICATE_API_TOKEN="r8_..."
    python test_florence.py photo1.jpeg
"""

import replicate
import base64
import sys
import os
import json
from PIL import Image


def test_florence(image_path: str):
    """Testa Florence-2 para detecção de objetos."""

    # Verificar token
    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("Erro: REPLICATE_API_TOKEN não configurado")
        print("1. Crie conta em: https://replicate.com")
        print("2. Pegue o token em: https://replicate.com/account/api-tokens")
        print("3. Execute: export REPLICATE_API_TOKEN='r8_...'")
        return

    print(f"Analisando: {image_path}")

    # Converter imagem para base64 URI
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()

    ext = image_path.lower().split(".")[-1]
    mime = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"
    data_uri = f"data:{mime};base64,{img_data}"

    print("Enviando para Florence-2...")

    # Testar diferentes tasks do Florence-2
    tasks_to_test = [
        "<OD>",  # Object Detection
        "<CAPTION_TO_PHRASE_GROUNDING>",  # Grounding com caption
    ]

    for task in tasks_to_test:
        print(f"\n--- Task: {task} ---")

        try:
            if task == "<CAPTION_TO_PHRASE_GROUNDING>":
                # Para grounding, precisamos de um texto
                output = replicate.run(
                    "lucataco/florence-2-large:e946a1ac25f7b65c82c30eb5c3e8118a2e92b80dd42840157dd532f55eea3e2a",
                    input={
                        "image": data_uri,
                        "task_input": task,
                        "text_input": "icons, drawings, illustrations, pictures"
                    }
                )
            else:
                output = replicate.run(
                    "lucataco/florence-2-large:e946a1ac25f7b65c82c30eb5c3e8118a2e92b80dd42840157dd532f55eea3e2a",
                    input={
                        "image": data_uri,
                        "task_input": task
                    }
                )

            print(f"Resultado: {json.dumps(output, indent=2, ensure_ascii=False)}")

        except Exception as e:
            print(f"Erro: {e}")

    print("\n✓ Teste concluído")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python test_florence.py <imagem>")
        sys.exit(1)

    test_florence(sys.argv[1])
