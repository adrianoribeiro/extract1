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

    # Testar com Florence-2 (modelo oficial da Microsoft)
    # Usando versão mais recente disponível no Replicate

    try:
        print("\n--- Object Detection ---")
        output = replicate.run(
            "adirik/florence-2-base-promptgen-v2:87e26ba3e153e15fef62c972ed165b9c8a14b5c0bb60e27eee3f6e5b0dc5f602",
            input={
                "image": data_uri,
                "task": "Object Detection"
            }
        )
        print(f"Resultado: {json.dumps(output, indent=2, ensure_ascii=False)}")

    except Exception as e:
        print(f"Erro modelo 1: {e}")

        # Tentar modelo alternativo
        print("\n--- Tentando modelo alternativo ---")
        try:
            output = replicate.run(
                "zsxkib/florence-2:4d5c5835389d0a501c565e3c8af7e2e5b97ce32f0584cefe5f0ed12017ed2c2c",
                input={
                    "image": data_uri,
                    "task": "object_detection"
                }
            )
            print(f"Resultado: {json.dumps(output, indent=2, ensure_ascii=False)}")
        except Exception as e2:
            print(f"Erro modelo 2: {e2}")

    print("\n✓ Teste concluído")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python test_florence.py <imagem>")
        sys.exit(1)

    test_florence(sys.argv[1])
