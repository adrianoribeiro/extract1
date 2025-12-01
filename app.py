"""
App Semi-Automática para Extração de Imagens de Vocabulário

Como usar:
1. streamlit run app.py
2. Faz upload da foto
3. Sistema detecta automaticamente
4. Ajusta os retângulos se necessário
5. Clica "Extrair" para salvar as imagens
"""

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2
import numpy as np
import json
import os
from io import BytesIO
import base64


def detect_colored_regions(img_array):
    """Detecta regiões coloridas na imagem."""
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    colored_mask = saturation > 25

    kernel = np.ones((5, 5), np.uint8)
    colored_mask = cv2.morphologyEx(colored_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    colored_mask = cv2.morphologyEx(colored_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(colored_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    h, w = img_array.shape[:2]
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        aspect = bw / bh if bh > 0 else 0

        # Filtros básicos
        if 1000 < area < 50000 and 0.3 < aspect < 3.0:
            boxes.append({
                "left": x,
                "top": y,
                "width": bw,
                "height": bh
            })

    # Ordenar por posição (top-left para bottom-right)
    boxes.sort(key=lambda b: (b["top"], b["left"]))
    return boxes


def boxes_to_canvas_objects(boxes):
    """Converte boxes para formato do canvas."""
    objects = []
    for i, box in enumerate(boxes):
        objects.append({
            "type": "rect",
            "left": box["left"],
            "top": box["top"],
            "width": box["width"],
            "height": box["height"],
            "fill": "rgba(0, 255, 0, 0.1)",
            "stroke": "#00ff00",
            "strokeWidth": 2,
        })
    return objects


def extract_images_from_boxes(img, boxes, output_dir="extracted_images"):
    """Extrai imagens baseado nos boxes."""
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for i, box in enumerate(boxes):
        x = int(box["left"])
        y = int(box["top"])
        w = int(box["width"])
        h = int(box["height"])

        # Recortar
        cropped = img.crop((x, y, x + w, y + h))

        # Salvar
        filename = f"image_{i+1:02d}.png"
        filepath = os.path.join(output_dir, filename)
        cropped.save(filepath)

        results.append({
            "id": i + 1,
            "filename": filename,
            "bbox": [x, y, x + w, y + h]
        })

    return results


# =====================
# STREAMLIT APP
# =====================

st.set_page_config(page_title="Extrator de Imagens", layout="wide")
st.title("🖼️ Extrator Semi-Automático de Imagens")

# Upload
uploaded_file = st.file_uploader("Faça upload da foto", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Carregar imagem
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)

    # Redimensionar se muito grande (para o canvas)
    max_width = 800
    scale = 1.0
    if img.width > max_width:
        scale = max_width / img.width
        new_size = (int(img.width * scale), int(img.height * scale))
        img_display = img.resize(new_size)
        img_array_display = np.array(img_display)
    else:
        img_display = img
        img_array_display = img_array

    # Detectar automaticamente
    if "boxes" not in st.session_state or st.button("🔄 Re-detectar"):
        detected = detect_colored_regions(img_array_display)
        st.session_state.boxes = detected
        st.success(f"Detectadas {len(detected)} regiões automaticamente")

    # Criar objetos iniciais para o canvas
    initial_objects = boxes_to_canvas_objects(st.session_state.get("boxes", []))

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Ajuste os retângulos")
        st.caption("Desenhe novos retângulos ou ajuste os existentes")

        # Canvas interativo
        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 0, 0.1)",
            stroke_width=2,
            stroke_color="#00ff00",
            background_image=img_display,
            initial_drawing={"objects": initial_objects} if initial_objects else None,
            drawing_mode="rect",
            height=img_display.height,
            width=img_display.width,
            key="canvas",
        )

    with col2:
        st.subheader("Ações")

        # Contar retângulos
        num_boxes = 0
        if canvas_result.json_data and canvas_result.json_data.get("objects"):
            num_boxes = len(canvas_result.json_data["objects"])

        st.metric("Retângulos", num_boxes)

        if st.button("✅ Extrair Imagens", type="primary"):
            if canvas_result.json_data and canvas_result.json_data.get("objects"):
                # Converter coordenadas de volta para escala original
                boxes = []
                for obj in canvas_result.json_data["objects"]:
                    if obj["type"] == "rect":
                        boxes.append({
                            "left": obj["left"] / scale,
                            "top": obj["top"] / scale,
                            "width": obj["width"] / scale,
                            "height": obj["height"] / scale
                        })

                # Extrair
                results = extract_images_from_boxes(img, boxes)

                st.success(f"✓ {len(results)} imagens extraídas!")

                # Mostrar preview
                st.subheader("Imagens Extraídas")
                cols = st.columns(4)
                for i, result in enumerate(results):
                    with cols[i % 4]:
                        extracted_img = Image.open(f"extracted_images/{result['filename']}")
                        st.image(extracted_img, caption=f"#{result['id']}", width=100)

                # Salvar metadados
                with open("extracted_images/metadata.json", "w") as f:
                    json.dump(results, f, indent=2)

                st.info("Imagens salvas em: extracted_images/")
            else:
                st.warning("Desenhe pelo menos um retângulo!")

        # Instruções
        st.markdown("---")
        st.markdown("""
        **Instruções:**
        1. Arraste para criar retângulos
        2. Ajuste os existentes se necessário
        3. Clique "Extrair Imagens"
        """)

else:
    st.info("👆 Faça upload de uma foto para começar")

    # Exemplo
    st.markdown("---")
    st.subheader("Exemplo de uso:")
    st.markdown("""
    1. Upload de foto de página de vocabulário
    2. Sistema detecta as imagens automaticamente
    3. Você ajusta se necessário
    4. Extrai e usa no quiz!
    """)
