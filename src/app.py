import streamlit as st
from PIL import Image
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import gc
import re

st.set_page_config(page_title="LaTeX OCR Demo", layout="wide")
st.title("Распознавание рукописных формул в LaTeX")
st.markdown("Загрузите изображение с рукописной формулой, и модель преобразует её в LaTeX код и покажет результат")

if "image" not in st.session_state:
    st.session_state.image = None

@st.cache_resource
def load_model():
    from transformers import BitsAndBytesConfig
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, "KaFFa0/qwen3-vl-latex-lora")
    model = model.merge_and_unload()
    model.eval()
    processor = AutoProcessor.from_pretrained("KaFFa0/qwen3-vl-latex-lora")
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    return model, processor

with st.spinner("Загрузка модели..."):
    model, processor = load_model()

with st.sidebar:
    st.header("Загрузите изображение")
    uploaded_file = st.file_uploader("Выберите файл", type=["png", "jpg", "jpeg", "bmp"])
    use_camera = st.checkbox("Использовать камеру")
    if use_camera:
        camera_image = st.camera_input("Сделайте снимок")
        if camera_image:
            uploaded_file = camera_image

    if uploaded_file is not None:
        st.session_state.image = Image.open(uploaded_file).convert("RGB")

col1, col2 = st.columns(2)

with col1:
    st.header("Исходное изображение")
    if st.session_state.image is not None:
        st.image(st.session_state.image, caption="Загруженное изображение", use_container_width=True)
    else:
        st.info("Загрузите изображение слева.")
        st.stop()

prompt = processor.apply_chat_template(
    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Convert this handwritten formula into LaTeX"}]}],
    tokenize=False,
    add_generation_prompt=True
)

with st.spinner("Распознавание..."):
    inputs = processor(
        images=st.session_state.image,
        text=prompt,
        return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
            pad_token_id=processor.tokenizer.pad_token_id,
            use_cache=True,
        )
    generated = processor.decode(
        output_ids[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

with col2:
    st.header("Результат")
    st.subheader("LaTeX-код:")
    st.code(generated, language="latex")

    st.subheader("Рендер формулы:")
    cleaned = re.sub(r'^\s*(\${1,2}|\\[\(\[])|(\${1,2}|\\[\)\]])\s*$', '', generated)
    cleaned = cleaned.strip()
    if cleaned:
        try:
            st.latex(cleaned)
        except Exception as e:
            st.error(f"Ошибка st.latex: {e}")
    else:
        st.warning("Формула пуста после очистки")
        st.write("Сырая строка:", repr(generated))
