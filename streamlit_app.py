import streamlit as st
import torch
from PIL import Image
from image_processing import *
from neural_transfer import *
from torchvision.models import vgg19, VGG19_Weights
from torchvision.models import resnet50

# Определение устройства (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Максимальный размер изображения
imsize = 512 if torch.cuda.is_available() else 128

# Интерфейс Streamlit
st.title("Перенос стиля")

# Функция для отображения чисел в научной нотации
def format_scientific(val):
    return f"{val:.0e}"

# Разделение экрана на 3 колонки
col1, col2, col3 = st.columns([1, 1, 1])

# Выбор модели
with col1:
    selected_model = st.selectbox("Выберите модель:", ["VGG19", "ResNet50"])

# Определение веса по умолчанию в зависимости от выбранной модели
if selected_model == "VGG19":
    index = 3
elif selected_model == "ResNet50":
    index = 8

# Выбор веса стиля и количества шагов
with col2:
    weights = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14]
    formatted_weights = [format_scientific(w) for w in weights]
    selected_weight = st.selectbox("Выберите вес стиля:", formatted_weights, index=index)
    selected_weight = weights[formatted_weights.index(selected_weight)]

with col3:
    selected_steps = st.selectbox("Выберите количество шагов:", [300, 500, 1000])

# Загрузка контентного и стилевого изображений
content_file = st.file_uploader("Загрузите контентное изображение", type=["png", "jpg", "jpeg"])
style_file = st.file_uploader("Загрузите стилевое изображение", type=["png", "jpg", "jpeg"])

# Обработка изображений
if content_file and style_file:
    content_img_pil = Image.open(content_file).convert("RGB")
    style_img_pil = Image.open(style_file).convert("RGB")

    # Масштабируем изображения
    content_img, content_size = load_content_image(content_img_pil, imsize, device)
    style_img = load_style_image(style_img_pil, content_size, device)

    # Отображение загруженных изображений
    with col1:
        st.image(content_img_pil, caption="Контентное изображение", use_container_width=True)
    with col2:
        st.image(style_img_pil, caption="Стилевое изображение", use_container_width=True)

    # Прокрутка страницы вниз после загрузки
    st.markdown(
        """
        <script>
            function scrollToBottom() {
                window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
            }
            setTimeout(scrollToBottom, 500);
        </script>
        """,
        unsafe_allow_html=True
    )

    # Нормализация и подготовка моделей
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
    input_img = content_img.clone()
    vgg_cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    resnet_cnn = resnet50(pretrained=True).eval()

    # Применяем перенос стиля для выбранной модели
    if selected_model == "VGG19":
        output = run_style_transfer(vgg_cnn, cnn_normalization_mean, cnn_normalization_std,
                                    content_img, style_img, input_img, type="VGG19", style_weight=selected_weight, num_steps=selected_steps)
        with col3:
            imshow(output, title=f"Количество шагов: {selected_steps}, вес cтиля: {selected_weight:.0e}")

    elif selected_model == "ResNet50":
        output = run_style_transfer(resnet_cnn, cnn_normalization_mean, cnn_normalization_std,
                                    content_img, style_img, input_img, type="ResNet50", style_weight=selected_weight, num_steps=selected_steps)
        with col3:
            imshow(output, title=f"Количество шагов: {selected_steps}, вес cтиля: {selected_weight:.0e}")

    st.success("Изображения загружены и обработаны!")

    # Разделитель перед вторым рядом
    st.markdown("---")

    # Модели и веса для второго вывода
    models = {"VGG19": vgg_cnn, "ResNet50": resnet_cnn}
    models_weights = {"VGG19": [1e4, 1e5, 1e6, 1e7, 1e8], "ResNet50": [1e7, 1e8, 1e9, 1e10, 1e11]}
    order = [selected_model] + [m for m in models if m != selected_model]  # Сначала выбранная, затем вторая

    # Функция для вывода результатов
    def display_results(model_name):
        st.subheader(f"Результаты с разными style_weight для {model_name} (300 шагов)")
        
        with st.container():
            columns = st.columns(5)

            for idx, weight in enumerate(models_weights[model_name]):
                input_img = content_img.clone()
                output = run_style_transfer(
                    models[model_name], cnn_normalization_mean, cnn_normalization_std,
                    content_img, style_img, input_img, style_weight=weight, type=model_name
                )
                with columns[idx]:  
                    imshow(output, f"Style_weight: {weight:.0e}")

    # Сначала выполняем выбранную модель, затем оставшуюся
    for model in order:
        display_results(model)
