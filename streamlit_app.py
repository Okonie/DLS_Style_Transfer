import streamlit as st
import torch
from PIL import Image
from image_processing import *
from neural_transfer import *
from torchvision.models import vgg19, VGG19_Weights

# Определение устройства (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Максимальный размер изображения
imsize = 512 if torch.cuda.is_available() else 128

# Интерфейс Streamlit
st.title("Нейронный стиль")

# Загрузка контентного изображения
content_file = st.file_uploader("Загрузите контентное изображение", type=["png", "jpg", "jpeg"])
style_file = st.file_uploader("Загрузите стилевое изображение", type=["png", "jpg", "jpeg"])

if content_file and style_file:
    content_img_pil = Image.open(content_file).convert("RGB")
    style_img_pil = Image.open(style_file).convert("RGB")


    # Разбиваем экран на 3 колонки
    col1, col2, col3 = st.columns(3)
    
    # Обработка изображений
    content_img, content_size = load_content_image(content_img_pil, imsize, device)
    style_img = load_style_image(style_img_pil, content_size, device)

    # Отображение загруженных изображений
    with col1:
        st.image(content_img_pil, caption="Контентное изображение", use_container_width=True)
    with col2:
        st.image(style_img_pil, caption="Стилевое изображение (подогнано по размеру)", use_container_width=True)

    
    print(device)
    #imshow(style_img, title='Style Image')
    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
    input_img = content_img.clone()
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img)
    with col3:
        imshow(output, title="Результат")
    st.success("Изображения загружены и обработаны!")



