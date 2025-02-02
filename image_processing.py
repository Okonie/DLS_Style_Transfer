import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import torch

# Масштабирование изображения с сохранением пропорций
def resize_with_aspect_ratio(image, max_size):
    w, h = image.size
    scale = max_size / max(w, h) if max(w, h) > max_size else 1.0  # Сохраняем пропорции
    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size, Image.LANCZOS), new_size

# Загрузка контентного изображения
def load_content_image(image, max_size, device):
    image, new_size = resize_with_aspect_ratio(image, max_size)
    transform = transforms.ToTensor()  # Преобразуем в тензор
    return transform(image).unsqueeze(0).to(device, torch.float), new_size

# Загрузка стилевого изображения с подгонкой под размер контента
def load_style_image(image, target_size, device):
    image = image.resize(target_size, Image.LANCZOS)  # Принудительно подгоняем стиль под контент
    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0).to(device, torch.float)

# Преобразование тензора в изображение для отображения
unloader = transforms.ToPILImage()

def imshow(tensor, title=None):
    """Отображает изображение из тензора в Streamlit"""
    image = tensor.cpu().clone()  # Клонируем, чтобы не изменять оригинал
    image = image.squeeze(0)      # Убираем лишнее измерение батча
    image = unloader(image)       # Преобразуем тензор в изображение
    st.image(image, caption=title if title else "Результат", use_container_width=True)  # Выводим в Streamlit
