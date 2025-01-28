from PIL import Image
import torchvision.transforms as transforms
import torch
import streamlit as st

# Функция загрузки изображения с сохранением пропорций
def resize_with_aspect_ratio(image, max_size):
    w, h = image.size
    scale = max_size / max(w, h) if max(w, h) > max_size else 1.0
    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size, Image.LANCZOS), new_size

# Функция загрузки контентного изображения
def load_content_image(image, max_size, device):
    image, new_size = resize_with_aspect_ratio(image, max_size)
    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0).to(device, torch.float), new_size

# Функция загрузки стилевого изображения
def load_style_image(image, target_size, device):
    image = image.resize(target_size, Image.LANCZOS)
    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0).to(device, torch.float)

# Преобразование тензора в изображение
unloader = transforms.ToPILImage()

def imshow(tensor, title=None):
    """Отображает тензорное изображение в Streamlit."""
    image = tensor.cpu().clone()  # Клонируем тензор, чтобы не изменять его
    image = image.squeeze(0)      # Убираем фиктивное измерение батча
    image = unloader(image)       # Преобразуем в PIL-изображение
    
    # Вывод изображения в Streamlit
    st.image(image, caption=title if title else "Результат", use_container_width=True)