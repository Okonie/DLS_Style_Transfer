import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import streamlit as st

# Функция для вычисления потерь контента
class ContentLoss(nn.Module):

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # Отсоединяем целевое изображение, чтобы оно не участвовало в вычислениях градиентов
        self.target = target.detach()

    def forward(self, input):
        # Вычисляем MSE loss между входом и целевым изображением
        self.loss = F.mse_loss(input, self.target)
        return input

# Функция для вычисления Грам-матрицы (стиль)
def gram_matrix(input):
    a, b, c, d = input.size()  
    features = input.view(a * b, c * d)  # Ресайзим для вычисления Грам-матрицы
    G = torch.mm(features, features.t())  # Вычисляем скалярное произведение
    return G.div(a * b * c * d)  # Нормализуем матрицу

# Функция для вычисления потерь стиля
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        # Отсоединяем целевую Грам-матрицу
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        # Вычисляем Грам-матрицу для входного изображения
        G = gram_matrix(input)
        # Вычисляем MSE loss между Грам-матрицами
        self.loss = F.mse_loss(G, self.target)
        return input

# Модуль для нормализации изображений перед подачей в нейронную сеть
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)  # Нормализуем по каналам
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # Нормализуем входное изображение
        return (img - self.mean) / self.std

# Модели для слоев контента и стиля по умолчанию
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# Создание модели с потерями для контента и стиля
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    normalization = Normalization(normalization_mean, normalization_std)

    content_losses = []  # Список потерь для контента
    style_losses = []    # Список потерь для стиля

    model = nn.Sequential(normalization)  # Начинаем с нормализации

    i = 0  # Индекс для подсчета слоев
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)  # Заменяем inplace версию
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError(f'Неизвестный слой: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Обрезаем модель после последних потерь контента и стиля
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

# Оптимизатор для входного изображения
def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img])  # Используем LBFGS для оптимизации
    return optimizer

# Функция для выполнения передачи стиля
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1e5, content_weight=1, type="VGG19"):
    print('Building the style transfer model..')
    progress_bar = st.progress(0)  # Инициализация progress bar
    status_text = st.empty()  # Место для текста статуса
    status_text.text("Обучение началось")

    # В зависимости от выбранной модели, получаем модель и потери
    if type == "VGG19":
        model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img)
    elif type == "ResNet50":
        model, style_losses, content_losses = get_resnet_with_losses(cnn, normalization_mean, normalization_std, style_img, content_img)

    input_img.requires_grad_(True)  # Разрешаем градиенты для входного изображения
    model.eval()  # Переводим модель в режим оценки
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)  # Ограничиваем значения пикселей

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            # Суммируем потери стиля и контента
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            # Обновляем progress bar
            progress_bar.progress((run[0]) / (num_steps + 20))
            run[0] += 1

            if run[0] % 50 == 0:
                print(f"Run {run} Style Loss : {style_score.item():.4f} Content Loss: {content_score.item():.4f}")
                status_text.text(f'Run {run} Style Loss : {style_score.item():.2f} Content Loss: {content_score.item():.2f}')  # Обновляем текст статуса

            return style_score + content_score

        optimizer.step(closure)

    # Финальная коррекция изображения
    with torch.no_grad():
        input_img.clamp_(0, 1)
    status_text.text("✅ Обучение завершено!")
    return input_img

# Определяем слои для ResNet50
resnet50_style_layers = [1, 2, 3, 4, 5]
resnet50_content_layers = [4]

def get_resnet_with_losses(resnet50, normalization_mean, normalization_std, style_img, content_img,
                           style_layers=resnet50_style_layers, content_layers=resnet50_content_layers):
    resnet = resnet50
    normalization = Normalization(normalization_mean, normalization_std)
    model = nn.Sequential(normalization)
    content_losses = []
    style_losses = []

    i = 0  # Счётчик слоев
    for layer in resnet.children():
        if i >= max(style_layers + content_layers):
            break
        if isinstance(layer, nn.Sequential):  # Обработка блоков ResNet
            for bottleneck in layer:
                i += 1
                model.add_module(f'bottleneck_{i}', bottleneck)

                # Добавляем потери для стиля
                if i in style_layers:
                    target_feature = model(style_img).detach()
                    style_loss = StyleLoss(target_feature)
                    model.add_module(f'style_loss_{i}', style_loss)
                    style_losses.append(style_loss)

                # Добавляем потери для контента
                if i in content_layers:
                    target_feature = model(content_img).detach()
                    content_loss = ContentLoss(target_feature)
                    model.add_module(f'content_loss_{i}', content_loss)
                    content_losses.append(content_loss)
        else:
            model.add_module(str(len(model)), layer)

    return model, style_losses, content_losses
