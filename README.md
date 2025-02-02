# Перенос стиля (Neural style transfer) с использованием глубоких нейронных сетей

Ссылка на проект: https://dls-style-transfer.streamlit.app/

Этот проект позволяет выполнить перенос стиля с использованием популярных предобученных моделей глубоких нейронных сетей, таких как VGG19 и ResNet50. Веб-интерфейс, реализованный с помощью **Streamlit**, позволяет пользователю загружать контентное и стилевое изображения, а затем выбирать модель, вес стиля и количество шагов для оптимизации. После этого производится процесс переноса стиля, и результат отображается на экране.

Подробнее о Neural style transfer можно почитать вот здесь
https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
https://en.wikipedia.org/wiki/Neural_style_transfer

## Основные функции:
- Загрузка контентного и стилевого изображения.
- Выбор модели (VGG19 или ResNet50) (фишка)
- Выбор веса стиля, который регулирует степень влияния стиля на результат (фишка)
- Выбор количества шагов для оптимизации (фишка)
- Отображение результатов в реальном времени
- Вариация весов для получения лучшего переноса стиля

## Используемые технологии:
- **Streamlit** — для создания интерактивного веб-интерфейса.
- **PyTorch** — для работы с глубокими нейронными сетями и выполнения переноса стиля.
- **VGG19** и **ResNet50** — предобученные модели для переноса стиля.
- **PIL (Python Imaging Library)** — для обработки изображений.

## Установка:
Для работы с проектом нужно установить несколько зависимостей. Вы можете использовать **virtualenv** или **conda** для создания изолированного окружения.

1. Клонируйте репозиторий:
    ```bash
    git clone https://github.com/ваш_пользователь/ваш_репозиторий.git
    cd ваш_репозиторий
    ```

2. Установите необходимые зависимости:
    ```bash
    pip install -r requirements.txt
    ```

## Запуск:
Чтобы запустить приложение, выполните команду:
```bash
streamlit run streamlit_app.py
```
## Результаты:
<p align="center">
  <img src="https://github.com/user-attachments/assets/92a2a4f6-7eef-4d6f-a9b5-84798a2aa10e" width="30%" />
  <img src="https://github.com/user-attachments/assets/9fe748b2-e1ed-45a1-b593-23c07099afca" width="30%" />
  <img src="https://github.com/user-attachments/assets/aae034ef-c11a-4900-a371-f787d2be594e" width="30%" />
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/92a2a4f6-7eef-4d6f-a9b5-84798a2aa10e" width="30%" />
  <img src="https://github.com/user-attachments/assets/11bf0b90-7d29-4bce-a8d6-fbc62ed9d16c" width="30%" />
  <img src="https://github.com/user-attachments/assets/09abf8c4-5dc8-45e3-9e11-c8d6aed3ee59" width="30%" />
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/92a2a4f6-7eef-4d6f-a9b5-84798a2aa10e" width="30%" />
  <img src="https://github.com/user-attachments/assets/4a444eee-d9e1-4225-b363-cb68baea3a52" width="30%" />
  <img src="https://github.com/user-attachments/assets/d8525a07-5acd-4f81-957d-6aae74b67e58" width="30%" />
</p>



