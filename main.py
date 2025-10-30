import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random

print("🚀 ЗАПУСК ПРОЕКТА")
print("=" * 50)

# Девайс и сиды для воспроизводимости
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# CNN модель
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        
        # Автоматический расчет размера для полносвязного слоя
        self._fc_size = None
        self.fc1 = None
        self.fc2 = nn.Linear(128, 10)
        
        # Регистрируем хук для последнего сверточного слоя
        self.gradients = None
        self.activations = None
        self.conv3.register_forward_hook(self.forward_hook)
        # register_backward_hook устарел; используем full_backward_hook
        self.conv3.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def _calculate_fc_size(self, x):
        # Пробный forward pass для расчета размера
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = self.dropout1(x)
        return x.view(x.size(0), -1).size(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 28x28 -> 14x14
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 14x14 -> 7x7
        x = F.relu(self.conv3(x))  # 7x7
        x = self.dropout1(x)
        
        # Динамическое создание fc1 при первом запуске
        if self.fc1 is None:
            fc_size = x.view(x.size(0), -1).size(1)
            self.fc1 = nn.Linear(fc_size, 128).to(x.device)
            self._fc_size = fc_size
        
        x = x.view(-1, self._fc_size)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x  # Возвращаем логиты, без softmax

# Настоящая реализация Grad-CAM
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
    def generate_cam(self, images, target_class=None):
        batch_size = images.shape[0]
        cam_maps = []
        
        for i in range(batch_size):
            image = images[i].unsqueeze(0).to(device)
            image.requires_grad_()
            
            # Forward pass
            output = self.model(image)
            
            # Выбираем класс для текущего изображения
            if target_class is None:
                class_idx = output.argmax(dim=1).item()
            else:
                class_idx = (target_class[i] if isinstance(target_class, (list, tuple, np.ndarray)) else int(target_class))
            
            # Backward pass для целевого класса
            self.model.zero_grad()
            one_hot = torch.zeros_like(output)
            one_hot[0, class_idx] = 1
            output.backward(gradient=one_hot)
            
            # Получаем градиенты и активации
            gradients = self.model.gradients[0].detach().cpu().numpy()
            activations = self.model.activations[0].detach().cpu().numpy()
            
            # Вычисляем weights (средние градиенты по пространственным измерениям)
            weights = np.mean(gradients, axis=(1, 2))
            
            # Вычисляем CAM
            cam = np.zeros(activations.shape[1:], dtype=np.float32)
            for j, w in enumerate(weights):
                cam += w * activations[j]
            
            # Применяем ReLU и ресайзим к размеру изображения
            cam = np.maximum(cam, 0)
            if cam.max() > 0:
                cam = cam / cam.max()  # Нормализация
            else:
                cam = np.zeros_like(cam)
            
            # Ресайз до размера исходного изображения с помощью torch
            cam_tensor = torch.from_numpy(cam).float().unsqueeze(0).unsqueeze(0)
            cam_resized = F.interpolate(cam_tensor, size=(28, 28), mode='bilinear', align_corners=False)
            cam = cam_resized.squeeze().numpy()
            
            cam_maps.append(cam)
            
        return np.array(cam_maps)

# Функция обучения
def train_model():
    print("Загружаем данные MNIST...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Загружаем данные
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)
    
    # Создаем и обучаем модель
    model = ImprovedCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    
    print("Обучаем модель...")
    
    # Инициализируем fc1 с помощью пробного forward pass
    with torch.no_grad():
        dummy_input = torch.randn(1, 1, 28, 28, device=device)
        _ = model(dummy_input)
    
    for epoch in range(10):  # Больше эпох для лучшей точности
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Эпоха {epoch+1}, Батч {batch_idx}, Loss: {loss.item():.4f}')
        
        scheduler.step()
        
        # Валидация
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Эпоха {epoch+1}, Точность: {accuracy:.2f}%')
        
        if accuracy >= 80.0:  # Проверка необходимй точности
            break
    
    # Финальное тестирование
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\n✅ Обучение завершено!')
    print(f'Финальная точмность на тесте: {accuracy:.2f}%')
    
    # Сохраняем модель
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    
    return model

def apply_mask(image, mask, threshold=0.3):
    """Применяет маску к изображению"""
    image_np = image.squeeze().cpu().numpy()
    mask_binary = mask > threshold
    masked_image = image_np * mask_binary
    return torch.from_numpy(masked_image).unsqueeze(0).float()

# Функция визуализации
def visualize_results(model, num_images=5):
    print("\n🎨 Создаем визуализации...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=num_images, shuffle=True)
    
    # Берем несколько тестовых изображений
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images = images.to(device)
    labels = labels.to(device)
    
    # Получаем предсказания (логиты)
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        probabilities = F.softmax(outputs, dim=1)
    
    # Создаем Grad-CAM
    gradcam = GradCAM(model)
    cam_maps = gradcam.generate_cam(images)
    
    # Визуализируем
    fig, axes = plt.subplots(num_images, 4, figsize=(15, 3 * num_images))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        # Исходное изображение
        axes[i, 0].imshow(images[i].detach().cpu().squeeze(), cmap='gray')
        axes[i, 0].set_title(f'Истина: {labels[i].item()}\nПредсказание: {predictions[i].item()}\nВероятность: {probabilities[i, predictions[i]].item():.3f}')
        axes[i, 0].axis('off')
        
        # Heatmap
        axes[i, 1].imshow(cam_maps[i], cmap='jet')
        axes[i, 1].set_title('Grad-CAM Heatmap')
        axes[i, 1].axis('off')
        
        # Наложение
        axes[i, 2].imshow(images[i].detach().cpu().squeeze(), cmap='gray')
        axes[i, 2].imshow(cam_maps[i], cmap='jet', alpha=0.5)
        axes[i, 2].set_title('Grad-CAM наложение')
        axes[i, 2].axis('off')
        
        # Маскированное изображение
        masked_image = apply_mask(images[i].detach().cpu(), cam_maps[i])
        axes[i, 3].imshow(masked_image.squeeze(), cmap='gray')
        axes[i, 3].set_title('Маскированное изображение')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_results.png', dpi=120, bbox_inches='tight')
    plt.show()
    
    return images.cpu(), labels.cpu(), predictions.cpu(), cam_maps

# Расчет Fidelity (правильная реализация)
def calculate_fidelity(model, images, labels, cam_maps):
    print("\nВычисляем Fidelity...")
    
    model.eval()
    fidelities = []
    
    with torch.no_grad():
        for i in range(len(images)):
            img = images[i].unsqueeze(0).to(device)
            # Класс берём предсказанный моделью на исходном изображении
            output_orig_all = model(img)
            class_idx = output_orig_all.argmax(dim=1).item()
            logit_orig = output_orig_all[0, class_idx].item()
            
            # Маскированное изображение (на CPU), перенесём на девайс
            masked_image = apply_mask(images[i], cam_maps[i]).unsqueeze(0).to(device)
            output_masked_all = model(masked_image)
            logit_masked = output_masked_all[0, class_idx].item()
            
            # Fidelity = f(x) - f(x ⊙ m) для того же класса
            fidelity = logit_orig - logit_masked
            fidelities.append(fidelity)
            
            print(f'Изображение {i+1}: Fidelity = {fidelity:.4f} '
                  f'(logit_orig: {logit_orig:.4f}, logit_masked: {logit_masked:.4f})')
    
    mean_fid = np.mean(fidelities)
    var_fid = np.var(fidelities)
    
    print(f'\nСредняя Fidelity: {mean_fid:.4f}')
    print(f'Дисперсия Fidelity: {var_fid:.4f}')
    
    return fidelities, mean_fid, var_fid

# Главная функция
def main():

    
    # Пытаемся загрузить модель или обучаем новую
    try:
        model = ImprovedCNN().to(device)
        
        # Инициализируем модель перед загрузкой весов
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 28, 28, device=device)
            _ = model(dummy_input)
        
        # Загружаем веса (с учётом устройства)
        state = torch.load('mnist_cnn.pth', map_location=device)
        model.load_state_dict(state)
        print("✅ Загружена сохраненная модель")
        
        # Проверяем точность загруженной модели
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)
        
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Точность загруженной модели: {accuracy:.2f}%')
        
        if accuracy < 80.0:
            print('Точность ниже 80%, переобучаем модель...')
            model = train_model()
            
    except Exception as e:
        print(f"Обучаем новую модель. Причина: {e}")
        model = train_model()
    
    # Визуализация
    images, labels, predictions, cam_maps = visualize_results(model, 5)
    
    # Расчет метрик
    fidelities, mean_fid, var_fid = calculate_fidelity(model, images, labels, cam_maps)
    
    # Финальный результат
    print(f"Средняя Fidelity: {mean_fid:.4f}")
    print(f"Дисперсия Fidelity: {var_fid:.4f}")
    print("Результаты сохранены в 'gradcam_results.png'")

if __name__ == "__main__":
    main()