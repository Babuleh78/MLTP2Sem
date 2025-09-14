import torch
import torchvision.transforms.functional as F
import random
from torchvision import transforms

class Noise: # Шумы 
    
    def __init__(self, mean=0.0, stddev=0.1): 
        self.mean = mean 
        self.stddev = stddev

    def __call__(self, tensor):
        # Шумы той же размерности, что и входной тензор
        noise = torch.zeros_like(tensor).normal_(self.mean, self.stddev)  
        return tensor + noise
    
    def __repr__(self):
        return f"Noise(mean={self.mean}, stddev={self.stddev})"

class RandomRotation: # Небольшие вращения
    
    def __init__(self, angle_range=15): # 15 градусов
        self.angle_range = angle_range

    def __call__(self, img):
        angle = random.uniform(-self.angle_range, self.angle_range)
        return F.rotate(img, angle)
    
    def __repr__(self):
        return f"RandomRotation(angle_range={self.angle_range})"

class RandomShift:
    
    def __init__(self, max_random_shift=8): 
        self.max_shift = max_random_shift # Макс кол-во пикселей для сдвига (поставил 8, тк изображения цифр 28 на 28)

    def __call__(self, img):
        
        dx = random.randint(-self.max_shift, self.max_shift) #Сдвиг по горизонтали
        dy = random.randint(-self.max_shift, self.max_shift) # Сдвиг по вертикали
        
        shifted = torch.zeros_like(img) # Изображение черного цвета такого же размера
        
        h, w = img.shape[1], img.shape[2]
        
        #Не хотим выйти за рамки изображения (массива)
        src_x_start = max(0, -dx)
        src_x_end = min(w, w - dx)
        src_y_start = max(0, -dy)
        src_y_end = min(h, h - dy)
        # Все так же не хотим выйти за рамки изображения (массива)
        dst_x_start = max(0, dx)
        dst_x_end = min(w, w + dx)
        dst_y_start = max(0, dy)
        dst_y_end = min(h, h + dy)
        
        if src_x_end > src_x_start and src_y_end > src_y_start: # Проверка, что мы не уехали за границу полностью
            shifted[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = img[:, src_y_start:src_y_end, src_x_start:src_x_end]
        
        return shifted

    def __repr__(self):
        return f"RandomShift(max_random_shift={self.max_shift})"


 