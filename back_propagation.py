import numpy as np

class MLPRegressor:
    def __init__(self, hidden_layer_sizes=(100,), learning_rate=0.001, max_iter=10):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = []    
        self.biases = []     

    def _init_weights(self, size):
        self.weights = []
        self.biases = []
        # Входной + скрытый + выходной 
        layer_size = [size] + list(self.hidden_layer_sizes) + [1]
        
        for i in range(len(layer_size) - 1):
            # Норм распределение
            w = np.random.normal(0, 1, (layer_size[i], layer_size[i + 1])) 
            # Стандартизация
            std = np.sqrt(2 / (layer_size[i] + layer_size[i + 1]))
            w = w * std
            # Нули
            b = np.zeros((1, layer_size[i + 1]))
            
            self.weights.append(w)
            self.biases.append(b)

    def _update_weights(self, gradients):
        gradients_w, gradients_b = gradients
        
        for i in range(len(self.weights)):
            
            self.weights[i] = self.weights[i] - self.learning_rate * gradients_w[i]
            self.biases[i] = self.biases[i] - self.learning_rate * gradients_b[i]

    def _forward(self, x):
        self.activations = [x]
        
        for i in range(len(self.weights) - 1):
            z = np.matmul(self.activations[-1], self.weights[i]) + self.biases[i]
            
            sigm_value = self._sigmoid(z)
            self.activations.append(sigm_value)
            
        z_output = np.matmul(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.activations.append(z_output)  
        
        return z_output

    def _back(self, x, y_true, y_pred):
        grads_w = []
        grads_b = []

        error = y_pred - y_true  
        n_samples = x.shape[0]

        for i in range(len(self.weights)-1, -1, -1):
            if i == len(self.weights) - 1: 
                grad_z = error  
            else:
                grad_z = grad_a * self._sigmoid_proizv(self.activations[i + 1])
                
            grad_w = np.matmul(self.activations[i].T, grad_z / n_samples)
            grad_b = np.sum(grad_z, axis=0, keepdims=True) / n_samples
            
            grads_w.append(grad_w)
            grads_b.append(grad_b)
            
            if i > 0:
                grad_a = np.matmul(grad_z, self.weights[i].T)
        
        # Разворачиваем градиенты
        return grads_w[::-1], grads_b[::-1]

    def _sigmoid(self, x): # Надо будет узнать, сигмоиду ли использовать
        return 1 / (1 + np.exp(-x))

    def _sigmoid_proizv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def _mse(self, y_true, y_pred):
        error = y_true - y_pred
        return np.mean(error**2)

    
    
    def fit(self, X, y):
        
        self._init_weights(X.shape[1])
        
        for epoch in range(self.max_iter):
            # Прямой проход
            y_pred = self._forward(X)
            # Текущая ошибка
            loss = self._mse(y, y_pred)
            # Обратный проход
            gradients = self._back(X, y, y_pred)
            # Обновление весов
            self._update_weights(gradients)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    def predict(self, X):
        current_activation = X
        
        for i in range(len(self.weights) - 1):
            z = np.matmul(current_activation , self.weights[i]) + self.biases[i]
            current_activation = self._sigmoid(z)
        
        z_output = np.matmul(current_activation, self.weights[-1]) + self.biases[-1]
        
        return z_output
