import numpy as np


class CustomLinearRegression:
    def __init__(
        self,
        *,
        penalty="l2",  # тип регуляризации
        alpha=0.01,  # коэф регуляризации
        max_iter=1000,  # максимальное количество эпох
        tol=0.01,  # порог сходимости
        random_state=None,
        eta0=0.2,  # начальная скорость обучения
        early_stopping=True,  # ранняя остановка для предотвращения переобучения
        validation_fraction=0.2,  # доля данных для валидации
        n_iter_no_change=8,  # эпохи без улучшения функции потерь
        shuffle=False,  # перемешивание данных перед каждой эпохой
    ):
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.eta0 = eta0
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.shuffle = shuffle
        self._coef = None
        self._intercept = None

    def get_params(self, deep=True):
        return {
            'penalty': self.penalty,
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'random_state': self.random_state,
            'eta0': self.eta0,
            'early_stopping': self.early_stopping,
            'validation_fraction': self.validation_fraction,
            'n_iter_no_change': self.n_iter_no_change,
            'shuffle': self.shuffle
        }

    def _prepare_data(self, x, y):
        x_np = np.array(x)
        if x_np.ndim == 1:
            x_np = x_np.reshape(-1, 1) 

        y_np = np.array(y)
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)
            
        return x_np, y_np

    def _initialize_weights(self, n_features):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self._coef = np.random.randn(1, n_features) * 0.01  
        self._intercept = np.random.randn(1) * 0.01

    def _split_validation_data(self, x, y):
        if self.early_stopping:  
            
            validation_size = int(self.validation_fraction * len(x))  
            
            x_train, x_val = x[:-validation_size], x[-validation_size:]
            y_train, y_val = y[:-validation_size], y[-validation_size:]
            
            return x_train, y_train, x_val, y_val
        else:
            return x, y, None, None

    def _shuffle_data(self, x, y):
        if self.shuffle:  
            indices = np.random.permutation(len(x))
            return x[indices], y[indices]
        return x, y

    def _compute_gradients(self, x_batch, y_batch, y_pred):
        
        grad_coef = -2 * np.dot(x_batch.T, (y_batch - y_pred)) / len(x_batch)
        
        grad_intercept = -2 * np.sum(y_batch - y_pred) / len(x_batch)
        grad_coef += self.get_penalty_grad().T
        
        return grad_coef, grad_intercept

    def _update_weights(self, grad_coef, grad_intercept):
        
        self._coef -= self.eta0 * grad_coef.T
        self._intercept -= self.eta0 * grad_intercept

    def _compute_loss(self, x, y):
        
        y_pred = self.predict(x)
        return np.mean((y - y_pred) ** 2)

    def _check_early_stopping(self, current_loss, best_loss, no_improvement_count, is_validation=True):
       
        if current_loss < best_loss - self.tol:
            return current_loss, 0, False  
        else:
            no_improvement_count += 1
            if no_improvement_count >= self.n_iter_no_change:
                print(f"Ранняя остановка на итерации: достигнут предел без улучшения")
                return best_loss, no_improvement_count, True  # Останавливаем
            return best_loss, no_improvement_count, False  # Продолжаем

    def get_penalty_grad(self):  
        
        if self.penalty == "l2":  
            return 2 * self.alpha * self._coef
        elif self.penalty == "l1":  
            return self.alpha * np.sign(self._coef)
        else:
            return 0  

    def fit(self, x, y):  
        x_np, y_np = self._prepare_data(x, y)
        
        self._initialize_weights(x_np.shape[1])
        
        x_train, y_train, x_val, y_val = self._split_validation_data(x_np, y_np)

        best_loss = float('inf')
        no_improvement_count = 0

        # Основной цикл обучения
        for iter in range(self.max_iter):
            # Перемешивание
            x_shuffled, y_shuffled = self._shuffle_data(x_train, y_train)
            # Прямой проход
            y_pred = self.predict(x_shuffled)
            # Градиенты
            grad_coef, grad_intercept = self._compute_gradients(x_shuffled, y_shuffled, y_pred)
            # Обновляеи веса
            self._update_weights(grad_coef, grad_intercept)
            # Считаем ошибку
            current_loss = self._compute_loss(x_train, y_train)
      
            if self.early_stopping:
                val_loss = self._compute_loss(x_val, y_val)
                best_loss, no_improvement_count, should_stop = self._check_early_stopping(val_loss, best_loss, no_improvement_count, True)
            else:
                best_loss, no_improvement_count, should_stop = self._check_early_stopping(current_loss, best_loss, no_improvement_count, False)
                
            if should_stop:
                break

    def predict(self, x):
        x_np = np.array(x)
        if x_np.ndim == 1:
            x_np = x_np.reshape(-1, 1)
        return np.dot(x_np, self._coef.T) + self._intercept

    @property
    def coef_(self):
        return self._coef

    @property
    def intercept_(self):
        return self._intercept

    @coef_.setter
    def coef_(self, value):
        self._coef = value

    @intercept_.setter
    def intercept_(self, value):
        self._intercept = value
