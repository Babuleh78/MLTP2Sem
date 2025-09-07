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

    def get_penalty_grad(self):  
        if self.penalty == "l2":  
            return 2 * self.alpha * self._coef
        elif self.penalty == "l1":  
            return self.alpha * np.sign(self._coef)
        else:
            return 0  

    def predict(self, x):
        x_np = np.array(x)
        if x_np.ndim == 1:
            x_np = x_np.reshape(-1, 1)
        return np.dot(x_np, self._coef.T) + self._intercept

  def fit(self, x, y):  
        x_np = np.array(x)
        if x_np.ndim == 1:
            x_np = x_np.reshape(-1, 1) 

        y_np = np.array(y)
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1) 

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self._coef = np.random.randn(1, x_np.shape[1]) * 0.01  
        self._intercept = np.random.randn(1) * 0.01  

        if self.early_stopping:  
            validation_size = int(self.validation_fraction * len(x_np))  
            x_train, x_val = x_np[:-validation_size], x_np[-validation_size:]
            y_train, y_val = y_np[:-validation_size], y_np[-validation_size:]
        else:
            x_train, y_train = x_np, y_np

        best_loss = float('inf')
        no_improvement_count = 0

        for iter in range(self.max_iter):
            if self.shuffle:  
                indices = np.random.permutation(len(x_train))
                x_train_shuffled = x_train[indices]
                y_train_shuffled = y_train[indices]
            else:
                x_train_shuffled = x_train
                y_train_shuffled = y_train

            y_pred = np.dot(x_train_shuffled, self._coef.T) + self._intercept

            grad_coef = -2 * np.dot(x_train_shuffled.T, (y_train_shuffled - y_pred)) / len(x_train_shuffled)
            grad_intercept = -2 * np.sum(y_train_shuffled - y_pred) / len(x_train_shuffled)

            grad_coef += self.get_penalty_grad().T

            self._coef -= self.eta0 * grad_coef.T
            self._intercept -= self.eta0 * grad_intercept

            y_pred_train = np.dot(x_train, self._coef.T) + self._intercept
            loss = np.mean((y_train - y_pred_train) ** 2)

       
            if self.early_stopping: # Если включена ранняя остановка
                y_pred_val = np.dot(x_val, self._coef.T) + self._intercept
                val_loss = np.mean((y_val - y_pred_val) ** 2)

                if val_loss < best_loss - self.tol:
                    best_loss = val_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= self.n_iter_no_change:
                    break
            else:
                if loss < best_loss - self.tol:
                    best_loss = loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= self.n_iter_no_change:
                    break

            if iter % 100 == 0:
                print(f"Итерация {iter}, Ошибка: {loss:.6f}")

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
