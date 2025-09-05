import numpy as np


class LinearRegression:
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
        n_iter_no_change=4,  # эпохи без улучшения функции потерь
        shuffle=False,  # перемешивание данных перед каждой эпохой
        batch_size=56  # размер обучающих примеров
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
        self.batch_size = batch_size
        self._coef = None
        self._intercept = None

    def get_penalty_grad(self):  # возвращает градиент регуляризации
        if self.penalty == "l2":  # градиент для L2-регуляризации
            return 2 * self.alpha * self._coef
        elif self.penalty == "l1":  # градиент для L1-регуляризации
            return self.alpha * np.sign(self._coef)
        else:  # Без регуляризации
            return np.zeros_like(self._coef)

    def fit(self, x, y):  # обучает модель
        x_np = np.array(x)
        if x_np.ndim == 1:
            x_np = x_np.reshape(-1, 1)  # приведение массива x к требуемому виду

        y_np = np.array(y)
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)  # приведение массива y к требуемому виду

        if self.random_state is not None:
            np.random.seed(self.random_state)
        self._coef = np.random.randn(1, x_np.shape[1]) * 0.01  # случайная инициализация
        self._intercept = -1  # изначальное значение свободного коэффициента

        if self.early_stopping:  # деление массива на обучающую и валидационную выборку, если это необходимо
            validation_size = int(self.validation_fraction * len(x_np))  # размер валидационной выборки
            x_train, x_val = x_np[:-validation_size], x_np[-validation_size:]
            y_train, y_val = y_np[:-validation_size], y_np[-validation_size:]
        else:
            x_train, y_train = x_np, y_np

        best_loss = float('inf')  # функция потерь
        no_improvement_count = 0  # эпохи без улучшения

        for iter in range(self.max_iter):  # проходимся по эпохам
            if self.shuffle:  # Перемешиваем данные при необходимости
                indices = np.random.permutation(len(x_train))
                x_train = x_train[indices]
                y_train = y_train[indices]

            num_batches = int(np.ceil(len(x_train) / self.batch_size))  # количество батчей

            for i in range(num_batches):  # цикл по батчам
                start_index = i * self.batch_size
                end_index = min((i + 1) * self.batch_size, len(x_train))
                x_batch = x_train[start_index:end_index]
                y_batch = y_train[start_index:end_index]

                y_pred = np.dot(x_batch, self._coef.T) + self._intercept

                grad_coef = -2 * np.dot(x_batch.T, (y_batch - y_pred)) / len(x_batch)  # Вычисляем градиенты
                grad_intercept = -2 * np.sum(y_batch - y_pred) / len(x_batch)

                grad_coef += self.get_penalty_grad().T  # Добавляем градиент регуляризации

                self._coef -= self.eta0 * grad_coef.T  # Обновляем вес для k
                self._intercept -= self.eta0 * grad_intercept  # Обновляем вес для b

            y_pred_train = np.dot(x_train, self._coef.T) + self._intercept  # Вычисляем функцию потерь
            loss = np.mean((y_train - y_pred_train) ** 2)  # MSE

            if self.early_stopping:  # обрабатываем раннюю остановку
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
                best_loss = loss  # Если ранняя остановка отключена, используем loss

    def predict(self, x):
        x_np = np.array(x)
        if x_np.ndim == 1:
            x_np = x_np.reshape(-1, 1)
        return np.dot(x_np, self._coef.T) + self._intercept

    @property
    def coef_(self):  # возвращает веса модели
        return self._coef

    @property
    def intercept_(self):  # возвращает b
        return self._intercept

    @coef_.setter
    def coef_(self, value):
        self._coef = value

    @intercept_.setter
    def intercept_(self, value):
        self._intercept = value
