### Моделирование случайных событий

**Часть 1:**  
Приложение отвечает на вопрос пользователя словами Да/Нет с вероятностью 50 на 50. В качестве генератора случайных чисел - встроенный метод random.random()

**Часть 2:**  
Приложение отвечает на вопрос пользователя одной из заранее заготовленных фраз с вероятностью 0.125 каждой:
```python
answers = [
    "Definitely yes",
    "Probably yes",
    "Most likely",
    "Ask again later",
    "Unclear",
    "Probably not",
    "Very doubtful",
    "Definitely no"
]
```

Выбор события(ответа) из группы событий(ответов) происходит по следующему алгоритму
```python
def choice(options):
    epsilon = random.random()
    A = 1.
    k = 1
    while True:
        option, Pk = options[k-1]
        A = A - Pk
        if A <= epsilon:
            return option
        k += 1
```
