## Модель классификации пользовательских интентов

### Архитектура и результаты

Я файн-тьюнила предобученный Берт с добавленным линейным слоем сверху для итоговой классификации. Такой выбор был обоснован тем, что в репозитории с датасетом лежала [статья](https://arxiv.org/pdf/2003.04807.pdf), в которой авторы рассматривали различные архитектуры для решения данной задачи. И Берт с файн-тьюнингом показал весьма хорошие результаты в сравнении с остальными моделями.

С чем можно было поэксперементировать:
1. Размер батча (опять же в статье увеличение размера батча приводило к большей точности);
2. Сэмплинг батчей (чтобы решить проблему несбалансированности классов);
3. Выбор лосс-функции и, конечно же, выбор архитектуры.

К сожалению, мне не хватило времени даже на то, чтобы качественно обучить свою модель:(
В итоге на тесте f1-score составлял примерно 0.6

### Структура репозитория

В директории *jupyter notebook* находятся два блокнота. В *experiment.ipynb* описан процесс эксперимента, начиная от небольшой предобработки данных, далее построение и обучение модели. В конце блокнота приведена оценка качества на тестовой выборке. Затем в *example.ipynb* можно увидеть post-запрос, ответом на который являются категории подаваемых текстов (сейчас работает только при локальном запуске app.py, потому что на herokuapp я превысила лимит загружаемых приложений...)

В основной части репозитория содержутся файлы с моделью *model.py*, различными вспомогательными функциями *utils.py* и самим rest сервисом. В целом, приложение готово к использованию, но в идеале мне ещё нужно запилить bash-скрипт, чтобы это всё легко запускалось.
