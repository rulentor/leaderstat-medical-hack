# leaderstat-medical-hack
Хакатон Самара. Медицина
# Teaser
Описание решения.

Сервис - мобильное приложение с загрузкой Dataset через интерфейс

Особенности качества Dataset

При извлечении данных было много пропусков и выбросов, которые мы удаляли, что уменьшало размер датасета с 76 тыс с лишним наблюдений до 33 тыс, валидационная выборка уменьшилась с 17 тыс до 11 тыс наблюдений.

Технические особенности.

По представленным показателям мы не нашли явно выраженную корреляцию. Косвенно выявленна корреляция с фильтрацией входных данных по наличию диагноза по МКБ10.

Особенности решения.

Все использованные библиотеки удовлетворяют идеологии opensource и имеют открытую лицензию BSD.

Уникальность.

Реализована идея pipeline, что позволяет масштабировать систему и приспосабливать к конкретной ситуации. Создан прототип системы прогнозирования внутригоспитальной летальности.

# Baseline
Код построения модели
baseline_example_finish_2.py
# Model
Модель
medical.joblib
# LeaderStat UI
Рабочий код user
LeaderStat_UI.py
