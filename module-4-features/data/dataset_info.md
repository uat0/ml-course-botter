# Online Shoppers Purchasing Intention

Источник: [UCI №468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset).
Авторы: C. Sakar, Y. Kastro (2018). DOI: https://doi.org/10.24432/C5F88Q.
Лицензия: CC BY 4.0. Скачано 02.11.2026.

Исходный CSV: 12 330 строк, 17 входных столбцов + Revenue. Класс False — 10 422
(84.5255%), True — 1 908 (15.4745%). Пропусков нет, полных дублей — 125.
CSV сохранён без изменений; в эксперименте удаляются точные дубли перед разделением.
SHA-256: `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9`.

| Поля | Значение |
|---|---|
| Administrative / Informational / ProductRelated | Число страниц по типам |
| Соответствующие *_Duration | Время просмотра в секундах |
| BounceRates / ExitRates | Показатели отказов и выходов |
| PageValues | Средняя ценность посещённых страниц |
| SpecialDay | Близость специального дня |
| Month | Месяц: Feb, Mar, May, June, Jul, Aug, Sep, Oct, Nov, Dec |
| OperatingSystems / Browser / Region / TrafficType | Номинальные идентификаторы |
| VisitorType | Returning_Visitor, New_Visitor, Other |
| Weekend | Сессия в выходной |
| Revenue | Покупка совершена / не совершена |

Ограничения: нет точного времени, идентификаторов последовательных событий, текстов или
координат. Регион — код, поэтому из него нельзя вычислять расстояние Haversine.
PageValues и агрегаты завершённой сессии требуют проверки доступности до момента
решения перед использованием модели в реальном времени.

`example_sessions.csv` — три сессии из test для проверки сериализации; это не отдельная
валидационная выборка. Полное разделение сохранено в `models/split_indices.json`.
