"""Расчёт признаков сессии перед вызовом сохранённого Pipeline."""
import numpy as np

MONTH_MAP = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6,
             'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

def engineer_features(frame):
    # Копия предотвращает изменение исходных данных при повторном вызове.
    out = frame.drop(columns=['Revenue'], errors='ignore').copy()
    out['Weekend'] = out['Weekend'].astype(int)
    out['TotalPages'] = out[['Administrative', 'Informational', 'ProductRelated']].sum(axis=1)
    out['TotalDuration'] = out[['Administrative_Duration', 'Informational_Duration',
                               'ProductRelated_Duration']].sum(axis=1)
    # Сглаживание знаменателя исключает деление на ноль.
    out['AvgPageValue'] = out['PageValues'] / (out['TotalPages'] + 1)
    out['BounceExitRatio'] = out['BounceRates'] / (out['ExitRates'] + 0.001)
    out['IsNewVisitor'] = out['VisitorType'].eq('New_Visitor').astype(int)
    out['IsWeekend'] = out['Weekend']
    # Неизвестный месяц станет NaN и будет обработан обученным imputer.
    out['MonthNum'] = out['Month'].map(MONTH_MAP)
    out['IsHighTraffic'] = out['TrafficType'].gt(10).astype(int)
    out['DurationPerPage'] = out['TotalDuration'] / (out['TotalPages'] + 1)
    out['ProductPageShare'] = out['ProductRelated'] / out['TotalPages'].clip(lower=1)
    out['ProductTimeShare'] = out['ProductRelated_Duration'] / out['TotalDuration'].clip(lower=1)
    # Временные и логарифмические преобразования используют только значения текущей строки.
    out['LogTotalDuration'] = np.log1p(out['TotalDuration'])
    out['MonthSin'] = np.sin(2 * np.pi * out['MonthNum'] / 12)
    out['MonthCos'] = np.cos(2 * np.pi * out['MonthNum'] / 12)
    return out
