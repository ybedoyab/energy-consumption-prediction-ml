# Predicción de consumo energético — Proyecto ML

Proyecto académico de **regresión supervisada** para predecir el consumo de electrodomésticos (`Appliances`, Wh) en un hogar monitorizado, usando sensores ambientales, variables temporales y rezagos del consumo.

## Problema

Estimar el consumo energético cada 10 minutos a partir de condiciones interiores/exteriores y patrones de uso. Se comparan tres modelos de regresión con evaluación en un **split temporal** (más realista que un split aleatorio en series de tiempo).

## Dataset

**[UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)**

| Detalle | Valor |
|---------|--------|
| Archivo | `data/raw/energydata_complete.csv` |
| Variable objetivo | `Appliances` (Wh) |
| Registros | ~19 735 (12 filas iniciales se eliminan tras crear rezagos) |
| Frecuencia | Cada 10 minutos |

## Estructura del proyecto

```
energy-consumption-prediction-ml/
├── data/raw/energydata_complete.csv
├── notebooks/
│   ├── 01_eda_energy_consumption.ipynb
│   ├── 02_model_training_comparison.ipynb
│   ├── 03_results_interpretation.ipynb
│   └── 04_demo_interactiva_poster.ipynb
├── reports/
│   ├── figures/          # Gráficos exportados
│   └── metrics/          # CSV de métricas y predicciones
├── requirements.txt
└── README.md
```

## Cómo ejecutar

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

Ejecutar en orden: **01 → 02 → 03 → 04** (el 04 es opcional para la demo del póster).

El notebook **02** genera `reports/metrics/model_metrics.csv` y `reports/metrics/predictions_best_model.csv`, además de figuras en `reports/figures/`. El **04** requiere haber ejecutado el 02.

## Modelos y métricas

| Modelo | Pipeline |
|--------|----------|
| Ridge | `StandardScaler` + regresión lineal regularizada |
| Decision Tree | Sin escalado |
| Random Forest | Sin escalado |
| Gradient Boosting (opcional) | Sin escalado; activar con `USE_GRADIENT_BOOSTING = True` en el notebook 02 |

**Métricas:** MAE, MSE, RMSE, R² (tabla ordenada por RMSE en conjunto de prueba temporal).

## Resultados principales (split temporal 80/20)

Ejemplo de ejecución con el dataset incluido:

| Modelo | MAE | RMSE | R² |
|--------|-----|------|-----|
| Ridge | 27.7 | 59.8 | 0.57 |
| Random Forest | 53.8 | 87.1 | 0.08 |
| Decision Tree | 43.4 | 87.7 | 0.06 |

El **Ridge** suele ser el más estable en este setup; los árboles capturan no linealidades pero pueden sobreajustar en el tramo final de la serie. Los valores exactos dependen de la ejecución: revisar `reports/metrics/model_metrics.csv` tras correr el notebook 02.

## Feature engineering

- Temporales: `hour`, `day_of_week`, `month`, `is_weekend`, codificación cíclica (`hour_sin/cos`, `day_sin/cos`)
- Clima indoor: `T_inside_mean`, `RH_inside_mean`, `delta_T_out_inside`
- Rezagos: `Appliances_lag_1/3/6`, medias móviles `Appliances_roll_3/6/12` (solo información pasada, sin leakage)

## Limitaciones

- Un solo hogar → generalización limitada a otros edificios.
- Sin búsqueda exhaustiva de hiperparámetros.
- Posible deriva estacional no modelada de forma explícita.
- Gradient Boosting requiere tuning adicional si se usa.

## Mejoras futuras

- Validación cruzada temporal (`TimeSeriesSplit`).
- Features de calendario (festivos, estaciones).
- Comparar con XGBoost/LightGBM bien tuneados.
- Extensión a datasets multi-edificio (p. ej. ASHRAE con muestreo).

## ¿Hace falta más data?

**No** para la entrega actual. Este dataset UCI cubre EDA, tres modelos, métricas, figuras, póster y video.
