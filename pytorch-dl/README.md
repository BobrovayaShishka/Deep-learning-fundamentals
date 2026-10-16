<div align="center">

# 🧠 Deep Learning Fundamentals

**Пять исследовательских мини-проектов: от тензоров и автоградиента до свёрточных сетей, аугментаций и fine-tuning предобученных моделей**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-T4%20GPU-F9AB00?logo=googlecolab&logoColor=white)

</div>

---

## 📖 О репозитории

Репозиторий — учебный курс «Основы глубокого обучения и ИИ», оформленный как набор самостоятельных исследований. В каждой папке: код экспериментов, графики, логи и отчёт `REPORT.md` с таблицами результатов и выводами. Названия папок отражают исследуемый вопрос, а не номер задания.

```text
Тензоры → Автоград → Линейные модели → MLP и регуляризация → CNN → Аугментации и transfer learning
```

## 🗂 Исследования

| Папка | Вопрос исследования | Методы и инструменты | Отчёт |
|-------|---------------------|----------------------|-------|
| [`01_tensors_autograd_gpu_benchmark`](./01_tensors_autograd_gpu_benchmark) | Как устроены тензоры и автоград, и насколько GPU быстрее CPU? | Операции над тензорами, `autograd`, замеры через `torch.cuda.Event` | [REPORT](./01_tensors_autograd_gpu_benchmark/REPORT.md) |
| [`02_linear_models_and_hyperparameters`](./02_linear_models_and_hyperparameters) | Как гиперпараметры и feature engineering влияют на линейные модели? | L1/L2, early stopping, F1 / ROC-AUC, `CustomCSVDataset`, SGD / Adam / RMSprop | [REPORT](./02_linear_models_and_hyperparameters/REPORT.md) |
| [`03_mlp_depth_width_regularization`](./03_mlp_depth_width_regularization) | Как глубина, ширина и регуляризация влияют на MLP? | MNIST, CIFAR-10, Dropout / BatchNorm / L2, grid search, heatmap | [REPORT](./03_mlp_depth_width_regularization/REPORT.md) |
| [`04_cnn_vs_fc_architectures`](./04_cnn_vs_fc_architectures) | Что даёт свёртка и как выбирать архитектуру CNN? | Residual-блоки, размер ядра, глубина, градиенты по слоям, рецептивные поля, кастомные слои | [REPORT](./04_cnn_vs_fc_architectures/REPORT.md) |
| [`05_augmentations_and_transfer_learning`](./05_augmentations_and_transfer_learning) | Как аугментации и предобученные модели работают на малом датасете? | torchvision и кастомные аугментации, `AugmentationPipeline`, ResNet18 / EfficientNet-B0 / MobileNetV3 | [REPORT](./05_augmentations_and_transfer_learning/REPORT.md) |

## 🏆 Ключевые результаты

| Исследование | Результат |
|--------------|-----------|
| **01 · GPU benchmark** | GPU (T4) быстрее CPU в **22–57×** на тензорных операциях; матричное умножение — **43×** |
| **02 · Линейные модели** | Для регрессии лучше всего сработали статистические признаки, для классификации — полиномиальные; лучший оптимизатор — SGD с `lr=0.1` |
| **03 · MLP** | BatchNorm — самая эффективная регуляризация (+2.2% на CIFAR-10); оптимум на MNIST — 2–3 слоя, на CIFAR-10 — 5 слоёв |
| **04 · CNN vs FC** | На CIFAR-10 CNN обходит FC на **+27.7 п.п.** (76.15% против 48.49%); 6-блочная ResNet-подобная сеть — **80.97%** |
| **05 · Transfer learning** | Лучший fine-tuning — **EfficientNet-B0: 91.67%** на val за 10 эпох |

## 🖼 Примеры результатов

<table>
  <tr>
    <td align="center"><b>03 — точность vs глубина (CIFAR-10)</b></td>
    <td align="center"><b>04 — сравнение точности (CIFAR-10)</b></td>
    <td align="center"><b>05 — точность fine-tuned моделей</b></td>
  </tr>
  <tr>
    <td><img src="./03_mlp_depth_width_regularization/plots/depth_experiments/cifar_depth_accuracy.png" width="300"></td>
    <td><img src="./04_cnn_vs_fc_architectures/results/cifar_comparison/CIFAR-10_Test_Accuracy_Comparison.png" width="300"></td>
    <td><img src="./05_augmentations_and_transfer_learning/results/task_6/model_comparison_acc.png" width="300"></td>
  </tr>
</table>

## 🧩 Подробнее

<details>
<summary><b>01 · Тензоры, автоград и CPU vs GPU</b></summary>

- `tensor_basics.py` — создание, индексация и reshape тензоров
- `autograd_basics.py` — градиенты функции трёх переменных, MSE для линейной модели, цепное правило с проверкой через `torch.autograd.grad`
- `cpu_vs_gpu_benchmark.py` — бенчмарк на больших тензорах

</details>

<details>
<summary><b>02 · Линейные модели и гиперпараметры</b></summary>

- `regression_models.py` — `LinearRegressionWithRegularization` (L1/L2, early stopping) и `MulticlassLogisticRegression` с метриками и confusion matrix
- `csv_dataset_experiments.py` — `CustomCSVDataset`: пропуски, кодирование категорий, нормализация, автоопределение типа задачи
- `hyperparameter_and_feature_experiments.py` — `learning_rate`, `batch_size`, оптимизаторы, полиномиальные и статистические признаки
- Данные: `housing.csv` (регрессия), `heart_disease_uci.csv` (классификация)

</details>

<details>
<summary><b>03 · Глубина, ширина и регуляризация MLP</b></summary>

- `depth_experiments.py` — сети глубиной 1–7 слоёв, анализ переобучения
- `width_experiments.py` — конфигурации ширины и grid search
- `regularization_experiments.py` — Dropout, BatchNorm, L2, адаптивная регуляризация, распределения весов
- `utils/` — общие утилиты; `results/` — сырые результаты (`.json`, `.log`)

</details>

<details>
<summary><b>04 · CNN vs FC и архитектурный анализ</b></summary>

- `cnn_vs_fc_comparison.py` — FC, Simple CNN и Residual CNN на MNIST и CIFAR-10
- `cnn_architecture_analysis.py` — размер ядра, глубина, residual-связи, градиенты, рецептивные поля
- `custom_layers_experiments.py` + `models/custom_layers.py` — свёртка с L1, attention, Swish, pooling с индексами

</details>

<details>
<summary><b>05 · Аугментации и transfer learning</b></summary>

- `standard_augmentations.py`, `custom_augmentations.py` — стандартные и кастомные (`AddGaussianNoise`, `RandomErasingCustom`, `ElasticTransform`) аугментации
- `dataset_analyzer.py` — распределение классов, размеры, соотношение сторон
- `augmentation_pipeline.py` — `AugmentationPipeline` с конфигурациями Light / Medium / Heavy
- `image_size_benchmark.py` — влияние размера изображения на время и память
- `finetune_pretrained.py` — fine-tuning ResNet18, EfficientNet-B0 и MobileNetV3-Small на 6 классах

</details>

## 🚀 Запуск

Протокол экспериментов: **train / val / test**. Чекпоинт выбирается по val-accuracy, тестовая метрика считается **один раз**. Каждый прогон повторяется на нескольких seed (по умолчанию 3), в отчёте — **mean ± std**.

```bash
git clone https://github.com/BobrovayaShishka/FoDLaAI.git
cd FoDLaAI
pip install -r requirements.txt

# CPU smoke-тест (без GPU, синтетические данные)
python pytorch-dl/run_experiments.py smoke

# Все основные эксперименты на CUDA, 3 seed
python pytorch-dl/run_experiments.py --device cuda --seeds 42,43,44

# Один эксперимент
python pytorch-dl/run_experiments.py --device cuda --experiments cnn_vs_fc
```

Отдельный скрипт:

```bash
cd pytorch-dl/04_cnn_vs_fc_architectures
python cnn_vs_fc_comparison.py --device cuda --seeds 42,43,44 --epochs 10
```

> [!NOTE]
> - Скрипты запускаются **из папки своего исследования** либо через `run_experiments.py`.
> - MNIST и CIFAR-10 скачиваются автоматически через `torchvision`.
> - Для `05_augmentations_and_transfer_learning` нужен датасет в `data/train`, `data/val`, `data/test`.

## 🖥 Оборудование

Эксперименты запускаются на **NVIDIA GeForce RTX 3050 Laptop GPU** (`--device cuda`). Smoke-тесты — на CPU.
