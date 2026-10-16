<div align="center">

# Deep Learning Fundamentals

**Пять исследовательских мини-проектов на PyTorch: от тензоров и автоградиента до CNN, аугментаций и fine-tuning**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)

</div>

Код, отчёты и графики лежат в [`pytorch-dl`](./pytorch-dl). Названия папок отражают вопрос исследования, а не номер задания.

| Папка | Вопрос |
|-------|--------|
| [`01_tensors_autograd_gpu_benchmark`](./pytorch-dl/01_tensors_autograd_gpu_benchmark) | Тензоры, автоград, CPU vs GPU |
| [`02_linear_models_and_hyperparameters`](./pytorch-dl/02_linear_models_and_hyperparameters) | Линейные модели и гиперпараметры |
| [`03_mlp_depth_width_regularization`](./pytorch-dl/03_mlp_depth_width_regularization) | Глубина, ширина и регуляризация MLP |
| [`04_cnn_vs_fc_architectures`](./pytorch-dl/04_cnn_vs_fc_architectures) | CNN vs FC и выбор архитектуры |
| [`05_augmentations_and_transfer_learning`](./pytorch-dl/05_augmentations_and_transfer_learning) | Аугментации и transfer learning |

Подробный обзор, результаты и команды запуска — в [`pytorch-dl/README.md`](./pytorch-dl/README.md).

```bash
git clone https://github.com/BobrovayaShishka/FoDLaAI.git
cd FoDLaAI
pip install -r requirements.txt
python pytorch-dl/run_experiments.py smoke
```
