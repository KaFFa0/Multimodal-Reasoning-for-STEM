# Multimodal Reasoning for STEM

SFT мультимодальной модели **Qwen3-VL-4B-Instruct** для задачи распознавания рукописных математических формул и их преобразования в код LaTeX.  
Проект исследует несколько подходов: zero-shot, one-shot inference и SFT с использованием QLoRA на специализированных датасетах

## Общая информация
- Модель: [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- Итоговая лучшая SFT модель: [KaFFa0/Qwen3-VL-latex-lora](https://huggingface.co/KaFFa0/qwen3-vl-latex-lora)
- SFT на комбинированном датасете: [KaFFa0/Qwen3-VL-comb-lora](https://huggingface.co/KaFFa0/qwen3-vl-comb-lora)
- 4-битная квантизация для оптимизации обучения и инференса
- LoRA-адаптеры для быстрого дообучения
- Оценка по метрикам Exact Match, BLEU и Levenshtein similarity
- Веб-приложение на Streamlit для интерактивного тестирования

## Результаты
| Метод                  | Exact Match | BLEU    | Levenshtein | Arith Mean | Harm Mean |
|------------------------|-------------|---------|-------------|------------|-----------|
| Zero-shot              | 0.0         | 0.43756 | 0.70542     | 0.38       | 0.0       |
| One-shot               | 0.15714     | 0.64242 | 0.79585     | 0.53       | 0.32      |
| SFT (linxy)            | 0.75714     | 0.95226 | 0.95991     | 0.89       | 0.88      |
| SFT (comb)             | 0.0         | 0.52028 | 0.73171     | 0.41       | 0.0       |

Подробный анализ приведён в [отчёте](report/Technical_Report.pdf).

## Структура репозитория
- `src/` – исходный код обучения и приложения
- `notebooks/` – Jupyter ноутбук со всеми результатами
- `report/` – отчет и примеры работы приложения
