# TNEWS 中文新闻分类

把中文新闻标题分到 TNEWS 的 15 个类别。对比了 BERT CLS、Qwen2-0.5B 零样本和 Qwen2-0.5B LoRA 三种做法。

## 评估

同一批 200 条验证样本，随机种子 42。

| 模型 | 准确率 | 宏平均 F1 |
|---|---:|---:|
| BERT CLS | 55.0% | 0.5405 |
| Qwen 零样本 | 36.0% | 0.3022 |
| Qwen LoRA | 58.0% | 0.5189 |

LoRA 将无法解析的输出从 58 条降到 2 条。BERT 在完整验证集（10,000 条）上的准确率为 55.33%。各类别结果见[评估报告](report/评估报告.md)。

## 运行

在仓库根目录执行以下命令。评估使用 Python 3.12 和带 CUDA 的 PyTorch。

```sh
python -m pip install -r requirements.txt
python prepare_data.py
```

BERT 训练、验证和单条预测。

```sh
python src/train.py --bert_path google-bert/bert-base-chinese --pool cls
python src/evaluate.py --bert_path google-bert/bert-base-chinese --ckpt_path outputs/checkpoints/best_cls.pt
python src/predict.py --bert_path google-bert/bert-base-chinese --ckpt_path outputs/checkpoints/best_cls.pt --text "梅西在比赛中打入一球"
```

Qwen 零样本、LoRA 训练和评估。

```sh
hf download Qwen/Qwen2-0.5B-Instruct --local-dir models/Qwen2-0.5B-Instruct
python src_llm/classify_llm.py --model_path models/Qwen2-0.5B-Instruct --num_samples 200 --seed 42
python src_llm/train_sft.py --model_path models/Qwen2-0.5B-Instruct
python src_llm/evaluate_sft.py --model_path models/Qwen2-0.5B-Instruct --ckpt_dir outputs/sft_adapter --num_samples 200 --seed 42
```

数据与模型由上游提供。[CLUE/TNEWS](https://huggingface.co/datasets/clue/clue/viewer/tnews) · [BERT](https://huggingface.co/google-bert/bert-base-chinese) · [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)

项目代码采用 [MIT 许可](LICENSE)。
