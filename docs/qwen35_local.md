# 本地 Qwen3.5-9B：部署与首次任务更新评测

更新日期：2026-11-05。本阶段使用纯文本公共请求，固定非思考模式；GPU 服务由用户手动启动。

## 已完成的 CPU 检查

- 模型路径：`/mnt/sdc/weiguanzhao/Qwen3.5-9B`，架构为 `Qwen3_5ForConditionalGeneration`。
- 四个 Safetensors 分片的文件头、张量索引、数据偏移和文件长度一致，覆盖 775 个张量；
  索引总数据量为 19,306,216,416 字节，约 18 GiB。检查未加载权重，也未核验发布方文件校验和。
- 配置、分词器及处理器 JSON 可读取，聊天模板支持 `enable_thinking=False`。
- 使用本地分词器和聊天模板进行 CPU 检查，54 条请求的输入为 666–718 token；
  最长输入加 512 token 输出预算为 1230，小于本轮配置的 4096 上下文上限。
- 现有 `llm` 环境为 Python 3.10.20，没有 vLLM；本方案新建独立推理环境。
- 代理会话无法通过 `nvidia-smi` 访问 NVIDIA 驱动；用户实际终端已完成本地服务的 54 条预测。
  原始响应、哈希和评分逐条核对通过，详见 [首次真实模型结果](qwen35_task_update_results_2026-11-05.md)。
  客户端日志未保存 GPU 编号、显存或完整服务启动参数，不据此补写这些信息。

## 1. 在实际 GPU 终端检查并安装服务

2026-11-05 首次自动安装得到 vLLM 0.33.0 与 Torch 2.14.0+cu126。
`readelf` 确认 vLLM 的 `_C_stable_libtorch` 扩展需要 `libcudart.so.13`，
而 Torch 使用 CUDA 12.6；CUDA 13 库文件实际存在，但未被当前导入链找到。
因此此处改为新的 CUDA 12 环境和明确版本，保留原环境供诊断。
新环境 `qwen35-cu129` 已完成安装核对：vLLM 0.17.1、Torch 2.10.0+cu129、
torchvision 0.25.0+cu129、torchaudio 2.10.0+cu129、CuTe DSL 4.4.0、QuACK 0.2.7；
`pip check` 通过，Torch 与 vLLM 顶层导入通过。
当前代理会话无法初始化 NVML、Triton 找不到有效 GPU 驱动，`serve --help` 在识别设备类型时失败。
此处保留代理会话的检查边界；随后用户已完成真实模型请求，54/54 调用返回并通过原始记录核对。

先确认分配的 GPU 可用。若同一终端也无法访问驱动，先解决 GPU 分配、容器透传或节点环境问题。
约 18 GiB 是完整权重数据量，不是服务显存预算；推理还需要运行时、缓存和工作区。
以下是单卡 BF16 配置，要求所选 GPU 支持 BF16 且有足够空闲显存；本批日志未冻结实际显存占用和服务参数。

```bash
cd /mnt/sdc/weiguanzhao/MIVLN-W
source /mnt/sdc/weiguanzhao/miniconda3/etc/profile.d/conda.sh
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

conda create -n qwen35-cu129 python=3.12 -y
conda activate qwen35-cu129
python -m pip install --upgrade pip
python -m pip install \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  "vllm==0.17.1" \
  "torch==2.10.0+cu129" \
  "torchvision==0.25.0+cu129" \
  "torchaudio==2.10.0+cu129" \
  "nvidia-cutlass-dsl==4.4.0" \
  "quack-kernels==0.2.7"
python -m pip check
python -m pip freeze > outputs/task_update_qwen35_environment_cu129_001.txt

python - <<'PY'
import torch
import vllm
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("vLLM:", vllm.__version__)
print("CUDA available:", torch.cuda.is_available())
assert torch.__version__ == "2.10.0+cu129"
assert torch.version.cuda == "12.9"
assert vllm.__version__ == "0.17.1"
assert torch.cuda.is_available(), "GPU unavailable in this terminal"
PY
vllm serve --help
```

环境已创建时，后续只需激活，不重复创建。安装步骤需要访问包源，但模型从本地路径读取。
vLLM 0.17.1 默认 CUDA 12.9 构建及独立环境要求见
[对应版本安装指南](https://docs.vllm.ai/en/v0.17.1/getting_started/installation/gpu/)，
该版本 [CUDA 依赖](https://github.com/vllm-project/vllm/blob/v0.17.1/requirements/cuda.txt)
明确匹配 Torch 2.10.0，并 [列出 Qwen3.5 架构支持](https://docs.vllm.ai/en/v0.17.1/models/supported_models/)。
CuTe DSL 与 QuACK 同时固定在该发布所支持的依赖版本，避免旧版服务导入不断更新的内核接口。
`pip check` 检查包依赖声明，不能代替原生扩展导入检查。
当前读到的内核驱动为 535.274.02；CUDA 12 的次版本兼容仍有功能和 PTX 限制，
导入成功后也需要真实 GPU 启动与推理验证，不能据此承诺完整可用。
本阶段不修改 Habitat 环境；后续复现使用保存的实际依赖版本。

## 2. 终端 A：启动本地服务

将 `TASK_UPDATE_GPU=0` 改为分配给你的 GPU 编号。服务绑定本机，启动后保持终端运行。

```bash
conda activate qwen35-cu129
export TASK_UPDATE_GPU=0

CUDA_VISIBLE_DEVICES="$TASK_UPDATE_GPU" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
vllm serve /mnt/sdc/weiguanzhao/Qwen3.5-9B \
  --served-model-name qwen3.5-9b \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --language-model-only \
  --reasoning-parser qwen3
```

`--language-model-only` 跳过视觉编码器；当前开发请求都是文本，服务不接收图片。
4096 是本轮短请求的上下文预算，后续长历史实验应重新检查 token 长度并调整。
非思考模式通过官方聊天模板开关控制，不依靠在提示词里添加“不要思考”。
本固定版本命令通过客户端 `--thinking-mode off` 显式发送 `enable_thinking=false`，
记录到 `run.json`；不依赖较新版本服务端的 `--default-chat-template-kwargs` 开关。
只允许按下一节客户端配置进行本轮非思考模式比较。
当前命令没有开启 JSON 约束解码，模型格式错误会进入原始评分。

## 3. 终端 B：检查服务并生成 54 条预测

```bash
cd /mnt/sdc/weiguanzhao/MIVLN-W
source /mnt/sdc/weiguanzhao/miniconda3/etc/profile.d/conda.sh
conda activate mivln

curl --fail --silent --show-error http://127.0.0.1:8000/v1/models

python -m scripts.predict_task_updates \
  --requests-dir outputs/task_update_requests_003 \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3.5-9b \
  --thinking-mode off \
  --temperature 0 \
  --max-tokens 512 \
  --timeout 120 \
  --output-dir outputs/task_update_qwen35_001
```

服务检查应返回模型 ID `qwen3.5-9b`。若检查失败，先检查终端 A 的启动日志，再运行预测。
首个真实推理调用可能包括 GPU 内核编译；120 秒是每次 HTTP 调用的超时上限，可按实际日志调整。
温度零用于本轮固定参数开发检查，并非 Qwen 官方建议采样参数的性能复现。
不要在本轮不同上下文模式间改变参数；若另行测试采样配置，使用新的结果目录并单独报告。

预测客户端不读取答案、几何或模型目录，仅发送公共输入；每条请求是独立对话。
模型权重由终端 A 的服务加载，客户端的 `--model` 必须与服务公开名称一致。
输出目录必须是新目录，调用失败仍保留计划槽位。

## 4. 评分与验收

```bash
python -m scripts.task_update score \
  --requests-dir outputs/task_update_requests_003 \
  --predictions outputs/task_update_qwen35_001/predictions.jsonl \
  --updater-name qwen3.5-9b-nonthinking \
  --output-dir outputs/task_update_qwen35_eval_001
```

- `outputs/task_update_qwen35_001/run.json`：应记录 `planned_records=54`、
  `thinking_mode=off` 和 `chat_template_kwargs.enable_thinking=false`；正常调用预期 `received=54`、`failed=0`。
- `predictions.jsonl`：应有 54 条互不重复且哈希对齐的预测；非法 JSON 和调用失败也保留。
- `responses.jsonl`：检查调用错误与 `finish_reason`，尤其关注长度截断。
- `outputs/task_update_qwen35_eval_001/summary.json`：每种模式分母为 18，报告合法率和完整更新准确率。
- `cases.csv` / `cases.jsonl`：检查缺失上下文、旧目标、恢复目标、禁区引用及拒绝判断的错误。

先验收记录完整性与错误处理，不要求模型达到 100%。这些是开发样例，尚不能支撑泛化结论。
完成后下一步是补充按语义组隔离的留出语言样例；连续模型状态与导航闭环仍需后续实现。

## 提交与验证范围

客户端、测试和部署说明可以提交为 `feat: configure non-thinking Qwen3.5 task update inference`。
CPU 模拟响应测试已验证模式参数透传及记录；用户已完成真实 54 条预测，调用和评分链路已验收。
完整语义正确 46/54，部署成功不等于模型在所有消息上理解正确。
权重存放在仓库外，环境记录与预测输出位于 Git 忽略的 `outputs/`；不自动暂存、提交或推送。
部署说明与首次模型结果记录现可随客户端一并提交；实际 GPU 编号、显存与服务启动参数仍需后续补充保存。
