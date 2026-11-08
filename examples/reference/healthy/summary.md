# 运行总结

## Attempt v1 (2026-11-08 20:40)
- **状态**: ✅ 成功
- **Score**: 尚未提交 | Improvement: N/A | Best: N/A
- **训练类型**: SFT (LoRA)
- **超参数**: `{"training_type": "SFT", "seed": 20261108, "split_seed": 20261108, "lr": 0.0001, "steps": 225, "batch_size": 8, "gradient_accumulation": 4, "effective_batch": 32, "max_length": 512, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.05, "weight_decay": 0.01, "warmup_steps": 15, "resume": null, "answer_only_loss": true, "remove_calculator_annotations": true}`
- **做了什么**: 用训练集的逐步解答进行仅回答部分的交叉熵训练；移除 <<计算器标记>>，保留文字推理与 #### 数字答案；随机保留256条验证数据并记录索引。保存适配器和合并后的完整模型。完成225步训练；指标见 logs/v1_metrics.jsonl。
- **为什么**: 一小时预算内优先使用高质量已知正确的推理轨迹，避免在线采样成本；LoRA限制参数变化以保留原模型数学能力，验证损失用于监测泛化。
- **问题/进步**: 完成225步训练；指标见 logs/v1_metrics.jsonl。
- **关键代码**: `labels = [-100]*len(prompt_ids) + answer_ids; loss.backward(); model.merge_and_unload()`
- **下一步建议**: 释放GPU后提交完整模型，根据准确率与剩余时间调整训练。

## Submission v1 (2026-11-08 20:43)
- **状态**: ✅ 成功
- **Score**: 57.85 | Improvement: -6.06 | Best: {'baseline_score': 63.91, 'elapsed_seconds': 171.17, 'improvement': -6.06, 'model_path': './output/v1', 'score': 57.85, 'submission_id': 1, 'timestamp': '2026-11-08T20:43:44.615358'}
- **训练类型**: SFT (LoRA merged full model)
- **超参数**: `{"training_type": "SFT", "seed": 20261108, "split_seed": 20261108, "lr": 0.0001, "steps": 225, "batch_size": 8, "gradient_accumulation": 4, "effective_batch": 32, "max_length": 512, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.05, "weight_decay": 0.01, "warmup_steps": 15, "resume": null, "answer_only_loss": true, "remove_calculator_annotations": true}`
- **做了什么**: 训练进程结束并释放GPU后，将 output/v1 完整模型提交到官方评测；等待完整测试集评测结果，用时171.2秒。
- **为什么**: 使用官方准确率反馈选择下一轮训练策略，保留已提交最佳模型。
- **问题/进步**: 官方返回如下。
- **下一步建议**: 根据分数和剩余时间继续训练或保留最佳结果。
- **Grading server response (verbatim)**:
```json
{"baseline_score":63.91,"best":{"baseline_score":63.91,"elapsed_seconds":171.17,"improvement":-6.06,"model_path":"./output/v1","score":57.85,"submission_id":1,"timestamp":"2026-11-08T20:43:44.615358"},"elapsed_seconds":171.17,"improvement":-6.06,"model_path":"./output/v1","score":57.85,"submission_id":1,"timestamp":"2026-11-08T20:43:44.615358","total_submissions":1}

```

## Model adjustment v2 (2026-11-08 20:45)
- **状态**: ✅ 成功
- **训练类型**: SFT checkpoint interpolation
- **超参数**: seed=20261108, v1 LoRA merge scale=0.2
- **做了什么**: 将已训练v1适配器的增量缩小至20%，合并到基础模型形成完整v2。
- **为什么**: v1准确率57.85低于基准63.91，测试降低训练偏移能否保留原有数学能力并获得格式适配收益。
- **问题/进步**: 不进行新梯度更新，复用真实训练权重。
- **下一步建议**: 官方提交评测。

## Submission v2 (2026-11-08 20:49)
- **状态**: ✅ 成功
- **Score**: 65.81 | Improvement: 1.9 | Best: {'baseline_score': 63.91, 'elapsed_seconds': 240.37, 'improvement': 1.9, 'model_path': './output/v2', 'score': 65.81, 'submission_id': 2, 'timestamp': '2026-11-08T20:49:19.236500'}
- **训练类型**: SFT (LoRA merged full model)
- **超参数**: `{"training_type": "SFT", "seed": 20261108, "split_seed": 20261108, "lr": 0.0001, "steps": 225, "batch_size": 8, "gradient_accumulation": 4, "effective_batch": 32, "max_length": 512, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.05, "weight_decay": 0.01, "warmup_steps": 15, "resume": null, "answer_only_loss": true, "remove_calculator_annotations": true, "adapter_merge_scale": 0.2, "source": "v1_adapter"}`
- **做了什么**: 训练进程结束并释放GPU后，将 output/v2 完整模型提交到官方评测；等待完整测试集评测结果，用时240.4秒。
- **为什么**: 使用官方准确率反馈选择下一轮训练策略，保留已提交最佳模型。
- **问题/进步**: 官方返回如下。
- **下一步建议**: 根据分数和剩余时间继续训练或保留最佳结果。
- **Grading server response (verbatim)**:
```json
{"baseline_score":63.91,"best":{"baseline_score":63.91,"elapsed_seconds":240.37,"improvement":1.9,"model_path":"./output/v2","score":65.81,"submission_id":2,"timestamp":"2026-11-08T20:49:19.236500"},"elapsed_seconds":240.37,"improvement":1.9,"model_path":"./output/v2","score":65.81,"submission_id":2,"timestamp":"2026-11-08T20:49:19.236500","total_submissions":2}

```

## Data processing: reward filtering (2026-11-08 20:53)
- **做了什么**: 基础模型按seed=20261108对2048个训练问题各采样2条解答，temperature=0.7，top_p=0.95，max_tokens=512。检查训练标签数值，排除截断输出，正确解答补齐####格式。原始轨迹和重算奖励分别保留。
- **为什么**: 初始严格####奖励遗漏了多数boxed/普通数字答案；数值检查允许保留模型自身正确推理风格，同时学习要求的结束格式。
- **结果**: `{"timestamp": "2026-11-08T20:53:37.513637+00:00", "seed": 20261108, "counts": {"total": 4096, "correct": 3180, "truncated": 67}, "reward": 0.7763671875, "rules": "last hash answer, else boxed number, else last number; reject truncations; canonicalize correct answer"}`

## Attempt v3 (2026-11-08 20:53)
- **状态**: ❌ 失败
- **Score**: 尚未提交 | Improvement: N/A | Best: N/A
- **训练类型**: RFT (LoRA)
- **超参数**: `{"training_type": "RFT", "seed": 20261108, "split_seed": 20261108, "lr": 3e-05, "steps": 150, "batch_size": 4, "gradient_accumulation": 4, "effective_batch": 16, "max_length": 768, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.05, "weight_decay": 0.01, "warmup_steps": 15, "resume": null, "data": "logs/rewarded.jsonl", "answer_only_loss": true, "remove_calculator_annotations": true}`
- **做了什么**: 用训练集的逐步解答（若data非空，则筛选模型自身最终答案正确的采样轨迹）进行仅回答部分的交叉熵训练；移除 <<计算器标记>>，保留文字推理与 #### 数字答案；随机保留256条验证数据并记录索引。保存适配器和合并后的完整模型。Traceback (most recent call last):
  File "/workspace/code/train.py", line 64, in <module>
    loss=model(**batch(ex)).loss; total+=loss.item()/a.accum; (loss/a.accum).backward()
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/peft/peft_model.py", line 1923, in forward
    return self.base_model(
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/peft/tuners/tuners_utils.py", line 311, in forward
    return self.model.forward(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/transformers/utils/generic.py", line 918, in wrapper
    output = func(self, *args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 449, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/transformers/utils/generic.py", line 1072, in wrapper
    outputs = func(self, *args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 384, in forward
    hidden_states = decoder_layer(
  File "/opt/fullbench/lib/python3.10/site-packages/transformers/modeling_layers.py", line 94, in __call__
    return super().__call__(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/transformers/utils/deprecation.py", line 172, in wrapped_func
    return func(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 249, in forward
    hidden_states = self.mlp(hidden_states)
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 46, in forward
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/peft/tuners/lora/layer.py", line 807, in forward
    result = result + lora_B(lora_A(dropout(x))) * scaling
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/linear.py", line 134, in forward
    return F.linear(input, self.weight, self.bias)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 54.00 MiB. GPU 0 has a total capacity of 47.58 GiB of which 26.44 MiB is free. Process 2621742 has 37.09 GiB memory in use. Process 2623845 has 10.29 GiB memory in use. Of the allocated memory 9.73 GiB is allocated by PyTorch, and 117.33 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

- **为什么**: 一小时预算内优先使用高质量已知正确的推理轨迹，避免在线采样成本；LoRA限制参数变化以保留原模型数学能力，验证损失用于监测泛化。
- **问题/进步**: Traceback (most recent call last):
  File "/workspace/code/train.py", line 64, in <module>
    loss=model(**batch(ex)).loss; total+=loss.item()/a.accum; (loss/a.accum).backward()
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/peft/peft_model.py", line 1923, in forward
    return self.base_model(
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/peft/tuners/tuners_utils.py", line 311, in forward
    return self.model.forward(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/transformers/utils/generic.py", line 918, in wrapper
    output = func(self, *args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 449, in forward
    outputs: BaseModelOutputWithPast = self.model(
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/transformers/utils/generic.py", line 1072, in wrapper
    outputs = func(self, *args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 384, in forward
    hidden_states = decoder_layer(
  File "/opt/fullbench/lib/python3.10/site-packages/transformers/modeling_layers.py", line 94, in __call__
    return super().__call__(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/transformers/utils/deprecation.py", line 172, in wrapped_func
    return func(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 249, in forward
    hidden_states = self.mlp(hidden_states)
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 46, in forward
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/peft/tuners/lora/layer.py", line 807, in forward
    result = result + lora_B(lora_A(dropout(x))) * scaling
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
    return forward_call(*args, **kwargs)
  File "/opt/fullbench/lib/python3.10/site-packages/torch/nn/modules/linear.py", line 134, in forward
    return F.linear(input, self.weight, self.bias)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 54.00 MiB. GPU 0 has a total capacity of 47.58 GiB of which 26.44 MiB is free. Process 2621742 has 37.09 GiB memory in use. Process 2623845 has 10.29 GiB memory in use. Of the allocated memory 9.73 GiB is allocated by PyTorch, and 117.33 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

- **关键代码**: `labels = [-100]*len(prompt_ids) + answer_ids; loss.backward(); model.merge_and_unload()`
- **下一步建议**: 释放GPU后提交完整模型，根据准确率与剩余时间调整训练。

### v3 failure root-cause clarification
- 采样已完成，但vLLM生成进程在退出清理时仍占用37GiB GPU；随后训练分配验证前向内存触发OOM。已终止本任务生成进程506及其worker542，等待GPU恢复后使用新尝试名称v3r重试，保留失败日志。

## Attempt v3r (2026-11-08 20:56)
- **状态**: ✅ 成功
- **Score**: 尚未提交 | Improvement: N/A | Best: N/A
- **训练类型**: RFT (LoRA)
- **超参数**: `{"training_type": "RFT", "seed": 20261108, "split_seed": 20261108, "lr": 3e-05, "steps": 150, "batch_size": 4, "gradient_accumulation": 4, "effective_batch": 16, "max_length": 768, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.05, "weight_decay": 0.01, "warmup_steps": 15, "resume": null, "data": "logs/rewarded.jsonl", "answer_only_loss": true, "remove_calculator_annotations": true}`
- **做了什么**: 用训练集的逐步解答（若data非空，则筛选模型自身最终答案正确的采样轨迹）进行仅回答部分的交叉熵训练；移除 <<计算器标记>>，保留文字推理与 #### 数字答案；随机保留256条验证数据并记录索引。保存适配器和合并后的完整模型。完成150步训练；指标见 logs/v3r_metrics.jsonl。
- **为什么**: 一小时预算内优先使用高质量已知正确的推理轨迹，避免在线采样成本；LoRA限制参数变化以保留原模型数学能力，验证损失用于监测泛化。
- **问题/进步**: 完成150步训练；指标见 logs/v3r_metrics.jsonl。
- **关键代码**: `labels = [-100]*len(prompt_ids) + answer_ids; loss.backward(); model.merge_and_unload()`
- **下一步建议**: 释放GPU后提交完整模型，根据准确率与剩余时间调整训练。

## Submission v3r (2026-11-08 21:00)
- **状态**: ✅ 成功
- **Score**: 68.23 | Improvement: 4.32 | Best: {'baseline_score': 63.91, 'elapsed_seconds': 232.51, 'improvement': 4.32, 'model_path': './output/v3r', 'score': 68.23, 'submission_id': 3, 'timestamp': '2026-11-08T21:00:36.343210'}
- **训练类型**: RFT (LoRA merged full model)
- **超参数**: `{"training_type": "RFT", "seed": 20261108, "split_seed": 20261108, "lr": 3e-05, "steps": 150, "batch_size": 4, "gradient_accumulation": 4, "effective_batch": 16, "max_length": 768, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.05, "weight_decay": 0.01, "warmup_steps": 15, "resume": null, "data": "logs/rewarded.jsonl", "answer_only_loss": true, "remove_calculator_annotations": true}`
- **做了什么**: 训练进程结束并释放GPU后，将 output/v3r 完整模型提交到官方评测；等待完整测试集评测结果，用时232.5秒。
- **为什么**: 使用官方准确率反馈选择下一轮训练策略，保留已提交最佳模型。
- **问题/进步**: 官方返回如下。
- **下一步建议**: 根据分数和剩余时间继续训练或保留最佳结果。
- **Grading server response (verbatim)**:
```json
{"baseline_score":63.91,"best":{"baseline_score":63.91,"elapsed_seconds":232.51,"improvement":4.32,"model_path":"./output/v3r","score":68.23,"submission_id":3,"timestamp":"2026-11-08T21:00:36.343210"},"elapsed_seconds":232.51,"improvement":4.32,"model_path":"./output/v3r","score":68.23,"submission_id":3,"timestamp":"2026-11-08T21:00:36.343210","total_submissions":3}

```

## Attempt v4 (2026-11-08 21:03)
- **状态**: ✅ 成功
- **Score**: 尚未提交 | Improvement: N/A | Best: N/A
- **训练类型**: DPO
- **超参数**: `{"training_type": "DPO", "seed": 20261108, "split_seed": 20261108, "steps": 60, "lr": 2e-05, "beta": 0.1, "batch_size": 2, "gradient_accumulation": 8, "effective_pair_batch": 16, "max_length": 768, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.0, "weight_decay": 0.01, "reference": "frozen base model", "data": "logs/rewarded.jsonl"}`
- **做了什么**: 对同一道训练问题的正确与错误模型采样构建偏好对；双方均补齐####答案格式并排除截断样本。以冻结基础模型为参考，训练LoRA偏好损失并保存合并完整模型。完成60步，354个偏好对。
- **为什么**: 首次SFT降低准确率，弱化增量有收益；DPO直接比较模型自身正确与错误推理，参考模型约束有助于保留原有能力，避免只拟合标准解答风格。
- **问题/进步**: 完成60步，354个偏好对。
- **关键代码**: `loss = -logsigmoid(beta * ((chosen_logp-rejected_logp)-(ref_chosen_logp-ref_rejected_logp))).mean()`
- **下一步建议**: GPU释放后提交，根据官方准确率选择最佳模型。

## Submission v4 (2026-11-08 21:07)
- **状态**: ✅ 成功
- **Score**: 66.26 | Improvement: 2.35 | Best: {'baseline_score': 63.91, 'elapsed_seconds': 232.51, 'improvement': 4.32, 'model_path': './output/v3r', 'score': 68.23, 'submission_id': 3, 'timestamp': '2026-11-08T21:00:36.343210'}
- **训练类型**: DPO (LoRA merged full model)
- **超参数**: `{"training_type": "DPO", "seed": 20261108, "split_seed": 20261108, "steps": 60, "lr": 2e-05, "beta": 0.1, "batch_size": 2, "gradient_accumulation": 8, "effective_pair_batch": 16, "max_length": 768, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.0, "weight_decay": 0.01, "reference": "frozen base model", "data": "logs/rewarded.jsonl"}`
- **做了什么**: 训练进程结束并释放GPU后，将 output/v4 完整模型提交到官方评测；等待完整测试集评测结果，用时220.0秒。
- **为什么**: 使用官方准确率反馈选择下一轮训练策略，保留已提交最佳模型。
- **问题/进步**: 官方返回如下。
- **下一步建议**: 根据分数和剩余时间继续训练或保留最佳结果。
- **Grading server response (verbatim)**:
```json
{"baseline_score":63.91,"best":{"baseline_score":63.91,"elapsed_seconds":232.51,"improvement":4.32,"model_path":"./output/v3r","score":68.23,"submission_id":3,"timestamp":"2026-11-08T21:00:36.343210"},"elapsed_seconds":220.04,"improvement":2.35,"model_path":"./output/v4","score":66.26,"submission_id":4,"timestamp":"2026-11-08T21:07:50.909167","total_submissions":4}

```

## Data processing: reward filtering (2026-11-08 21:12)
- **做了什么**: 根据已记录采样配置按seed=20261108生成训练问题解答，temperature=0.7，top_p=0.95，max_tokens=512。检查训练标签数值，排除截断输出，正确解答补齐####格式。原始轨迹和重算奖励分别保留，输入logs/rollout2s.jsonl，输出logs/rewarded2.jsonl。
- **为什么**: 初始严格####奖励遗漏了多数boxed/普通数字答案；数值检查允许保留模型自身正确推理风格，同时学习要求的结束格式。
- **结果**: `{"timestamp": "2026-11-08T21:12:28.224479+00:00", "seed": 20261108, "counts": {"total": 5169, "correct": 4328, "truncated": 63}, "reward": 0.8372992841942348, "rules": "last hash answer, else boxed number, else last number; reject truncations; canonicalize correct answer"}`

## Attempt v5 (2026-11-08 21:16)
- **状态**: ✅ 成功
- **Score**: 尚未提交 | Improvement: N/A | Best: N/A
- **训练类型**: RFT (LoRA)
- **超参数**: `{"training_type": "RFT", "seed": 20261108, "split_seed": 20261108, "lr": 2e-05, "steps": 250, "batch_size": 4, "gradient_accumulation": 4, "effective_batch": 16, "max_length": 768, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.05, "weight_decay": 0.01, "warmup_steps": 15, "resume": "output/v3r_adapter", "data": "logs/rewarded2.jsonl", "answer_only_loss": true, "remove_calculator_annotations": true}`
- **做了什么**: 用训练集的逐步解答（若data非空，则筛选模型自身最终答案正确的采样轨迹）进行仅回答部分的交叉熵训练；移除 <<计算器标记>>，保留文字推理与 #### 数字答案；随机保留256条验证数据并记录索引。保存适配器和合并后的完整模型。完成250步训练；指标见 logs/v5_metrics.jsonl。
- **为什么**: 一小时预算内优先使用高质量已知正确的推理轨迹，避免在线采样成本；LoRA限制参数变化以保留原模型数学能力，验证损失用于监测泛化。
- **问题/进步**: 完成250步训练；指标见 logs/v5_metrics.jsonl。
- **关键代码**: `labels = [-100]*len(prompt_ids) + answer_ids; loss.backward(); model.merge_and_unload()`
- **下一步建议**: 释放GPU后提交完整模型，根据准确率与剩余时间调整训练。

## Model adjustment v6 (2026-11-08 21:17)
- **状态**: ✅ 成功
- **训练类型**: trained checkpoint weighted merge
- **超参数**: `{"training_type": "trained checkpoint weighted merge", "seed": 20261108, "sources": ["v3r", "v4"], "weights": [1.0, 0.5], "source_settings": {"v3r": {"training_type": "RFT", "seed": 20261108, "split_seed": 20261108, "lr": 3e-05, "steps": 150, "batch_size": 4, "gradient_accumulation": 4, "effective_batch": 16, "max_length": 768, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.05, "weight_decay": 0.01, "warmup_steps": 15, "resume": null, "data": "logs/rewarded.jsonl", "answer_only_loss": true, "remove_calculator_annotations": true}, "v4": {"training_type": "DPO", "seed": 20261108, "split_seed": 20261108, "steps": 60, "lr": 2e-05, "beta": 0.1, "batch_size": 2, "gradient_accumulation": 8, "effective_pair_batch": 16, "max_length": 768, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.0, "weight_decay": 0.01, "reference": "frozen base model", "data": "logs/rewarded.jsonl"}}}`
- **做了什么**: 将['v3r', 'v4']的真实训练LoRA增量按[1.0, 0.5]加权，以cat方式精确表达加权增量后合并完整模型。
- **为什么**: 已观察弱化SFT增量优于完整增量，尝试保留数学能力并结合不同训练信号；此过程无新增梯度训练。
- **下一步建议**: 释放GPU后官方提交。

## Submission v5 (2026-11-08 21:20)
- **状态**: ✅ 成功
- **Score**: 67.85 | Improvement: 3.94 | Best: {'baseline_score': 63.91, 'elapsed_seconds': 232.51, 'improvement': 4.32, 'model_path': './output/v3r', 'score': 68.23, 'submission_id': 3, 'timestamp': '2026-11-08T21:00:36.343210'}
- **训练类型**: RFT (LoRA merged full model)
- **超参数**: `{"training_type": "RFT", "seed": 20261108, "split_seed": 20261108, "lr": 2e-05, "steps": 250, "batch_size": 4, "gradient_accumulation": 4, "effective_batch": 16, "max_length": 768, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.05, "weight_decay": 0.01, "warmup_steps": 15, "resume": "output/v3r_adapter", "data": "logs/rewarded2.jsonl", "answer_only_loss": true, "remove_calculator_annotations": true}`
- **做了什么**: 训练进程结束并释放GPU后，将 output/v5 完整模型提交到官方评测；等待完整测试集评测结果，用时239.5秒。
- **为什么**: 使用官方准确率反馈选择下一轮训练策略，保留已提交最佳模型。
- **问题/进步**: 官方返回如下。
- **下一步建议**: 根据分数和剩余时间继续训练或保留最佳结果。
- **Grading server response (verbatim)**:
```json
{"baseline_score":63.91,"best":{"baseline_score":63.91,"elapsed_seconds":232.51,"improvement":4.32,"model_path":"./output/v3r","score":68.23,"submission_id":3,"timestamp":"2026-11-08T21:00:36.343210"},"elapsed_seconds":239.49,"improvement":3.94,"model_path":"./output/v5","score":67.85,"submission_id":5,"timestamp":"2026-11-08T21:20:46.187084","total_submissions":5}

```

## Submission v6 (2026-11-08 21:25)
- **状态**: ✅ 成功
- **Score**: 68.92 | Improvement: 5.01 | Best: {'baseline_score': 63.91, 'elapsed_seconds': 245.73, 'improvement': 5.01, 'model_path': './output/v6', 'score': 68.92, 'submission_id': 6, 'timestamp': '2026-11-08T21:25:08.142896'}
- **训练类型**: trained checkpoint weighted merge (LoRA merged full model)
- **超参数**: `{"training_type": "trained checkpoint weighted merge", "seed": 20261108, "sources": ["v3r", "v4"], "weights": [1.0, 0.5], "source_settings": {"v3r": {"training_type": "RFT", "seed": 20261108, "split_seed": 20261108, "lr": 3e-05, "steps": 150, "batch_size": 4, "gradient_accumulation": 4, "effective_batch": 16, "max_length": 768, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.05, "weight_decay": 0.01, "warmup_steps": 15, "resume": null, "data": "logs/rewarded.jsonl", "answer_only_loss": true, "remove_calculator_annotations": true}, "v4": {"training_type": "DPO", "seed": 20261108, "split_seed": 20261108, "steps": 60, "lr": 2e-05, "beta": 0.1, "batch_size": 2, "gradient_accumulation": 8, "effective_pair_batch": 16, "max_length": 768, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.0, "weight_decay": 0.01, "reference": "frozen base model", "data": "logs/rewarded.jsonl"}}}`
- **做了什么**: 训练进程结束并释放GPU后，将 output/v6 完整模型提交到官方评测；等待完整测试集评测结果，用时245.7秒。
- **为什么**: 使用官方准确率反馈选择下一轮训练策略，保留已提交最佳模型。
- **问题/进步**: 官方返回如下。
- **下一步建议**: 根据分数和剩余时间继续训练或保留最佳结果。
- **Grading server response (verbatim)**:
```json
{"baseline_score":63.91,"best":{"baseline_score":63.91,"elapsed_seconds":245.73,"improvement":5.01,"model_path":"./output/v6","score":68.92,"submission_id":6,"timestamp":"2026-11-08T21:25:08.142896"},"elapsed_seconds":245.73,"improvement":5.01,"model_path":"./output/v6","score":68.92,"submission_id":6,"timestamp":"2026-11-08T21:25:08.142896","total_submissions":6}

```

## Model adjustment v7 (2026-11-08 21:25)
- **状态**: ✅ 成功
- **训练类型**: trained checkpoint weighted merge
- **超参数**: `{"training_type": "trained checkpoint weighted merge", "seed": 20261108, "sources": ["v3r", "v5", "v4"], "weights": [0.5, 0.5, 0.5], "source_settings": {"v3r": {"training_type": "RFT", "seed": 20261108, "split_seed": 20261108, "lr": 3e-05, "steps": 150, "batch_size": 4, "gradient_accumulation": 4, "effective_batch": 16, "max_length": 768, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.05, "weight_decay": 0.01, "warmup_steps": 15, "resume": null, "data": "logs/rewarded.jsonl", "answer_only_loss": true, "remove_calculator_annotations": true}, "v5": {"training_type": "RFT", "seed": 20261108, "split_seed": 20261108, "lr": 2e-05, "steps": 250, "batch_size": 4, "gradient_accumulation": 4, "effective_batch": 16, "max_length": 768, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.05, "weight_decay": 0.01, "warmup_steps": 15, "resume": "output/v3r_adapter", "data": "logs/rewarded2.jsonl", "answer_only_loss": true, "remove_calculator_annotations": true}, "v4": {"training_type": "DPO", "seed": 20261108, "split_seed": 20261108, "steps": 60, "lr": 2e-05, "beta": 0.1, "batch_size": 2, "gradient_accumulation": 8, "effective_pair_batch": 16, "max_length": 768, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.0, "weight_decay": 0.01, "reference": "frozen base model", "data": "logs/rewarded.jsonl"}}}`
- **做了什么**: 将['v3r', 'v5', 'v4']的真实训练LoRA增量按[0.5, 0.5, 0.5]加权，以cat方式精确表达加权增量后合并完整模型。
- **为什么**: 已观察弱化SFT增量优于完整增量，尝试保留数学能力并结合不同训练信号；此过程无新增梯度训练。
- **下一步建议**: 释放GPU后官方提交。

## Storage maintenance (2026-11-08 21:25)
- 删除非最佳v1的合并完整模型以保留后续候选空间；保留output/v1_adapter、全部训练日志及设置，可从只读基础模型重建。最佳v6及其它候选保持完整。

## Model adjustment v8 (2026-11-08 21:25)
- **状态**: ✅ 成功
- **训练类型**: trained checkpoint weighted merge
- **超参数**: `{"training_type": "trained checkpoint weighted merge", "seed": 20261108, "sources": ["v3r", "v4", "v1"], "weights": [1.0, 0.5, 0.1], "source_settings": {"v3r": {"training_type": "RFT", "seed": 20261108, "split_seed": 20261108, "lr": 3e-05, "steps": 150, "batch_size": 4, "gradient_accumulation": 4, "effective_batch": 16, "max_length": 768, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.05, "weight_decay": 0.01, "warmup_steps": 15, "resume": null, "data": "logs/rewarded.jsonl", "answer_only_loss": true, "remove_calculator_annotations": true}, "v4": {"training_type": "DPO", "seed": 20261108, "split_seed": 20261108, "steps": 60, "lr": 2e-05, "beta": 0.1, "batch_size": 2, "gradient_accumulation": 8, "effective_pair_batch": 16, "max_length": 768, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.0, "weight_decay": 0.01, "reference": "frozen base model", "data": "logs/rewarded.jsonl"}, "v1": {"training_type": "SFT", "seed": 20261108, "split_seed": 20261108, "lr": 0.0001, "steps": 225, "batch_size": 8, "gradient_accumulation": 4, "effective_batch": 32, "max_length": 512, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.05, "weight_decay": 0.01, "warmup_steps": 15, "resume": null, "answer_only_loss": true, "remove_calculator_annotations": true}}}`
- **做了什么**: 将['v3r', 'v4', 'v1']的真实训练LoRA增量按[1.0, 0.5, 0.1]加权，以cat方式精确表达加权增量后合并完整模型。
- **为什么**: 已观察弱化SFT增量优于完整增量，尝试保留数学能力并结合不同训练信号；此过程无新增梯度训练。
- **下一步建议**: 释放GPU后官方提交。

## Submission v7 (2026-11-08 21:29)
- **状态**: ✅ 成功
- **Score**: 69.37 | Improvement: 5.46 | Best: {'baseline_score': 63.91, 'elapsed_seconds': 263.79, 'improvement': 5.46, 'model_path': './output/v7', 'score': 69.37, 'submission_id': 7, 'timestamp': '2026-11-08T21:29:48.881073'}
- **训练类型**: trained checkpoint weighted merge (LoRA merged full model)
- **超参数**: `{"training_type": "trained checkpoint weighted merge", "seed": 20261108, "sources": ["v3r", "v5", "v4"], "weights": [0.5, 0.5, 0.5], "source_settings": {"v3r": {"training_type": "RFT", "seed": 20261108, "split_seed": 20261108, "lr": 3e-05, "steps": 150, "batch_size": 4, "gradient_accumulation": 4, "effective_batch": 16, "max_length": 768, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.05, "weight_decay": 0.01, "warmup_steps": 15, "resume": null, "data": "logs/rewarded.jsonl", "answer_only_loss": true, "remove_calculator_annotations": true}, "v5": {"training_type": "RFT", "seed": 20261108, "split_seed": 20261108, "lr": 2e-05, "steps": 250, "batch_size": 4, "gradient_accumulation": 4, "effective_batch": 16, "max_length": 768, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.05, "weight_decay": 0.01, "warmup_steps": 15, "resume": "output/v3r_adapter", "data": "logs/rewarded2.jsonl", "answer_only_loss": true, "remove_calculator_annotations": true}, "v4": {"training_type": "DPO", "seed": 20261108, "split_seed": 20261108, "steps": 60, "lr": 2e-05, "beta": 0.1, "batch_size": 2, "gradient_accumulation": 8, "effective_pair_batch": 16, "max_length": 768, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.0, "weight_decay": 0.01, "reference": "frozen base model", "data": "logs/rewarded.jsonl"}}}`
- **做了什么**: 训练进程结束并释放GPU后，将 output/v7 完整模型提交到官方评测；等待完整测试集评测结果，用时263.8秒。
- **为什么**: 使用官方准确率反馈选择下一轮训练策略，保留已提交最佳模型。
- **问题/进步**: 官方返回如下。
- **下一步建议**: 根据分数和剩余时间继续训练或保留最佳结果。
- **Grading server response (verbatim)**:
```json
{"baseline_score":63.91,"best":{"baseline_score":63.91,"elapsed_seconds":263.79,"improvement":5.46,"model_path":"./output/v7","score":69.37,"submission_id":7,"timestamp":"2026-11-08T21:29:48.881073"},"elapsed_seconds":263.79,"improvement":5.46,"model_path":"./output/v7","score":69.37,"submission_id":7,"timestamp":"2026-11-08T21:29:48.881073","total_submissions":7}

```

## Submission v8 (2026-11-08 21:34)
- **状态**: ✅ 成功
- **Score**: 62.77 | Improvement: -1.14 | Best: {'baseline_score': 63.91, 'elapsed_seconds': 263.79, 'improvement': 5.46, 'model_path': './output/v7', 'score': 69.37, 'submission_id': 7, 'timestamp': '2026-11-08T21:29:48.881073'}
- **训练类型**: trained checkpoint weighted merge (LoRA merged full model)
- **超参数**: `{"training_type": "trained checkpoint weighted merge", "seed": 20261108, "sources": ["v3r", "v4", "v1"], "weights": [1.0, 0.5, 0.1], "source_settings": {"v3r": {"training_type": "RFT", "seed": 20261108, "split_seed": 20261108, "lr": 3e-05, "steps": 150, "batch_size": 4, "gradient_accumulation": 4, "effective_batch": 16, "max_length": 768, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.05, "weight_decay": 0.01, "warmup_steps": 15, "resume": null, "data": "logs/rewarded.jsonl", "answer_only_loss": true, "remove_calculator_annotations": true}, "v4": {"training_type": "DPO", "seed": 20261108, "split_seed": 20261108, "steps": 60, "lr": 2e-05, "beta": 0.1, "batch_size": 2, "gradient_accumulation": 8, "effective_pair_batch": 16, "max_length": 768, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.0, "weight_decay": 0.01, "reference": "frozen base model", "data": "logs/rewarded.jsonl"}, "v1": {"training_type": "SFT", "seed": 20261108, "split_seed": 20261108, "lr": 0.0001, "steps": 225, "batch_size": 8, "gradient_accumulation": 4, "effective_batch": 32, "max_length": 512, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.05, "weight_decay": 0.01, "warmup_steps": 15, "resume": null, "answer_only_loss": true, "remove_calculator_annotations": true}}}`
- **做了什么**: 训练进程结束并释放GPU后，将 output/v8 完整模型提交到官方评测；等待完整测试集评测结果，用时270.0秒。
- **为什么**: 使用官方准确率反馈选择下一轮训练策略，保留已提交最佳模型。
- **问题/进步**: 官方返回如下。
- **下一步建议**: 根据分数和剩余时间继续训练或保留最佳结果。
- **Grading server response (verbatim)**:
```json
{"baseline_score":63.91,"best":{"baseline_score":63.91,"elapsed_seconds":263.79,"improvement":5.46,"model_path":"./output/v7","score":69.37,"submission_id":7,"timestamp":"2026-11-08T21:29:48.881073"},"elapsed_seconds":270.03,"improvement":-1.14,"model_path":"./output/v8","score":62.77,"submission_id":8,"timestamp":"2026-11-08T21:34:19.543767","total_submissions":8}

```

## Final result (2026-11-08 21:34)
- **状态**: ✅ 成功
- **Score**: 69.37 | Improvement: 5.46 | Best: 69.37
- **最佳完整模型**: `./output/v7`
- **设置**: seed=20261108；训练与切分设置见logs/*_settings.json，逐步时间戳/loss/reward见logs/*_metrics.jsonl；所有提交原文保留于logs/*_submission.json。
- **做了什么**: 完成8次完整测试集提交，比较SFT、弱化SFT、正确轨迹筛选训练、DPO、继续训练和权重组合；保留最佳完整模型，详细分数和轨迹汇总见best_model.json。
- **为什么**: 用官方准确率反馈调整方向；模型自身正确轨迹优于直接模仿标准解答，偏好训练的部分增量进一步提升最佳结果。
- **问题/进步**: 一次已记录OOM因采样worker延迟退出而发生，清理后成功重试；后续采样显式关闭worker。
- **下一步建议**: 本轮预算内停止新实验，以最佳完整模型作为任务结果。
