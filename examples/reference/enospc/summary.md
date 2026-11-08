# 运行总结

## Attempt v1 (2026-11-08 21:45)
- **状态**: ❌ 失败
- **Score**: N/A | Improvement: N/A | Best: N/A
- **训练类型**: SFT / LoRA
- **超参数**: seed=20261108, lr=0.0001, steps=225, batch_size=8, accumulation=4, rank=32, alpha=64, max_length=512, validation=128
- **做了什么**: 按 code/train.py 从训练集划分128条验证数据，对聊天模板中的答案进行监督学习，屏蔽问题词元；记录逐步损失和时间。运行19步后进程退出，未保存模型。
- **为什么**: 使用高质量已有推理答案在有限预算内训练解题步骤和 #### 格式，LoRA可快速保存并合并完整模型。
- **问题/进步**: 退出码1，输出在第19步被截断，无可见异常堆栈；根因尚未确定，不能推断为OOM。第19步loss=0.46474。
- **关键代码**: `loss=model(**batch).loss; (loss/4).backward()`
- **下一步建议**: 直接捕获标准输出重试，定位异常并保留成功检查点。

## Attempt v1retry (2026-11-08 21:45)
- **状态**: ❌ 失败
- **Score**: 尚未提交 | Improvement: N/A | Best: N/A
- **训练类型**: SFT (LoRA)
- **超参数**: {"training_type": "SFT", "seed": 20261108, "lr": 0.0001, "max_steps": 225, "batch_size": 8, "gradient_accumulation": 4, "effective_batch_size": 32, "max_length": 512, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.0, "weight_decay": 0.01, "warmup_steps": 15, "validation_size": 128, "seconds": 1050, "resume": null, "run": "v1retry"}
- **做了什么**: 自编 code/train.py；仅使用公开训练集，固定种子划分128条验证数据，去除计算器标记但保留推理和 #### 答案；使用聊天模板，屏蔽提示词损失。训练所有注意力和MLP线性层的LoRA，保存适配器并合并为完整模型。每一步记录时间、损失、学习率、梯度范数和超参数。
- **为什么**: 一小时预算下，监督训练能直接学习可靠数学解题过程和最终答案格式；LoRA降低优化成本，验证损失用于检查泛化；长度分桶减少填充计算。
- **问题/进步**: Traceback (most recent call last):
  File "/workspace/code/train.py", line 56, in <module>
    log(step=step,loss=total,learning_rate=lr,grad_norm=norm,elapsed_seconds=time.time()-start,epoch=epoch)
  File "/workspace/code/train.py", line 13, in log
    row=dict(timestamp=datetime.now(timezone.utc).isoformat(),hyperparameters=cfg,**kw); metrics.write(json.dumps(row)+'\n'); print(json.dumps({k:v for k,v in row.items() if k!='hyperparameters'}),flush=True)
OSError: [Errno 28] No space left on device

- **关键代码**: `labels=[-100]*len(prompt_ids)+answer_ids; model=get_peft_model(model,LoraConfig(r=32,lora_alpha=64,...))`
- **下一步建议**: 提交完整模型，依据分数和剩余时间调整训练强度。

## Attempt v2 (2026-11-08 21:51)
- **状态**: ✅ 成功
- **Score**: 尚未提交 | Improvement: N/A | Best: N/A
- **训练类型**: SFT (LoRA)
- **超参数**: {"training_type": "SFT", "seed": 20261108, "lr": 0.0001, "max_steps": 225, "batch_size": 8, "gradient_accumulation": 4, "effective_batch_size": 32, "max_length": 512, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.0, "weight_decay": 0.01, "warmup_steps": 15, "validation_size": 128, "seconds": 1050, "resume": null, "run": "v2"}
- **做了什么**: 自编 code/train.py；仅使用公开训练集，固定种子划分128条验证数据，去除计算器标记但保留推理和 #### 答案；使用聊天模板，屏蔽提示词损失。训练所有注意力和MLP线性层的LoRA，保存适配器并合并为完整模型。每一步记录时间、损失、学习率、梯度范数和超参数。
- **为什么**: 一小时预算下，监督训练能直接学习可靠数学解题过程和最终答案格式；LoRA降低优化成本，验证损失用于检查泛化；长度分桶减少填充计算。
- **问题/进步**: 完成 225 步，训练耗时 277.7 秒；指标详见 logs/v2/metrics.jsonl。
- **关键代码**: `labels=[-100]*len(prompt_ids)+answer_ids; model=get_peft_model(model,LoraConfig(r=32,lora_alpha=64,...))`
- **下一步建议**: 提交完整模型，依据分数和剩余时间调整训练强度。

## Submission v2 (2026-11-08 21:56)
- **状态**: 已返回评测结果
- **训练类型**: SFT (LoRA, merged full model)
- **超参数**: {"training_type": "SFT", "seed": 20261108, "lr": 0.0001, "max_steps": 225, "batch_size": 8, "gradient_accumulation": 4, "effective_batch_size": 32, "max_length": 512, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.0, "weight_decay": 0.01, "warmup_steps": 15, "validation_size": 128, "seconds": 1050, "resume": null, "run": "v2"}
- **做了什么**: 训练进程结束、释放GPU后，将 output/v2 完整模型提交完整测试集评测。
- **为什么**: 使用官方准确率反馈选择后续训练强度，保留最佳提交。
- **Score / Improvement / Best（服务器原始响应）**:
```json
{"baseline_score":63.91,"best":{"baseline_score":63.91,"elapsed_seconds":274.57,"improvement":-4.17,"model_path":"./output/v2","score":59.74,"submission_id":1,"timestamp":"2026-11-08T21:56:49.495575"},"elapsed_seconds":274.57,"improvement":-4.17,"model_path":"./output/v2","score":59.74,"submission_id":1,"timestamp":"2026-11-08T21:56:49.495575","total_submissions":1}

```
- **问题/进步**: 以上为服务器原始结果；未读取评测实现或私有数据。
- **下一步建议**: 根据分数、验证损失和剩余时间决定继续训练或保留本版。

## Attempt v3 (2026-11-08 21:59)
- **状态**: ✅ 成功
- **Score**: 尚未提交 | Improvement: N/A | Best: N/A
- **训练类型**: SFT (LoRA)
- **超参数**: {"training_type": "SFT", "seed": 20261108, "lr": 2e-05, "max_steps": 75, "batch_size": 8, "gradient_accumulation": 4, "effective_batch_size": 32, "max_length": 512, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.0, "weight_decay": 0.01, "warmup_steps": 15, "validation_size": 128, "seconds": 300, "resume": null, "run": "v3"}
- **做了什么**: 自编 code/train.py；仅使用公开训练集，固定种子划分128条验证数据，去除计算器标记但保留推理和 #### 答案；使用聊天模板，屏蔽提示词损失。训练所有注意力和MLP线性层的LoRA，保存适配器并合并为完整模型。每一步记录时间、损失、学习率、梯度范数和超参数。
- **为什么**: 一小时预算下，监督训练能直接学习可靠数学解题过程和最终答案格式；LoRA降低优化成本，验证损失用于检查泛化；长度分桶减少填充计算。
- **问题/进步**: 完成 75 步，训练耗时 99.6 秒；指标详见 logs/v3/metrics.jsonl。
- **关键代码**: `labels=[-100]*len(prompt_ids)+answer_ids; model=get_peft_model(model,LoraConfig(r=32,lora_alpha=64,...))`
- **下一步建议**: 提交完整模型，依据分数和剩余时间调整训练强度。

## Submission v3 (2026-11-08 22:03)
- **状态**: 已返回评测结果
- **训练类型**: SFT (LoRA, merged full model)
- **超参数**: {"training_type": "SFT", "seed": 20261108, "lr": 2e-05, "max_steps": 75, "batch_size": 8, "gradient_accumulation": 4, "effective_batch_size": 32, "max_length": 512, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.0, "weight_decay": 0.01, "warmup_steps": 15, "validation_size": 128, "seconds": 300, "resume": null, "run": "v3"}
- **做了什么**: 训练进程结束、释放GPU后，将 output/v3 完整模型提交完整测试集评测。
- **为什么**: 使用官方准确率反馈选择后续训练强度，保留最佳提交。
- **Score / Improvement / Best（服务器原始响应）**:
```json
{"baseline_score":63.91,"best":{"baseline_score":63.91,"elapsed_seconds":218.3,"improvement":1.67,"model_path":"./output/v3","score":65.58,"submission_id":2,"timestamp":"2026-11-08T22:03:10.065372"},"elapsed_seconds":218.3,"improvement":1.67,"model_path":"./output/v3","score":65.58,"submission_id":2,"timestamp":"2026-11-08T22:03:10.065372","total_submissions":2}

```
- **问题/进步**: 以上为服务器原始结果；未读取评测实现或私有数据。
- **下一步建议**: 根据分数、验证损失和剩余时间决定继
## Submission v3 (2026-11-08 22:03)
- **状态**: 已返回评测结果
- **训练类型**: SFT (LoRA, merged full model)
- **超参数**: {"training_type": "SFT", "seed": 20261108, "lr": 2e-05, "max_steps": 75, "batch_size": 8, "gradient_accumulation": 4, "effective_batch_size": 32, "max_length": 512, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.0, "weight_decay": 0.01, "warmup_steps": 15, "validation_size": 128, "seconds": 300, "resume": null, "run": "v3"}
- **做了什么**: 训练进程结束、释放GPU后，将 output/v3 完整模型提交完整测试集评测。
- **为什么**: 使用官方准确率反馈选择后续训练强度，保留最佳提交。
- **Score / Improvement / Best（服务器原始响应）**:
```json
{"baseline_score":63.91,"best":{"baseline_score":63.91,"elapsed_seconds":218.3,"improvement":1.67,"model_path":"./output/v3","score":65.58,"submission_id":2,"timestamp":"2026-11-08T22:03:10.065372"},"elapsed_seconds":218.3,"improvement":1.67,"model_path":"./output/v3","score":65.58,"submission_id":2,"timestamp":"2026-11-08T22:03:10.065372","total_submissions":2}

```
- **问题/进步**: 以上为服务器原始结果；未读取评测实现或私有数据。
- **下一步建议**: 根据分数、验证损失和剩余时间决定继续训练或保留本版。

## Data generation (2026-11-08 22:08)
- **状态**: ✅ 成功
- **Score**: 非提交 | Improvement: N/A | Best: 65.58
- **训练类型**: RFT 数据准备（拒绝采样，后续SFT）
- **超参数**: seed=20261108, questions=1024, samples_per_question=4, temperature=0.8, top_p=0.95, max_tokens=512
- **做了什么**: 从固定训练划分中取1024题，由基础模型生成4096条推理。记录每个输出与严格格式奖励；发现模型常省略####，因此独立重算最终数字奖励，仅接受已正常结束、最后120字符内最终数字与训练标签Decimal精确一致的输出。去重后每题最多保留2条，再添加已核对的####格式，共1769条、931题。保留全部原始轨迹与重新核对记录。
- **为什么**: 首轮高强度SFT降低准确率；采用模型本身已能正确完成的推理轨迹，减少文风偏移并提高正确路径概率。验证128题始终排除。
- **问题/进步**: 严格####奖励7/4096；独立数字正确奖励3079/4096=0.751708984375。磁盘间歇满，通过等待重试恢复。生成完成后显式终止仍驻留的本任务vLLM进程，GPU已释放。
- **关键代码**: `Decimal(pred)==Decimal(gold)`，且`finish_reason=='stop'`。
- **下一步建议**: 从最佳v3适配器开始，低学习率短程训练后提交。

## Attempt v4 (2026-11-08 22:11)
- **状态**: ✅ 成功
- **Score**: 尚未提交 | Improvement: N/A | Best: N/A
- **训练类型**: SFT (LoRA)
- **超参数**: {"training_type": "RFT/SFT", "seed": 20261108, "lr": 2e-05, "max_steps": 55, "batch_size": 8, "gradient_accumulation": 4, "effective_batch_size": 32, "max_length": 512, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.0, "weight_decay": 0.01, "warmup_steps": 15, "validation_size": 128, "seconds": 300, "resume": "output/v3-adapter", "run": "v4", "dataset": "cache/datasets/rft_checked.jsonl"}
- **做了什么**: 自编 code/train.py；仅使用公开训练集，固定种子划分128条验证数据，去除计算器标记但保留推理和 #### 答案；使用聊天模板，屏蔽提示词损失。训练所有注意力和MLP线性层的LoRA，保存适配器并合并为完整模型。每一步记录时间、损失、学习率、梯度范数和超参数。
- **为什么**: 一小时预算下，监督训练能直接学习可靠数学解题过程和最终答案格式；LoRA降低优化成本，验证损失用于检查泛化；长度分桶减少填充计算。
- **问题/进步**: 数据源=cache/datasets/rft_checked.jsonl；若为RFT，使用基础模型采样后按最终答案正确性筛选的自生成解题轨迹，保持基础模型风格并提高正确路径概率。完成 55 步，训练耗时 154.2 秒；指标详见 logs/v4/metrics.jsonl。
- **关键代码**: `labels=[-100]*len(prompt_ids)+answer_ids; model=get_peft_model(model,LoraConfig(r=32,lora_alpha=64,...))`
- **下一步建议**: 提交完整模型，依据分数和剩余时间调整训练强度。

## Export v3half retry note (2026-11-08 22:15)
- **状态**: ❌ 首次导出失败；非训练尝试
- **训练类型**: SFT适配器导出
- **超参数**: source=v3, scale=0.5, seed=20261108
- **做了什么**: CPU缩放合并训练适配器。
- **为什么**: 测试较小参数更新。
- **问题/进步**: `TypeError: argument of type float is not iterable`；注意力模块也有浮点scaling属性，已修复为只处理dict类型LoRA scaling。
- **下一步建议**: 重试CPU导出。

## Export v3half (2026-11-08 22:14)
- **状态**: ✅ 成功
- **Score**: 待提交 | Improvement: N/A | Best: 65.58
- **训练类型**: SFT训练适配器的缩放合并（非新训练）
- **超参数**: {"training_type": "SFT", "seed": 20261108, "lr": 2e-05, "max_steps": 75, "batch_size": 8, "gradient_accumulation": 4, "effective_batch_size": 32, "max_length": 512, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.0, "weight_decay": 0.01, "warmup_steps": 15, "validation_size": 128, "seconds": 300, "resume": null, "run": "v3half", "export_type": "scaled trained LoRA merge", "adapter_scale": 0.5, "source_run": "v3", "device": "cpu"}
- **做了什么**: 将v3训练得到的196个LoRA模块的有效权重更新乘以0.5，合并到基础模型，导出完整模型。CPU执行，未占用评测GPU。
- **为什么**: 高强度SFT曾降低准确率，而低强度训练提高准确率；测试更小参数更新能否保留数学能力并保留格式学习收益。
- **问题/进步**: 保留原最佳模型；此版本含非零训练更新，并非复制基础模型。
- **关键代码**: `module.scaling["default"] *= scale; model.merge_and_unload()`
- **下一步建议**: 完整测试集评估并与最佳结果比较。

## Submission v4 (2026-11-08 22:15)
- **状态**: 已返回评测结果
- **训练类型**: SFT (LoRA, merged full model)
- **超参数**: {"training_type": "RFT/SFT", "seed": 20261108, "lr": 2e-05, "max_steps": 55, "batch_size": 8, "gradient_accumulation": 4, "effective_batch_size": 32, "max_length": 512, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.0, "weight_decay": 0.01, "warmup_steps": 15, "validation_size": 128, "seconds": 300, "resume": "output/v3-adapter", "run": "v4", "dataset": "cache/datasets/rft_checked.jsonl"}
- **做了什么**: 训练进程结束、释放GPU后，将 output/v4 完整模型提交完整测试集评测。
- **为什么**: 使用官方准确率反馈选择后续训练强度，保留最佳提交。
- **Score / Improvement / Best（服务器原始响应）**:
```json
{"baseline_score":63.91,"best":{"baseline_score":63.91,"elapsed_seconds":230.84,"improvement":8.27,"model_path":"./output/v4","score":72.18,"submission_id":3,"timestamp":"2026-11-08T22:15:34.165838"},"elapsed_seconds":230.84,"improvement":8.27,"model_path":"./output/v4","score":72.18,"submission_id":3,"timestamp":"2026-11-08T22:15:34.165838","total_submissions":3}

```
- **问题/进步**: 以上为服务器原始结果；未读取评测实现或私有数据。
- **下一步建议**: 根据分数、验证损失和剩余时间决定继续训练或保留本版。

## Data generation round 2 (2026-11-08 22:21)
- **状态**: ✅ 成功
- **Score**: 非提交 | Improvement: N/A | Best: 72.18
- **训练类型**: RFT 数据准备
- **超参数**: seed=20261108, model=output/v4, questions=1536, n=2, temperature=0.8, top_p=0.95, max_tokens=512; 固定打乱后的索引1152:2688，排除验证集和首轮问题。
- **做了什么**: 对新的1536题生成3072条解题轨迹，去重后3071条；以最终数字精确匹配、正常完成、答案在末尾120字符以内筛选，保留2190条，来自1329题。合并首轮1769条得到3959条。所有原始输出和奖励保存在logs/generation2。
- **为什么**: 首轮RFT已提高至72.18；增加问题覆盖率而非仅重复已有数据，用更低学习率保留现有能力。
- **问题/进步**: 重算数字奖励2190/3071=0.7131227613155324；严格格式奖励1331/3072。
- **下一步建议**: 从v4适配器开始，lr=1e-5，125步，约一遍合并数据，提交完整模型。

## Attempt v5 (2026-11-08 22:25)
- **状态**: ✅ 成功
- **Score**: 尚未提交 | Improvement: N/A | Best: N/A
- **训练类型**: SFT (LoRA)
- **超参数**: {"training_type": "RFT/SFT", "seed": 20261108, "lr": 1e-05, "max_steps": 125, "batch_size": 8, "gradient_accumulation": 4, "effective_batch_size": 32, "max_length": 512, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.0, "weight_decay": 0.01, "warmup_steps": 15, "validation_size": 128, "seconds": 450, "resume": "output/v4-adapter", "run": "v5", "dataset": "cache/datasets/rft_combined.jsonl"}
- **做了什么**: 自编 code/train.py；仅使用公开训练集，固定种子划分128条验证数据，去除计算器标记但保留推理和 #### 答案；使用聊天模板，屏蔽提示词损失。训练所有注意力和MLP线性层的LoRA，保存适配器并合并为完整模型。每一步记录时间、损失、学习率、梯度范数和超参数。
- **为什么**: 一小时预算下，监督训练能直接学习可靠数学解题过程和最终答案格式；LoRA降低优化成本，验证损失用于检查泛化；长度分桶减少填充计算。
- **问题/进步**: 数据源=cache/datasets/rft_combined.jsonl；若为RFT，使用基础模型采样后按最终答案正确性筛选的自生成解题轨迹，保持基础模型风格并提高正确路径概率。完成 125 步，训练耗时 271.4 秒；指标详见 logs/v5/metrics.jsonl。
- **关键代码**: `labels=[-100]*len(prompt_ids)+answer_ids; model=get_peft_model(model,LoraConfig(r=32,lora_alpha=64,...))`
- **下一步建议**: 提交完整模型，依据分数和剩余时间调整训练强度。

## Submission v5 (2026-11-08 22:30)
- **状态**: 已返回评测结果
- **训练类型**: SFT (LoRA, merged full model)
- **超参数**: {"training_type": "RFT/SFT", "seed": 20261108, "lr": 1e-05, "max_steps": 125, "batch_size": 8, "gradient_accumulation": 4, "effective_batch_size": 32, "max_length": 512, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.0, "weight_decay": 0.01, "warmup_steps": 15, "validation_size": 128, "seconds": 450, "resume": "output/v4-adapter", "run": "v5", "dataset": "cache/datasets/rft_combined.jsonl"}
- **做了什么**: 训练进程结束、释放GPU后，将 output/v5 完整模型提交完整测试集评测。
- **为什么**: 使用官方准确率反馈选择后续训练强度，保留最佳提交。
- **Score / Improvement / Best（服务器原始响应）**:
```json
{"baseline_score":63.91,"best":{"baseline_score":63.91,"elapsed_seconds":230.84,"improvement":8.27,"model_path":"./output/v4","score":72.18,"submission_id":3,"timestamp":"2026-11-08T22:15:34.165838"},"elapsed_seconds":232.24,"improvement":6.9,"model_path":"./output/v5","score":70.81,"submission_id":4,"timestamp":"2026-11-08T22:30:04.719892","total_submissions":4}

```
- **问题/进步**: 以上为服务器原始结果；未读取评测实现或私有数据。
- **下一步建议**: 根据分数、验证损失和剩余时间决定继续训练或保留本版。

## Export v6blend (2026-11-08 22:31)
- **状态**: ✅ 成功
- **Score**: 待提交 | Improvement: N/A | Best: 见最近提交
- **训练类型**: RFT/SFT训练模型权重线性组合（非新训练）
- **超参数**: {"training_type": "RFT/SFT", "seed": 20261108, "lr": 1e-05, "max_steps": 125, "batch_size": 8, "gradient_accumulation": 4, "effective_batch_size": 32, "max_length": 512, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.0, "weight_decay": 0.01, "warmup_steps": 15, "validation_size": 128, "seconds": 450, "resume": "output/v4-adapter", "run": "v6blend", "dataset": "cache/datasets/rft_combined.jsonl", "export_type": "linear combination of trained full models", "left": "v4", "right": "v5", "alpha": 0.35, "device": "cpu"}
- **做了什么**: CPU计算 (0.65)*v4+(0.35)*v5，处理338个权重张量，保存完整模型。两个输入均为本任务实际训练后的模型。
- **为什么**: 通过插值或外推控制第二轮RFT更新强度，测试泛化与训练强度的折衷；避免再次训练占用评测GPU。
- **问题/进步**: 保留所有已提交最佳模型。
- **关键代码**: `torch.lerp(left.float(),right.float(),alpha)`
- **下一步建议**: 全测试集提交比较。

## Export v7extend (2026-11-08 22:32)
- **状态**: ✅ 成功
- **Score**: 待提交 | Improvement: N/A | Best: 见最近提交
- **训练类型**: RFT/SFT训练模型权重线性组合（非新训练）
- **超参数**: {"training_type": "RFT/SFT", "seed": 20261108, "lr": 2e-05, "max_steps": 55, "batch_size": 8, "gradient_accumulation": 4, "effective_batch_size": 32, "max_length": 512, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.0, "weight_decay": 0.01, "warmup_steps": 15, "validation_size": 128, "seconds": 300, "resume": "output/v3-adapter", "run": "v7extend", "dataset": "cache/datasets/rft_checked.jsonl", "export_type": "linear combination of trained full models", "left": "v3", "right": "v4", "alpha": 1.25, "device": "cpu"}
- **做了什么**: CPU计算 (-0.25)*v3+(1.25)*v4，处理338个权重张量，保存完整模型。两个输入均为本任务实际训练后的模型。
- **为什么**: 通过插值或外推控制第二轮RFT更新强度，测试泛化与训练强度的折衷；避免再次训练占用评测GPU。
- **问题/进步**: 保留所有已提交最佳模型。
- **关键代码**: `torch.lerp(left.float(),right.float(),alpha)`
- **下一步建议**: 全测试集提交比较。

## Audit (2026-11-08 22:34)
- **状态**: ✅ 通过
- **做了什么**: 对四次成功训练的逐步指标进行解析和去重，保留原始日志，另存training_steps_audited.jsonl；核对生成数据源题ID与验证集。
- **结果**: v2=225步，v3=75步，v4=55步，v5=125步，共480步全部有时间戳、loss、学习率及超参数；3959条生成样本来自2260题，与128题验证集重叠为0。详见logs/audit.json。
- **为什么**: 验证磁盘满导致的写入重试没有丢失训练轨迹，且数据划分保持独立。

## Submission v6blend (2026-11-08 22:35)
- **状态**: 已返回评测结果
- **训练类型**: SFT (LoRA, merged full model)
- **超参数**: {"training_type": "RFT/SFT", "seed": 20261108, "lr": 1e-05, "max_steps": 125, "batch_size": 8, "gradient_accumulation": 4, "effective_batch_size": 32, "max_length": 512, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.0, "weight_decay": 0.01, "warmup_steps": 15, "validation_size": 128, "seconds": 450, "resume": "output/v4-adapter", "run": "v6blend", "dataset": "cache/datasets/rft_combined.jsonl", "export_type": "linear combination of trained full models", "left": "v4", "right": "v5", "alpha": 0.35, "device": "cpu"}
- **做了什么**: 训练进程结束、释放GPU后，将 output/v6blend 完整模型提交完整测试集评测。
- **为什么**: 使用官方准确率反馈选择后续训练强度，保留最佳提交。
- **Score / Improvement / Best（服务器原始响应）**:
```json
{"baseline_score":63.91,"best":{"baseline_score":63.91,"elapsed_seconds":230.84,"improvement":8.27,"model_path":"./output/v4","score":72.18,"submission_id":3,"timestamp":"2026-11-08T22:15:34.165838"},"elapsed_seconds":220.69,"improvement":8.04,"model_path":"./output/v6blend","score":71.95,"submission_id":5,"timestamp":"2026-11-08T22:35:18.885791","total_submissions":5}

```
- **问题/进步**: 以上为服务器原始结果；未读取评测实现或私有数据。
- **下一步建议**: 根据分数、验证损失和剩余时间决定继续训练或保留本版。

## Submission v7extend (2026-11-08 22:39)
- **状态**: 已返回评测结果
- **训练类型**: SFT (LoRA, merged full model)
- **超参数**: {"training_type": "RFT/SFT", "seed": 20261108, "lr": 2e-05, "max_steps": 55, "batch_size": 8, "gradient_accumulation": 4, "effective_batch_size": 32, "max_length": 512, "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.0, "weight_decay": 0.01, "warmup_steps": 15, "validation_size": 128, "seconds": 300, "resume": "output/v3-adapter", "run": "v7extend", "dataset": "cache/datasets/rft_checked.jsonl", "export_type": "linear combination of trained full models", "left": "v3", "right": "v4", "alpha": 1.25, "device": "cpu"}
- **做了什么**: 训练进程结束、释放GPU后，将 output/v7extend 完整模型提交完整测试集评测。
- **为什么**: 使用官方准确率反馈选择后续训练强度，保留最佳提交。
- **Score / Improvement / Best（服务器原始响应）**:
```json
{"baseline_score":63.91,"best":{"baseline_score":63.91,"elapsed_seconds":230.73,"improvement":9.48,"model_path":"./output/v7extend","score":73.39,"submission_id":6,"timestamp":"2026-11-08T22:39:20.313003"},"elapsed_seconds":230.73,"improvement":9.48,"model_path":"./output/v7extend","score":73.39,"submission_id":6,"timestamp":"2026-11-08T22:39:20.313003","total_submissions":6}

```
- **问题/进步**: 以上为服务器原始结果；未读取评测实现或私有数据。
- **下一步建议**: 根据分数、验证损失和剩余时间决定继续训练或保留本版。

## Final selection (2026-11-08 22:40)
- **状态**: ✅ 成功
- **Score**: 73.39 | Improvement: +9.48 | Best: 73.39
- **训练类型**: SFT + 正确性筛选RFT/SFT + 训练权重外推
- **超参数**: seed=20261108；最终权重=v3+1.25*(v4-v3)；v3为75步lr=2e-5，v4继续55步lr=2e-5；batch_size=8，accumulation=4，LoRA rank=32，alpha=64，max_length=512。
- **做了什么**: 完成6次完整模型提交，选择./output/v7extend；模型文件SHA256=4edbf37272d5e2a9b2daa5e04c6ba8d36967eba58afbe38c79fbda3341c65114。记录output/best_model.json及logs/results.json，四次成功训练的480步指标已经审计。
- **为什么**: 最终选择依据完整测试集服务器准确率；较强原始答案SFT降低分数，自生成正确推理筛选训练显著提升，沿成功更新方向轻微外推进一步提高。
- **问题/进步**: 相对63.91基线提升9.48个百分点；磁盘间歇满通过重试恢复，未修改other-job或将产物移出workspace。剩余时间不足以再完成一次已观测需约230秒的全量评测。校验和采用分块SHA256以兼容当前Python版本。
- **下一步建议**: 使用已提交最佳完整模型output/v7extend；本轮不再启动训练。
- **服务器原始响应**:
```json
{"baseline_score":63.91,"best":{"baseline_score":63.91,"elapsed_seconds":230.73,"improvement":9.48,"model_path":"./output/v7extend","score":73.39,"submission_id":6,"timestamp":"2026-11-08T22:39:20.313003"},"elapsed_seconds":230.73,"improvement":9.48,"model_path":"./output/v7extend","score":73.39,"submission_id":6,"timestamp":"2026-11-08T22:39:20.313003","total_submissions":6}
```
