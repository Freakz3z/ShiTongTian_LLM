from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

from unsloth import is_bfloat16_supported
import torch
max_seq_length = 512 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/meta-Llama-3.1-8B-Instruct",


    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.8, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

import re
from datasets import load_dataset, Dataset

# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def format_recipe_as_xml(name: str, ingredients: str, instructions: str, nutrition: str) -> str:
    return (f"<recipe>\n"
            f"<name>\n{name}\n</name>\n"
            f"<ingredients>\n{ingredients}\n</ingredients>\n"
            f"<instructions>\n{instructions}\n</instructions>\n"
            f"<nutrition>\n{nutrition}\n</nutrition>\n"
            f"</recipe>")

# 更新数据集加载函数以适应新的食谱数据集
def get_recipes_data(split='train') -> Dataset:
    data = load_dataset("AkashPS11/recipes_data_food.com", split=split)

    def format_ingredients(quantities, parts):
        if quantities and parts:
            quantities_list = quantities.split(',')
            parts_list = parts.split(',')
            return ', '.join([f"{q.strip()} {p.strip()}" for q, p in zip(quantities_list, parts_list)])
        return "No ingredients available"

    # 过滤掉缺失关键字段的记录
    data = data.filter(lambda x: (
        x['Name'] and
        x['RecipeInstructions'] and
        x['RecipeIngredientQuantities'] and
        x['RecipeIngredientParts'] and
        x.get('Calories') is not None and
        x.get('FatContent') is not None and
        x.get('SaturatedFatContent') is not None and
        x.get('CholesterolContent') is not None and
        x.get('SodiumContent') is not None and
        x.get('CarbohydrateContent') is not None and
        x.get('FiberContent') is not None and
        x.get('SugarContent') is not None and
        x.get('ProteinContent') is not None
    ))

    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['Name']}
        ],
        'answer': XML_COT_FORMAT.format(
            reasoning=f"Recipe for {x['Name']}",  # 可以根据需要调整reasoning部分
            answer=format_recipe_as_xml(
                name=x['Name'],
                ingredients=format_ingredients(
                    x.get('RecipeIngredientQuantities', ''),
                    x.get('RecipeIngredientParts', '')
                ),
                instructions=x['RecipeInstructions'],
                nutrition=(
                    f"Calories: {x['Calories']}, "
                    f"Fat: {x['FatContent']}g, "
                    f"Saturated Fat: {x['SaturatedFatContent']}g, "
                    f"Cholesterol: {x['CholesterolContent']}mg, "
                    f"Sodium: {x['SodiumContent']}mg, "
                    f"Carbohydrates: {x['CarbohydrateContent']}g, "
                    f"Fiber: {x['FiberContent']}g, "
                    f"Sugar: {x['SugarContent']}g, "
                    f"Protein: {x['ProteinContent']}g"
                )
            )
        )
    })
    return data

# 使用新函数来获取数据集
dataset = get_recipes_data()

# 打印数据集中前几个示例以查看输入和输出
for i in range(5):  # 打印前5个示例
    print(f"Example {i+1}")
    print("Prompt:")
    print(dataset[i]['prompt'])
    print("Answer:")
    print(dataset[i]['answer'])
    print("-" * 80)


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 6, # Decrease if out of memory
    max_prompt_length = 256,
    max_completion_length = 200,
    num_train_epochs = 6, # Set to 1 for a full training run
    max_steps = 250,
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()

text = tokenizer.apply_chat_template([
    {"role" : "user", "content" : "为我推荐一份食谱。"},
], tokenize = False, add_generation_prompt = True)

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = 1024,
)
output = model.fast_generate(
    [text],
    sampling_params = sampling_params,
    lora_request = None,
)[0].outputs[0].text

print(output)

model.save_lora("grpo_saved_lora")

text = tokenizer.apply_chat_template([
    {"role" : "system", "content" : SYSTEM_PROMPT},
    {"role" : "user", "content" : "为我推荐一份食谱。"},
], tokenize = False, add_generation_prompt = True)

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = 1024,
)
output = model.fast_generate(
    text,
    sampling_params = sampling_params,
    lora_request = model.load_lora("grpo_saved_lora"),
)[0].outputs[0].text

print(output)

# 上传至Huggingface，记得做相应更改
model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")
