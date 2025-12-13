# use JackFram/llama-68m in the huggingface to predict the eos of the prompt
import torch
import time
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from typing import List
import json
import random
from dataclasses import dataclass
from rich.progress import track
from tqdm import tqdm


@dataclass
class Model:
    model_name: str
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = ""
    m = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:1")
    model = Model(model_name, m, tokenizer)
    return model


def predict(model: Model, inputs: torch.Tensor, input_idx: int, eos_token_id: int = 2):
    eos_poss = []
    eos_probabilities = []
    next_token = inputs[0][-1]
    total_count = 0
    count = 0
    start_time = time.time()
    device = inputs.device
    batch_size, initial_length = inputs.size()
    total_length = 3000
    input_length = inputs.shape
    output_length = 0
    if inputs.shape[1] > total_length:
        return [], [], -1, -1
    generated_tokens_tensor = torch.zeros(
        (batch_size, total_length), device=device, dtype=torch.long
    )
    generated_tokens_tensor[:, :initial_length] = inputs
    with torch.no_grad():
        while next_token != eos_token_id:
            generate_ids = model.model(
                inputs,
                return_dict=True,
            )
            logits = generate_ids["logits"]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token_probs = probs[:, -1]
            next_token = torch.argmax(next_token_probs, dim=-1)
            count += 1
            total_count += 1
            if count % 50 == 0:
                print(
                    f"Sequence {input_idx} has generated {total_count} tokens, tps: {count/(time.time()-start_time)}"
                )
                count = 0
                start_time = time.time()
            if total_count + initial_length > 2000:
                return [], [], -1, -1
            generated_tokens_tensor[:, initial_length + total_count] = next_token
            inputs = generated_tokens_tensor[:, : initial_length + total_count + 1]
            output_length += 1
            # eos_rank, eos_probability = get_eos_position(
            #     next_token_probs, eos_token_id
            # )
            eos_rank, eos_probability = get_eos_position_opt(
                next_token_probs, eos_token_id
            )
            eos_poss.append(float(eos_rank))
            eos_probabilities.append(float(eos_probability))

    torch.cuda.empty_cache()
    return eos_poss, eos_probabilities, input_length, output_length


def predict_batch(
    model: torch.nn.Module,
    batch_size: int,
    task_list: List[torch.Tensor],
    eos_token_id: int = 2,
):
    inputs = torch.stack(task_list[:batch_size]).to("cuda:1")
    task_list = task_list[batch_size:]
    eos_poss = [[] for _ in range(batch_size)]
    eos_probabilities = [[] for _ in range(batch_size)]
    input_length = [[] for _ in range(batch_size)]
    output_length = [[] for _ in range(batch_size)]
    finished = [False] * batch_size
    current_inputs = inputs
    input_indices = list(range(batch_size))  # Track the original indices of inputs

    with torch.no_grad():
        while current_inputs.size(0) > 0:
            generate_ids = model(
                current_inputs,
                return_dict=True,
            )
            logits = generate_ids["logits"]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token_probs = probs[:, -1]
            next_tokens = torch.argmax(next_token_probs, dim=-1)

            new_inputs = []
            new_indices = []
            for idx, i in enumerate(input_indices):
                if finished[i]:
                    continue  # Skip finished inputs
                next_token = next_tokens[idx].item()
                new_input = torch.cat(
                    (
                        current_inputs[idx],
                        torch.tensor([next_token], device=inputs.device),
                    )
                )
                new_inputs.append(new_input)
                if next_token == eos_token_id:
                    finished[i] = True
                    eos_position, eos_probability = get_eos_position(
                        next_token_probs[idx], eos_token_id
                    )
                    eos_poss[i].append(float(eos_position))
                    eos_probabilities[i].append(float(eos_probability))
                    input_length[i].append(float(len(inputs[idx])))
                    output_length[i].append(float(len(new_input)))
                    new_inputs.remove(new_input)
                    new_input.to("cpu")
                else:
                    new_indices.append(i)  # Keep track of unfinished inputs

            if len(new_inputs) < batch_size and task_list:
                # Fill the batch with new tasks from the task list
                num_new_tasks = batch_size - len(new_inputs)
                left_over_tasks = len(task_list)
                print(left_over_tasks)
                for _ in range(num_new_tasks):
                    if task_list:
                        new_task = task_list.pop(0)
                        new_inputs.append(new_task.to("cuda:1"))
                        new_indices.append(len(finished))
                        eos_poss.append([])
                        eos_probabilities.append([])
                        input_length.append([])
                        output_length.append([])
                        finished.append(False)

            if not new_inputs:
                break  # If no unfinished inputs and no new tasks, break the loop

            current_inputs = torch.stack(new_inputs)
            input_indices = new_indices

            torch.cuda.empty_cache()
    return eos_poss, eos_probabilities, input_length, output_length


def get_eos_position(mode_result: torch.Tensor, eos_token_id: int = 2):
    eos_probability = mode_result[:, eos_token_id]
    sort_result = torch.sort(mode_result, descending=True)
    eos_rank = torch.where(sort_result[0] == eos_probability)[1][0]
    return int(eos_rank), float(eos_probability)


def get_eos_position_opt(mode_result: torch.Tensor, eos_token_id: int = 2):
    eos_probability = mode_result[:, eos_token_id]
    eos_rank = (mode_result > eos_probability.unsqueeze(1)).sum(dim=1)
    return eos_rank, eos_probability


def test_max_model_output_length_batch(large_model: Model, prompt: List[str]):
    eos_poss = []
    eos_probabilities = []
    input_lengths = []
    output_lengths = []
    inputs = list(
        large_model.tokenizer(prompt, padding=True, return_tensors="pt")["input_ids"]
    )
    eos_poss, eos_probabilities, input_lengths, output_lengths = predict_batch(
        large_model.model, 4, inputs, large_model.tokenizer.eos_token_id
    )
    # for seq in tqdm(prompt):
    #     inputs = large_model.tokenizer(seq, return_tensors="pt")
    #     inputs.to("cuda:1")
    #     model_eos = 2
    #     eos_pos, eos_probability, input_length, output_length = predict(
    #         large_model, inputs.input_ids, model_eos
    #     )
    #     eos_poss.append(eos_pos)
    #     eos_probabilities.append(eos_probability)
    #     input_lengths.append(input_length)
    #     output_lengths.append(output_length)
    return eos_poss, eos_probabilities, input_lengths, output_lengths


def save_tmp_result(tmp_result):
    print("saving tmp_result")
    with open("tmp_result.json", "w") as f:
        json.dump(tmp_result, f)


def test_max_model_output_length(large_model: Model, prompt: List[str]):
    eos_poss = []
    eos_probabilities = []
    input_lengths = []
    output_lengths = []
    tmp_result = {"eos_poss": [], "eos_probabilities": [],"output_lengths": [], "input_lengths": []}
    for seq in track(prompt, description="Predicting eos position..."):
        inputs = large_model.tokenizer(seq, return_tensors="pt")
        inputs.to("cuda:1")
        model_eos = 2
        eos_pos, eos_probability, input_length, output_length = predict(
            large_model, inputs.input_ids, prompt.index(seq), model_eos
        )
        if len(eos_pos) == 0 and len(eos_probability) == 0:
            continue
        eos_poss.append(eos_pos)
        eos_probabilities.append(eos_probability)
        input_lengths.append(input_length)
        output_lengths.append(output_length)
        tmp_result["eos_poss"].append(eos_pos)
        tmp_result["eos_probabilities"].append(eos_probability)
        tmp_result["output_lengths"].append(output_length)
        tmp_result["input_lengths"].append(input_length)
        save_tmp_result(tmp_result)
    return eos_poss, eos_probabilities, input_lengths, output_lengths


def get_prompt():
    dataset_path = "/root/vllm/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
    selected_seqs = []

    with open(dataset_path) as f:
        dataset = json.load(f)
        # Filter out the conversations with less than 2 turns.
        dataset = [data for data in dataset if len(data["conversations"]) >= 2]
        # Only keep the first two turns of each conversation.
        dataset = [
            (
                data["conversations"][0]["value"],
                data["conversations"][1]["value"],
            )
            for data in dataset
        ]

        # Shuffle the dataset.
        random.seed(10)
        random.shuffle(dataset)
        for i in range(len(dataset)):
            if len(set(selected_seqs)) == 200:
                break

            # Tokenize the prompts and completions.
            prompt = dataset[i][0]
            selected_seqs.append(prompt)
        return selected_seqs


if __name__ == "__main__":
    test_prompts = get_prompt()
    # large_model_name = "JackFram/llama-160m"
    large_model_name = "meta-llama/Llama-2-13b-chat-hf"
    large_model = load_model(large_model_name)
    # eos_probability(large_model, small_model, test_prompts)
    eos_poss, eos_probability, input_lengths, output_lengths = (
        test_max_model_output_length(large_model, test_prompts)
    )
    result = {
        "eos_poss": eos_poss,
        "eos_probabilities": eos_probability,
        "input_lengths": input_lengths,
        "output_lengths": output_lengths,
    }
    # save eos poss to json
    with open("eos_input_output_length.json", "w") as f:
        json.dump(result, f)
