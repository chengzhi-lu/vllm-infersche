
"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    vLLM OpenAI API server
    vllm serve <your_model> \
        --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_tgi_server.sh <your_model> <max_batch_total_tokens>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --model <your_model> \
        --dataset-name sharegpt \
        --dataset-path <path to dataset> \
        --request-rate <request_rate> \ # By default <request_rate> is inf
        --num-prompts <num_prompts> # By default <num_prompts> is 1000

    when using tgi backend, add
        --endpoint /generate_stream
    to the end of the command above.
"""

import argparse
import asyncio
import json
import os
import random
import time
import warnings
import fnmatch
import multiprocessing
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import pandas as pd 

import numpy as np
from backend_request_func import ASYNC_REQUEST_FUNCS, RequestFuncInput, RequestFuncOutput
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    p99_itl_ms: float
    mean_lat_ms: float
    median_lat_ms: float
    p99_lat_ms: float
    std_itl_ms: float
    mean_avg_lat_ms: float
    median_avg_lat_ms: float
    p99_avg_lat_ms: float


def sample_alpaca_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    scheduler_policy: Optional[str] = None,
):
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversation"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversation"][0]["value"], data["conversation"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)
    prompt_len_list = []
    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    filtered_data_prompts = []
    data = None
    if args.scheduler_policy in ["srjf", "sjf"]:
        file_path = get_json_file(dataset_name="alpaca", qps=args.request_rate)
        if file_path:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            # print(f"Loaded data from {file_path}")
        else:
            print("No JSON file found in the current directory.")

    for i in range(len(dataset)):
        if len(filtered_data_prompts) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        if prompt in filtered_data_prompts:
            continue
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        prompt_len_list.append(prompt_len)
        # if data is not None:
        # if prompt not in data['prompts']:
        output_len = len(completion_token_ids)
        if prompt_len + output_len < 128:
            continue
            # else:
            #     index = data['prompts'].index(prompt)
            #     output_len = data['output_lens'][index]
            #     if output_len == 0:
            #         output_len = len(completion_token_ids
            #                     )
        # else:
        #     output_len=0
        filtered_dataset.append((prompt, prompt_len, output_len))
        filtered_data_prompts.append(prompt)
    print(f"Number of requests: {len(filtered_dataset)}")
    return filtered_dataset, data


def sample_leval_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    scheduler_policy: Optional[str] = None,
):
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"], data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)
    prompt_len_list = []
    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    filtered_data_prompts = []
    data = None
    if scheduler_policy in ["srjf", "sjf"]:
        file_path = get_json_file(dataset_name="leval", qps=args.request_rate)
        if file_path:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            # print(f"Loaded data from {file_path}")
        else:
            print("No JSON file found in the current directory.")
    if len(dataset) < num_requests:
        num_samples_needed = num_requests - len(dataset)
        dataset_index = [i for i in range(len(dataset))]
        additional_samples_index = np.random.choice(dataset_index, size=num_samples_needed, replace=True)
        additional_samples = [dataset[i] for i in additional_samples_index]
        dataset.extend(additional_samples)

    for i in range(len(dataset)):
        if len(filtered_data_prompts) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        if prompt in filtered_data_prompts:
            continue
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        prompt_len_list.append(prompt_len)
        # if data is not None:
        # if prompt not in data['prompts']:
        output_len = len(completion_token_ids)
        if prompt_len + output_len < 128:
            continue
        if "13b" in args.model:
            prompt = prompt[: 4096 - output_len]
            prompt_len = len(prompt[: 4096 - output_len])
            output_len = output_len
        if "70b" in args.model:
            prompt = prompt[: 8192 - output_len]
            prompt_len = len(prompt[: 8192 - output_len])
            output_len = output_len
            # filtered_dataset.append((prompt[:8192-output_len], len(prompt[:8192-output_len]), output_len))
        filtered_dataset.append((prompt, prompt_len, output_len))
        filtered_data_prompts.append(prompt)
    print(f"Number of requests: {len(filtered_dataset)}")
    return filtered_dataset, data


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    scheduler_policy: Optional[str] = None,
    phase: str = "hybrid",
):
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"], data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)
    prompt_len_list = []
    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    filtered_data_prompts = []
    data = None
    if args.scheduler_policy in ["srjf", "sjf"]:
        file_path = get_json_file(dataset_name="sharegpt", qps=args.request_rate)
        if file_path:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            # print(f"Loaded data from {file_path}")
        else:
            print("No JSON file found in the current directory.")

    for i in range(len(dataset)):
        if len(filtered_data_prompts) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        if prompt in filtered_data_prompts:
            continue
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        if phase == "decode" and prompt_len > 1:
            continue
        prompt_len_list.append(prompt_len)
        # if data is not None:
        # if prompt not in data['prompts']:
        output_len = len(completion_token_ids)
        if prompt_len + output_len < 128:
            continue
            # else:
            #     index = data['prompts'].index(prompt)
            #     output_len = data['output_lens'][index]
            #     if output_len == 0:
            #         output_len = len(completion_token_ids
            #                     )
        # else:
        #     output_len=0
        if phase == "prefill":
            output_len = 1
        filtered_dataset.append((prompt, prompt_len, output_len))
        filtered_data_prompts.append(prompt)
    print(f"Number of requests: {len(filtered_dataset)}")
    return filtered_dataset, data


def sample_lmsys_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    scheduler_policy: Optional[str] = None,
    phase: str = "hybrid",
):
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"], data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)
    prompt_len_list = []
    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    filtered_data_prompts = []
    data = None
    if args.scheduler_policy in ["srjf", "sjf"]:
        file_path = get_json_file(dataset_name="lmsys", qps=args.request_rate)
        if file_path:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            # print(f"Loaded data from {file_path}")
        else:
            print("No JSON file found in the current directory.")

    for i in range(len(dataset)):
        if len(filtered_data_prompts) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        if prompt in filtered_data_prompts:
            continue
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        if phase == "decode" and prompt_len > 4:
            continue
        prompt_len_list.append(prompt_len)
        # if data is not None:
        # if prompt not in data['prompts']:
        output_len = len(completion_token_ids)
        if prompt_len + output_len < 128:
            continue
            # else:
            #     index = data['prompts'].index(prompt)
            #     output_len = data['output_lens'][index]
            #     if output_len == 0:
            #         output_len = len(completion_token_ids
            #                     )
        # else:
        #     output_len=0
        if phase == "prefill":
            output_len = 1
        filtered_dataset.append((prompt, prompt_len, output_len))
        filtered_data_prompts.append(prompt)
    print(f"Number of requests: {len(filtered_dataset)}")
    return filtered_dataset, data


def sample_sonnet_requests(
    dataset_path: str,
    num_requests: int,
    input_len: int,
    output_len: int,
    prefix_len: int,
    tokenizer: PreTrainedTokenizerBase,
):
    assert input_len > prefix_len, "'args.sonnet-input-len' must be greater than 'args.prefix-input-len'."

    # Load the dataset.
    with open(dataset_path) as f:
        poem_lines = f.readlines()

    # Tokenize the poem lines.
    poem_token_ids = tokenizer(poem_lines).input_ids
    average_poem_len = sum(len(token_ids) for token_ids in poem_token_ids) / len(poem_token_ids)

    # Base prefix for all requests.
    base_prompt = "Pick as many lines as you can from these poem lines:\n"
    base_message = [
        {
            "role": "user",
            "content": base_prompt,
        }
    ]
    base_prompt_formatted = tokenizer.apply_chat_template(base_message, add_generation_prompt=True, tokenize=False)
    base_prompt_offset = len(tokenizer(base_prompt_formatted).input_ids)

    assert input_len > base_prompt_offset, f"Please set 'args.sonnet-input-len' higher than {base_prompt_offset}."
    num_input_lines = round((input_len - base_prompt_offset) / average_poem_len)

    # First approximately `prefix_len` number of tokens in the
    # prompt are fixed poem lines.
    assert prefix_len > base_prompt_offset, f"Please set 'args.sonnet-prefix-len' higher than {base_prompt_offset}."

    num_prefix_lines = round((prefix_len - base_prompt_offset) / average_poem_len)
    prefix_lines = poem_lines[:num_prefix_lines]

    # Sample the rest of lines per request.
    sampled_requests: List[Tuple[str, int, int]] = []
    for _ in range(num_requests):
        sampled_lines = "".join(prefix_lines + random.sample(poem_lines, num_input_lines - num_prefix_lines))

        prompt = f"{base_prompt}{sampled_lines}"
        message = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
        prompt_formatted = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        prompt_len = len(tokenizer(prompt_formatted).input_ids)
        sampled_requests.append((prompt, prompt_formatted, prompt_len, output_len))

    return sampled_requests


def sample_random_requests(
    input_len: int, output_len: int, num_prompts: int, range_ratio: float, tokenizer: PreTrainedTokenizerBase
) -> List[Tuple[str, int, int]]:
    input_lens = np.random.randint(
        int(input_len * range_ratio),
        input_len + 1,
        size=num_prompts,
    )
    output_lens = np.random.randint(
        int(output_len * range_ratio),
        output_len + 1,
        size=num_prompts,
    )
    offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
    input_requests = []
    for i in range(num_prompts):
        prompt = tokenizer.decode([(offsets[i] + i + j) % tokenizer.vocab_size for j in range(input_lens[i])])
        input_requests.append((prompt, int(input_lens[i]), int(output_lens[i])))

    return input_requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    intervals = [np.random.exponential(1.0 / request_rate) for _ in input_requests]
    input_requests = iter(input_requests)
    i = 0
    for request in input_requests:
        yield request

        if request_rate == float("inf") or request_rate == -1:
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the exponential distribution.
        interval = intervals[i]
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)
        i += 1


def get_json_file(dataset_name, qps):
    dir_name = f"./{dataset_name}"
    for file_name in os.listdir(f"./{dataset_name}"):
        if fnmatch.fnmatch(file_name, "*.json"):
            if str(qps) not in file_name:
                continue
            return f"{dir_name}/{file_name}"
    return None


def generate_request(input_requests: List[Tuple[str, int, int]]) -> Tuple[str, int, int]:
    request = input_requests[random.randint(0, len(input_requests) - 1)]
    return request


def get_azure_request_data():
    azure_request_data = pd.read_csv("/root/vllm/dataset/AzureLLMInferenceTrace_conv_1week.csv")
    # azure request data columns: TIMESTAMP, CONTEXT_LENGTH, OUTPUT_LENGTH
    azure_request_data = azure_request_data.sort_values(by=['TIMESTAMP'])
    # calculate the interval between two requests
    azure_request_data['INTERVAL'] = azure_request_data['TIMESTAMP'].diff()
    # filter out the requests with less than 100ms interval
    print(azure_request_data)

async def get_request_duration(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
    request_duration: float,
    scheduler_policy: Optional[str],
) -> AsyncGenerator[Tuple[str, int, int], None]:
    st = time.time()
    while time.time() - st < request_duration:
        request = input_requests[random.randint(0, len(input_requests) - 1)]
        yield request
        if request_rate == float("inf") or request_rate == -1:
            continue
        interval = np.random.exponential(1.0 / request_rate)
        await asyncio.sleep(interval)


def save_output(outputs: List[RequestFuncOutput], filename: str) -> None:
    results = []
    for output in outputs:
        if output.success:
            result = {
                "prompt": output.prompt,
                "generated": output.generated_text,
            }
            results.append(result)
    with open(filename, "w") as f:
        f.write(json.dumps(results))


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens: List[int] = []
    total_input = 0
    completed = 0
    latencies = []
    avg_latencies = []
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            # We use the tokenizer to count the number of output tokens for all
            # serving backends instead of looking at len(outputs[i].itl) since
            # multiple output tokens may be bundled together
            # Note: this may inflate the output token count slightly
            output_len = len(tokenizer(outputs[i].generated_text, add_special_tokens=False).input_ids)
            actual_output_lens.append(output_len)
            outputs[i].actual_output_len = max(output_len,1)
            total_input += outputs[i].prompt_len
            if output_len > 1:
                tpots.append((outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            latencies.append(outputs[i].latency)
            avg_latencies.append(outputs[i].latency/outputs[i].actual_output_len)
            completed += 1
        else:
            actual_output_lens.append(0)

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration on the benchmark arguments.", stacklevel=2
        )
    ttfts = [t for t in ttfts if t > 0]
    tpots = [t for t in tpots if t > 0]
    itls = [t for t in itls if t > 0]
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        mean_ttft_ms=float(np.mean(ttfts or 0) * 1000),  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=float(np.median(ttfts or 0) * 1000),
        std_ttft_ms=float(np.std(ttfts or 0) * 1000),
        p99_ttft_ms=float(np.percentile(ttfts or 0, 99) * 1000),
        mean_tpot_ms=float(np.mean(tpots or 0) * 1000),
        median_tpot_ms=float(np.median(tpots or 0) * 1000),
        std_tpot_ms=float(np.std(tpots or 0) * 1000),
        p99_tpot_ms=float(np.percentile(tpots or 0, 99) * 1000),
        mean_itl_ms=float(np.mean(itls or 0) * 1000),
        median_itl_ms=float(np.median(itls or 0) * 1000),
        p99_itl_ms=float(np.percentile(itls or 0, 99) * 1000),
        mean_lat_ms=float(np.mean(latencies or 0) * 1000),
        median_lat_ms=float(np.median(latencies or 0) * 1000),
        p99_lat_ms=float(np.percentile(latencies or 0, 99) * 1000),
        std_itl_ms=float(np.std(itls or 0) * 1000),
        mean_avg_lat_ms=float(np.mean(avg_latencies or 0)*1000),
        median_avg_lat_ms=float(np.median(avg_latencies or 0) * 1000),
        p99_avg_lat_ms=float(np.percentile(avg_latencies or 0, 99) * 1000),
    )

    return metrics, actual_output_lens


async def process_request(request, model_id, api_url, best_of, use_beam_search, backend, request_func, pbar):
    """
    Function to process a single request.
    """
    prompt, prompt_len, output_len = request
    request_func_input = RequestFuncInput(
        model=model_id,
        prompt=prompt,
        api_url=api_url,
        prompt_len=prompt_len,
        output_len=output_len,
        best_of=best_of,
        use_beam_search=use_beam_search,
    )

    if backend == "vllm":
        return await request_func(scheduler_policy=None, request_func_input=request_func_input, pbar=pbar)
    else:
        return await request_func(request_func_input=request_func_input, pbar=pbar)


def run_event_loop_in_process(requests, model_id, api_url, best_of, use_beam_search, backend, request_func, pbar):
    """
    Run the asyncio event loop to handle multiple requests in a process.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Run multiple async tasks within this process
    result = loop.run_until_complete(
        process_multiple_requests(requests, model_id, api_url, best_of, use_beam_search, backend, request_func, pbar)
    )

    loop.close()
    return result


async def process_multiple_requests(requests, model_id, api_url, best_of, use_beam_search, backend, request_func, pbar):
    tasks = []
    for request in requests:
        tasks.append(process_request(request, model_id, api_url, best_of, use_beam_search, backend, request_func, pbar))
    results = await asyncio.gather(*tasks)
    return results


def process_requests(backend, args, request_func, data1=None):
    async def handle_requests():
        tasks: List[asyncio.Task] = []
        while True:
            request_func_input = await asyncio.get_event_loop().run_in_executor(None, request_queue.get)
            if request_func_input is None:
                break

            if backend == "vllm":
                tasks.append(
                    asyncio.create_task(request_func(args.scheduler_policy, request_func_input=request_func_input))
                )
            else:
                tasks.append(asyncio.create_task(request_func(request_func_input=request_func_input)))

        # Gather the results of all tasks
        try:
            outputs = await asyncio.gather(*tasks)
        except Exception as e:
            raise e
        await asyncio.get_event_loop().run_in_executor(None, result_queue.put, outputs)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(handle_requests())
    finally:
        loop.close()
        loop.stop()


async def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
    disable_tqdm: bool,
    request_duration: float,
    scheduler_policy: Optional[str],
    data,
    phase,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    benchmark_start_time = time.perf_counter()

    num_workers = 20
    workers = []

    for i in range(num_workers):
        worker = multiprocessing.Process(target=process_requests, args=(backend, args, request_func))
        worker.start()
        workers.append(worker)

    send_request_num = 0
    async for request in get_request_duration(input_requests, request_rate, request_duration, scheduler_policy):
        prompt, prompt_len, output_len = request
        min_tokens = output_len if scheduler_policy in ["srjf", "sjf"] else None
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            best_of=best_of,
            use_beam_search=use_beam_search,
            min_tokens=min_tokens,
        )
        send_request_num += 1
        request_queue.put(request_func_input)

    for _ in range(num_workers):
        request_queue.put(None)

    outputs: List[RequestFuncOutput] = []
    while True:
        try:
            if len(outputs) == send_request_num:
                break
            result = result_queue.get()
            for res in result:
                pbar.update(1)
                outputs.append(res)
        except Exception as error:
            raise error

    for worker in workers:
        worker.join()

    if not disable_tqdm:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
    )

    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Phase:", phase))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):", metrics.input_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", metrics.output_throughput))
    print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):", metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print("{s:{c}^{n}}".format(s="Time per Output Token (excl. 1st token)", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):", metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("{s:{c}^{n}}".format(s="Inter-token Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
    print("{s:{c}^{n}}".format(s="Request Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean Lat. (ms):", metrics.mean_lat_ms))
    print("{:<40} {:<10.2f}".format("Median Lat. (ms):", metrics.median_lat_ms))
    print("{:<40} {:<10.2f}".format("P99 Lat. (ms):", metrics.p99_lat_ms))
    print("{:<40} {:<10.2f}".format("Mean Avg Lat. (ms):", metrics.mean_avg_lat_ms))
    print("{:<40} {:<10.2f}".format("Median Avg Lat. (ms):", metrics.mean_avg_lat_ms))
    print("{:<40} {:<10.2f}".format("P99 Avg Lat. (ms):", metrics.p99_avg_lat_ms))
    print("=" * 50)

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "std_ttft_ms": metrics.std_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "std_tpot_ms": metrics.std_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "p99_itl_ms": metrics.p99_itl_ms,
        "mean_lat_ms": metrics.mean_lat_ms,
        "median_lat_ms": metrics.median_lat_ms,
        "p99_lat_ms": metrics.p99_lat_ms,
        "std_itl_ms": metrics.std_itl_ms,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        # "generated_texts": [output.generated_text for output in outputs],
        # "errors": [output.error for output in outputs],
        "latencies": [output.latency for output in outputs],
        "avg_token_latency":[output.latency/output.actual_output_len for output in outputs],
    }
    return result, outputs


def check_health(api_url: str) -> bool:
    import requests

    ready = False
    while not ready:
        try:
            check_health_url = api_url + "/health"
            r = requests.get(check_health_url)
            if r.status_code == 200:
                ready = True
        except Exception:
            continue


def main(args: argparse.Namespace):
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    phase = args.phase
    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"

    tokenizer = get_tokenizer(tokenizer_id, trust_remote_code=args.trust_remote_code)
    check_health(f"http://{args.host}:{args.port}")
    if args.dataset is not None:
        warnings.warn(
            "The '--dataset' argument will be deprecated in the next "
            "release. Please use '--dataset-name' and "
            "'--dataset-path' in the future runs.",
            stacklevel=2,
        )
        input_requests, data = sample_sharegpt_requests(
            dataset_path=args.dataset,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
            scheduler_policy=args.scheduler_policy,
        )

    elif args.dataset_name == "sharegpt":
        input_requests, data = sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
            scheduler_policy=args.scheduler_policy,
            phase=phase,
        )
    elif args.dataset_name == "alpaca":
        input_requests, data = sample_alpaca_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
            scheduler_policy=args.scheduler_policy,
        )
    elif args.dataset_name == "leval":
        input_requests, data = sample_leval_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
            scheduler_policy=args.scheduler_policy,
        )
    elif args.dataset_name == "lmsys":
        input_requests, data = sample_lmsys_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            fixed_output_len=args.sharegpt_output_len,
            scheduler_policy=args.scheduler_policy,
            phase=phase,
        )
    elif args.dataset_name == "sonnet":
        # Do not format the prompt, pass to message directly
        if args.backend == "openai-chat":
            input_requests = sample_sonnet_requests(
                dataset_path=args.dataset_path,
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
            )
            input_requests = [
                (prompt, prompt_len, output_len) for prompt, prompt_formatted, prompt_len, output_len in input_requests
            ]
        else:
            assert tokenizer.chat_template or tokenizer.default_chat_template, (
                "Tokenizer/model must have chat template for sonnet dataset."
            )
            input_requests = sample_sonnet_requests(
                dataset_path=args.dataset_path,
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
            )
            input_requests = [
                (prompt_formatted, prompt_len, output_len)
                for prompt, prompt_formatted, prompt_len, output_len in input_requests
            ]

    elif args.dataset_name == "random":
        input_requests = sample_random_requests(
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            num_prompts=args.num_prompts,
            range_ratio=args.random_range_ratio,
            tokenizer=tokenizer,
        )

    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    benchmark_result, outputs = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            best_of=args.best_of,
            use_beam_search=args.use_beam_search,
            request_rate=args.request_rate,
            disable_tqdm=args.disable_tqdm,
            request_duration=args.request_duration,
            scheduler_policy=args.scheduler_policy,
            data=data,
            phase=phase,
        )
    )

    # Save config and results to json
    if args.save_result:
        result_json: Dict[str, Any] = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        seconds = datetime.now().strftime("%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["best_of"] = args.best_of
        result_json["use_beam_search"] = args.use_beam_search
        result_json["num_prompts"] = args.num_prompts
        result_json['phase']=args.phase
        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError("Invalid metadata format. Please use KEY=VALUE format.")

        # Traffic
        result_json["request_rate"] = args.request_rate if args.request_rate < float("inf") else "inf"

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}
        parallel_type = result_json["parallel_type"]
        # Save to file
        base_model_id = model_id.split("/")[-1]
        file_name = (
            f"{backend}-{args.request_rate}qps-{base_model_id}-{seconds}-{parallel_type}-{args.scheduler_policy}.json"  # noqa
        )
        if args.result_dir:
            if not os.path.exists(os.path.join(args.result_dir)):
                os.makedirs(os.path.join(args.result_dir))
            file_name = os.path.join(args.result_dir, file_name)
        with open(file_name, "w") as outfile:
            json.dump(result_json, outfile)

        if args.scheduler_policy == "tfittradeoff":
            prompt_output_lens_json = {"prompts": [], "output_lens": []}
            for i in range(len(outputs)):
                prompt = outputs[i].prompt
                output_len = benchmark_result["output_lens"][i]
                prompt_output_lens_json["prompts"].append(prompt)
                prompt_output_lens_json["output_lens"].append(output_len)
            prompt_output_lens_file_name = (
                f"prompt_output_{backend}-{args.request_rate}qps-{base_model_id}-{seconds}-{args.scheduler_policy}.json"  # noqa: E501
            )

            if args.result_dir:
                if not os.path.exists(os.path.join(args.result_dir, "prompt")):
                    os.makedirs(os.path.join(args.result_dir, "prompt"))
                prompt_output_lens_file_name = os.path.join(args.result_dir, "prompt", prompt_output_lens_file_name)  # noqa: E501
            with open(prompt_output_lens_file_name, "w") as prompt_output_lens_file_name_outfile:
                json.dump(prompt_output_lens_json, prompt_output_lens_file_name_outfile)
    save_output(outputs=outputs, filename=f"output_dir/output-{args.dataset_name}-{base_model_id}.jsonl")


if __name__ == "__main__":
    st = time.time()
    count = 0
    request_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    lock = multiprocessing.Lock()
    RESULTLEN = 0
    get_azure_request_data()
    # parser = FlexibleArgumentParser(description="Benchmark the online serving throughput.")
    # parser.add_argument(
    #     "--backend",
    #     type=str,
    #     default="vllm",
    #     choices=list(ASYNC_REQUEST_FUNCS.keys()),
    # )
    # parser.add_argument(
    #     "--base-url",
    #     type=str,
    #     default=None,
    #     help="Server or API base url if not using http host and port.",
    # )
    # parser.add_argument("--host", type=str, default="10.119.46.53")
    # parser.add_argument("--port", type=int, default=8000)
    # parser.add_argument(
    #     "--endpoint",
    #     type=str,
    #     default="/v1/completions",
    #     help="API endpoint.",
    # )
    # parser.add_argument(
    #     "--dataset",
    #     type=str,
    #     default=None,
    #     help="Path to the ShareGPT dataset, will be deprecated in the next release.",
    # )
    # parser.add_argument(
    #     "--dataset-name",
    #     type=str,
    #     default="sharegpt",
    #     choices=["sharegpt", "sonnet", "random", "alpaca", "leval", "lmsys"],
    #     help="Name of the dataset to benchmark on.",
    # )
    # parser.add_argument("--dataset-path", type=str, default=None, help="Path to the dataset.")
    # parser.add_argument(
    #     "--model",
    #     type=str,
    #     required=True,
    #     help="Name of the model.",
    # )
    # parser.add_argument(
    #     "--tokenizer",
    #     type=str,
    #     help="Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    # )
    # parser.add_argument(
    #     "--best-of",
    #     type=int,
    #     default=1,
    #     help="Generates `best_of` sequences per prompt and returns the best one.",
    # )
    # parser.add_argument("--use-beam-search", action="store_true")
    # parser.add_argument(
    #     "--num-prompts",
    #     type=int,
    #     default=1000,
    #     help="Number of prompts to process.",
    # )
    # parser.add_argument(
    #     "--sharegpt-output-len",
    #     type=int,
    #     default=None,
    #     help="Max output length for each request. Overrides the output length from the ShareGPT dataset.",
    # )
    # parser.add_argument(
    #     "--sonnet-input-len",
    #     type=int,
    #     default=550,
    #     help="Number of input tokens per request, used only for sonnet dataset.",
    # )
    # parser.add_argument(
    #     "--sonnet-output-len",
    #     type=int,
    #     default=150,
    #     help="Number of output tokens per request, used only for sonnet dataset.",
    # )
    # parser.add_argument(
    #     "--sonnet-prefix-len",
    #     type=int,
    #     default=200,
    #     help="Number of prefix tokens per request, used only for sonnet dataset.",
    # )
    # parser.add_argument(
    #     "--random-input-len",
    #     type=int,
    #     default=1024,
    #     help="Number of input tokens per request, used only for random sampling.",
    # )
    # parser.add_argument(
    #     "--random-output-len",
    #     type=int,
    #     default=128,
    #     help="Number of output tokens per request, used only for random sampling.",
    # )
    # parser.add_argument(
    #     "--random-range-ratio",
    #     type=float,
    #     default=1.0,
    #     help="Range of sampled ratio of input/output length, used only for random sampling.",
    # )
    # parser.add_argument(
    #     "--request-rate",
    #     type=float,
    #     default=float("inf"),
    #     help="Number of requests per second. If this is inf, "
    #     "then all the requests are sent at time 0. "
    #     "Otherwise, we use Poisson process to synthesize "
    #     "the request arrival times.",
    # )
    # parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument(
    #     "--trust-remote-code",
    #     action="store_true",
    #     help="Trust remote code from huggingface",
    # )
    # parser.add_argument(
    #     "--disable-tqdm",
    #     action="store_true",
    #     help="Specify to disable tqdm progress bar.",
    # )
    # parser.add_argument(
    #     "--save-result",
    #     action="store_true",
    #     help="Specify to save benchmark results to a json file",
    # )
    # parser.add_argument(
    #     "--metadata",
    #     metavar="KEY=VALUE",
    #     nargs="*",
    #     help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
    #     "for metadata of this run to be saved in the result JSON file "
    #     "for record keeping purposes.",
    # )
    # parser.add_argument(
    #     "--result-dir",
    #     type=str,
    #     default=None,
    #     help="Specify directory to save benchmark json results."
    #     "If not specified, results are saved in the current directory.",
    # )
    # parser.add_argument(
    #     "--result-filename",
    #     type=str,
    #     default=None,
    #     help="Specify the filename to save benchmark json results."
    #     "If not specified, results will be saved in "
    #     "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
    #     " format.",
    # )

    # parser.add_argument(
    #     "--request-duration",
    #     type=float,
    #     default=float("inf"),
    #     help="the duration of sending requests(Seconds)",
    # )
    # parser.add_argument(
    #     "--max-serving-time",
    #     type=int,
    #     default=-1,
    #     help="the maximum serving time(Seconds)",
    # )
    # parser.add_argument(
    #     "--scheduler-policy",
    #     type=str,
    #     default="fcfs",
    #     choices=["fcfs", "infer", "sjmlfq", "inferpreempt", "sjf", "srjf", "tfittradeoff", "las", "opt"],
    #     help="Specify the scheduler policy.",
    # )
    # parser.add_argument("--execution-counter", type=int, default=0, help="Specify the execution counter.")
    # parser.add_argument(
    #     "--phase",
    #     type=str,
    #     required=True,
    #     choices=["hybrid", "prefill", "decode"],
    #     help="Specify the phase of the benchmark. Prefill only handle prefill phase, Decode only handle decode phase, Hybrid handle both phases.",
    # )
    # args = parser.parse_args()
    
    # main(args)
