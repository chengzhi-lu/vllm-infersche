import argparse
from queue import Queue
import threading
from typing import List, Tuple

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm import RequestOutput, LLMEngine
import pandas as pd
import os
import numpy as np
from utils import Utils
from rich import print
from rich import pretty
from rich.progress import track
import traceback


pretty.install()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_requests(model_name, dataset_name) -> List[Tuple[str, SamplingParams, int]]:
    init_seq = []
    saved_seq = Utils.load_seq_from_file(BASE_DIR, "seq_data", f"{model_name}_{dataset_name}.json")
    for count in saved_seq:
        prompt, prompt_len = saved_seq[count]
        init_seq.append(
            (
                prompt,
                SamplingParams(
                    temperature=0.0,
                    repetition_penalty=2,
                    logprobs=1,
                    min_tokens=1,
                    max_tokens=2000,
                ),
                prompt_len,
            )
        )

    return init_seq


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def parse_batched_result(request_outputs: List[RequestOutput]):
    _results = []
    for request_output in request_outputs:
        request_id = request_output.request_id
        prompt_len = len(request_output.prompt_token_ids)
        output = request_output.outputs[0]
        output_len = len(output.token_ids)
        if output_len == 0:
            continue
        log_prob = np.exp(output.logprobs[-1][2].logprob)
        rank = request_output.outputs[0].logprobs[-1][2].rank
        _results.append([request_id, prompt_len, output_len, log_prob, rank])
    return _results


def put_requests(all_inputs: List[Tuple[str, SamplingParams, int]], input_queue: Queue):
    for seq in all_inputs:
        prompt, sampling_params, prompt_len = seq
        input_queue.put((prompt, sampling_params, prompt_len), block=True)


def inference_backend(engine: LLMEngine, input_queue: Queue):
    count = 0
    while count < batch_size:
        prompt, sampling_params, prompt_len = input_queue.get()
        engine.add_request(
            request_id=count,
            inputs=prompt,
            params=sampling_params,
        )
        count += 1
    while engine.has_unfinished_requests():
        request_outputs: List[RequestOutput] = engine.step()
        _result = parse_batched_result(request_outputs)
        if engine.has_finished_seq:
            while (
                not input_queue.empty()
                and len(engine.scheduler[0].waiting)
                + len(engine.scheduler[0].running)
                + len(engine.scheduler[0].swapped)
                <= batch_size
            ):
                prompt, sampling_params, prompt_len = input_queue.get()
                engine.add_request(
                    request_id=count,
                    inputs=prompt,
                    params=sampling_params,
                )
                count += 1
        engine.has_finished_seq = False
        yield _result


def main(
    max_token_num: int,
    batch_size: int,
    policy: str = "fcfs",
    preemption_mode: str = "swap",
    model_name: str = "llama",
    dataset_name: str = "sharegpt",
):
    """Main function that sets up and runs the prompt processing."""
    parser = argparse.ArgumentParser(description="Demo on using the LLMEngine class directly")

    parser = EngineArgs.add_cli_args(parser)
    args: argparse.Namespace = parser.parse_args()
    if model_name == "llama":
        args.model = "meta-llama/Llama-2-13b-chat-hf"
    elif model_name == "mistral":
        args.model = "mistralai/Mistral-7B-Instruct-v0.1"
    args.swap_space = 16
    args.max_num_seqs = batch_size
    args.scheduler_policy = policy
    args.default_preemption_mode = preemption_mode
    args.enable_chunked_prefill = True
    args.max_num_batched_tokens = max_token_num
    args.disable_sliding_window = True
    try:
        all_inputs = get_requests(model_name, dataset_name)
        engine = initialize_engine(args)
    except Exception as e:
        traceback.print_exc()
        print(e)
    input_queue = Queue(maxsize=batch_size)
    eos_result = []
    # split inputs into batches with batch_size=16
    # split_batch_size = 256
    # seqs = [
    #     all_inputs[i : i + split_batch_size]
    #     for i in range(0, len(all_inputs), split_batch_size)
    # ]
    # 创建线程执行put_requests函数
    t = threading.Thread(target=put_requests, args=(all_inputs, input_queue))
    t.start()
    for _result in inference_backend(engine, input_queue=input_queue):
        eos_result.extend(_result)
        if (len(eos_result) // batch_size) % 10 == 0:
            result_df = pd.DataFrame(
                eos_result,
                columns=[
                    "request_id",
                    "prompt_len",
                    "token_num",
                    "eos_prob",
                    "eos_token_rank",
                ],
            )
            result_df.to_csv(
                os.path.join(
                    BASE_DIR,
                    "data",
                    "eos_result",
                    f"{model_name}_{dataset_name}_eos_prob_result.csv",
                ),
                index=False,
            )

    # for input_seqs in track(seqs, description="Predicting eos position..."):
    #     for seq in input_seqs:
    #         prompt, sampling_params, prompt_len = seq
    #         engine.add_request(
    #             request_id=all_inputs.index(seq),
    #             inputs=prompt,
    #             params=sampling_params,
    #         )
    #     while engine.has_unfinished_requests():
    #         request_outputs: List[RequestOutput] = engine.step()
    #         _result = parse_batched_result(request_outputs)
    #         eos_result.extend(_result)
    # result_df = pd.DataFrame(
    #     eos_result,
    #     columns=[
    #         "request_id",
    #         "prompt_len",
    #         "token_num",
    #         "eos_prob",
    #         "eos_token_rank",
    #     ],
    # )
    # result_df.to_csv(os.path.join(BASE_DIR, f"{model_name}_{dataset_name}_eos_prob_result.csv"), index=False)
    return


def main_test(
    max_token_num: int,
    batch_size: int,
    policy: str = "fcfs",
    preemption_mode: str = "swap",
    model_name: str = "llama",
    dataset_name: str = "sharegpt",
):
    """Main function that sets up and runs the prompt processing."""
    parser = argparse.ArgumentParser(description="Demo on using the LLMEngine class directly")

    parser = EngineArgs.add_cli_args(parser)
    args: argparse.Namespace = parser.parse_args()
    args.model = "meta-llama/Llama-2-13b-chat-hf"
    args.swap_space = 16
    args.max_num_seqs = batch_size
    args.scheduler_policy = policy
    args.default_preemption_mode = preemption_mode
    args.enable_chunked_prefill = True
    args.disable_sliding_window = True
    args.max_num_batched_tokens = max_token_num
    try:
        all_inputs = get_requests(model_name, dataset_name)
        engine = initialize_engine(args)
    except Exception as e:
        traceback.print_exc()
        print(e)
    for i in range(10):
        prompt, sampling_params, prompt_len = all_inputs[0]
        engine.add_request(
            request_id=0,
            inputs=prompt,
            params=sampling_params,
        )
        while engine.has_unfinished_requests():
            request_outputs: List[RequestOutput] = engine.step()
        print(len(request_outputs[0].outputs[0].token_ids) + prompt_len)


if __name__ == "__main__":
    test_type = "infer_schedule_policy_test"
    models = ["llama"]
    datasets = ["alpaca"]
    rerun = True
    max_token_nums = [4096]
    batch_sizes = [512]
    total_iter_result, total_request_result = Utils.load_tmp_result(test_type, BASE_DIR)
    preemption_mode = "swap"
    policies = ["fcfs"]
    # If prefill mode is horizonal, the sequences length is equals to the token nums, otherwise, the batch size equals to the token nums  # noqa: E501
    for batch_size in batch_sizes:
        for policy in policies:
            for model in models:
                for dataset in datasets:
                    for max_token_num in max_token_nums:
                        try:
                            main(
                                max_token_num=max_token_num,
                                batch_size=batch_size,
                                policy=policy,
                                preemption_mode=preemption_mode,
                                model_name=model,
                                dataset_name=dataset,
                            )
                            # main_test(
                            #     max_token_num=max_token_num,
                            #     batch_size=batch_size,
                            #     policy=policy,
                            #     preemption_mode=preemption_mode,
                            # )
                        except Exception as e:
                            traceback.print_exc()
                            print(e)
