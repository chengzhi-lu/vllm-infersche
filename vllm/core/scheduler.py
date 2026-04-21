import enum
from math import ceil
import os
import random
from itertools import accumulate
import bisect
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union, ClassVar
from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.batch_solver import BatchSolver
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.core.policy import Policy, PolicyFactory, PolicyInfo
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceStatus, SequenceType)
import pandas as pd
logger = init_logger(__name__)

# Test-only. If configured, decode is preempted with
# ARTIFICIAL_PREEMPTION_PROB% probability.
ENABLE_ARTIFICIAL_PREEMPT = bool(
    os.getenv("VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT", False))  # noqa
ARTIFICIAL_PREEMPTION_PROB = 0.5
ARTIFICIAL_PREEMPTION_MAX_CNT = 500


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()
    KV_FREE_RECOMPUTE = enum.auto()


class SwapMode(enum.Enum):
    """Swap modes.

    1. Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.

    """
    FULL = enum.auto()
    PARTIAL = enum.auto()


class PreemptionReason(enum.Enum):
    """ Preemption reasons.
    1. Exhausted token budget
    2. Exhausted sequence budget
    3. All preempted sequences are exhausted
    4. No preempted sequences
    
    """
    BUDGET_EXHAUSTED = enum.auto()
    SEQ_NUM_EXHAUSTED = enum.auto()
    ALL_EXHAUSTED = enum.auto()
    NONE = enum.auto()


@dataclass
class SchedulingBudget:
    """The available slots for scheduling.

    TODO(sang): Right now, the budget is request_id-aware meaning it can ignore
    budget update from the same request_id. It is because in normal scheduling
    path, we update RUNNING num_seqs ahead of time, meaning it could be
    updated more than once when scheduling RUNNING requests. Since this won't
    happen if we only have chunked prefill scheduling, we can remove this
    feature from the API when chunked prefill is enabled by default.
    """
    token_budget: int
    max_num_seqs: int
    _request_ids_num_batched_tokens: Set[str] = field(default_factory=set)
    _request_ids_num_curr_seqs: Set[str] = field(default_factory=set)
    _num_batched_tokens: int = 0
    _num_curr_seqs: int = 0

    def can_schedule_infer(self, *, num_new_tokens: int,
                           num_new_seqs: int) -> PreemptionReason:
        request_tokens = self.num_batched_tokens + num_new_tokens
        request_seqs = self.num_curr_seqs + num_new_seqs
        if request_tokens >= self.token_budget and request_seqs >= self.max_num_seqs:
            return PreemptionReason.ALL_EXHAUSTED
        elif request_tokens >= self.token_budget and request_seqs < self.max_num_seqs:
            return PreemptionReason.BUDGET_EXHAUSTED
        elif request_tokens <= self.token_budget and request_seqs >= self.max_num_seqs:
            return PreemptionReason.SEQ_NUM_EXHAUSTED
        else:
            return PreemptionReason.NONE

    def can_schedule(self, *, num_new_tokens: int, num_new_seqs: int):
        # assert num_new_tokens != 0
        # assert num_new_seqs != 0
        return (self.num_batched_tokens + num_new_tokens <= self.token_budget
                and self.num_curr_seqs + num_new_seqs <= self.max_num_seqs)


    def remaining_token_budget(self):
        return self.token_budget - self.num_batched_tokens

    def add_num_batched_tokens(self, req_id: str, num_batched_tokens: int):
        if req_id in self._request_ids_num_batched_tokens:
            return

        self._request_ids_num_batched_tokens.add(req_id)
        self._num_batched_tokens += num_batched_tokens

    def subtract_num_batched_tokens(self, req_id: str,
                                    num_batched_tokens: int):
        if req_id in self._request_ids_num_batched_tokens:
            self._request_ids_num_batched_tokens.remove(req_id)
            self._num_batched_tokens -= num_batched_tokens

    def subtract_num_batched_tokens_partial(self, num_batched_tokens: int):
        self._num_batched_tokens -= num_batched_tokens

    def add_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            return

        self._request_ids_num_curr_seqs.add(req_id)
        self._num_curr_seqs += num_curr_seqs

    def subtract_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            self._request_ids_num_curr_seqs.remove(req_id)
            self._num_curr_seqs -= num_curr_seqs
        
    def update_token_budget(self, new_token_budget: int):
        self.token_budget = new_token_budget
    
    def update_max_num_seqs(self, new_max_num_seqs: int):
        self.max_num_seqs = new_max_num_seqs

    @property
    def num_batched_tokens(self):
        return self._num_batched_tokens

    @property
    def num_curr_seqs(self):
        return self._num_curr_seqs


@dataclass
class ScheduledSequenceGroup:
    # A sequence group that's scheduled.
    seq_group: SequenceGroup
    # The total chunk size (number of tokens) to process for next iteration.
    # 1 for decoding. Same as prompt tokens for prefill, but if prefill is
    # chunked, it can be smaller than that.
    token_chunk_size: int


@dataclass
class SchedulerOutputs:
    """The scheduling decision made from a scheduler."""
    # Scheduled sequence groups.
    scheduled_seq_groups: Iterable[ScheduledSequenceGroup]
    # Number of prefill groups scheduled.
    num_prefill_groups: int
    # Total number of batched tokens.
    num_batched_tokens: int
    # Blocks to swap in. List of CPU -> GPU block number.
    blocks_to_swap_in: List[Tuple[int, int]]
    # Blocks to swap out. List of GPU -> CPU block number.
    blocks_to_swap_out: List[Tuple[int, int]]
    # Blocks to copy. Source to dest block.
    blocks_to_copy: List[Tuple[int, int]]
    # Sequence groups that are going to be ignored.
    ignored_seq_groups: List[SequenceGroup]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int
    # The number of requests in the running queue
    running_queue_size: int
    preempted: int
    num_waiting_to_running: int
    num_running_to_waiting: int
    recomputed_token_nums: int

    need_score: bool 
    allow_both_swap: bool 

    def __post_init__(self):
        # Swap in and swap out should never happen at the same time.
        # assert not (self.blocks_to_swap_in and self.blocks_to_swap_out)

        self.num_loras: int = len(self.lora_requests)
        if self.num_loras > 0:
            self._sort_by_lora_ids()

        self.num_prompt_adapters: int = len(self.prompt_adapter_requests)
        assert self.allow_both_swap or (not (self.blocks_to_swap_in and self.blocks_to_swap_out))
        
    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)

    def _sort_by_lora_ids(self):
        self.scheduled_seq_groups = sorted(
            self.scheduled_seq_groups,
            key=lambda g: (g.seq_group.lora_int_id, g.seq_group.request_id))

    @property
    def lora_requests(self) -> Set[LoRARequest]:
        return {
            g.seq_group.lora_request
            for g in self.scheduled_seq_groups
            if g.seq_group.lora_request is not None
        }

    @property
    def prompt_adapter_requests(self) -> Set[PromptAdapterRequest]:
        return {
            g.seq_group.prompt_adapter_request
            for g in self.scheduled_seq_groups
            if g.seq_group.prompt_adapter_request is not None
        }


@dataclass
class SchedulerPreemption:
    decode_seq_groups_running: List[SequenceGroup]
    decode_seq_groups_swapped: List[SequenceGroup]
    prefill_seq_groups_running: List[SequenceGroup]
    prefill_seq_groups_swapped: List[SequenceGroup]
    preempted_running: List[SequenceGroup]
    swapped_out_running: List[SequenceGroup]
    blocks_to_swap_in: List[Tuple[int, int]]
    blocks_to_swap_out: List[Tuple[int, int]]
    blocks_to_copy_running: List[Tuple[int, int]]
    blocks_to_copy_swapped: List[Tuple[int, int]]
    infeasible_seq_groups: List[SequenceGroup]
    ignored_seq_groups: List[SequenceGroup]
    seq_groups_prefill: List[SequenceGroup]
    num_lookahead_slots_running: int
    num_lookahead_slots_swapped: int
    num_lookahead_slots_prefill: int


@dataclass
class SchedulerRunningOutputs:
    """The requests that are scheduled from a running queue.

    Could contain prefill (prefill that's chunked) or decodes. If there's not
    enough memory, it can be preempted (for recompute) or swapped out.
    """
    # Selected sequences that are running and in a decoding phase.
    decode_seq_groups: List[SequenceGroup]
    # Selected sequences that are running and in a prefill phase.
    # I.e., it means the prefill has been chunked.
    prefill_seq_groups: List[SequenceGroup]
    # The preempted sequences.
    preempted: List[SequenceGroup]
    # Sequences that are swapped out.
    swapped_out: List[SequenceGroup]
    # The blocks to swap out.
    blocks_to_swap_out: List[Tuple[int, int]]
    # The blocks to copy.
    blocks_to_copy: List[Tuple[int, int]]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int

    @classmethod
    def create_empty(cls) -> "SchedulerRunningOutputs":
        return SchedulerRunningOutputs(
            decode_seq_groups=[],
            prefill_seq_groups=[],
            preempted=[],
            swapped_out=[],
            blocks_to_swap_out=[],
            blocks_to_copy=[],
            num_lookahead_slots=0,
        )


@dataclass
class SchedulerSwappedInOutputs:
    """The requests that are scheduled from a swap queue.

    Could contain prefill (prefill that's chunked) or decodes.
    """
    # Selected sequences that are going to be swapped in and is in a
    # decoding phase.
    decode_seq_groups: List[SequenceGroup]
    # Selected sequences that are going to be swapped in and in a prefill
    # phase. I.e., it means the prefill has been chunked.
    prefill_seq_groups: List[SequenceGroup]
    # The blocks to swap in.
    blocks_to_swap_in: List[Tuple[int, int]]
    # The blocks to copy.
    blocks_to_copy: List[Tuple[int, int]]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int
    # Infeasible sequence groups.
    infeasible_seq_groups: List[SequenceGroup]

    @classmethod
    def create_empty(cls) -> "SchedulerSwappedInOutputs":
        return SchedulerSwappedInOutputs(
            decode_seq_groups=[],
            prefill_seq_groups=[],
            blocks_to_swap_in=[],
            blocks_to_copy=[],
            num_lookahead_slots=0,
            infeasible_seq_groups=[],
        )


@dataclass
class SchedulerPrefillOutputs:
    """The requests that are scheduled from a waiting queue.

    Could contain a fresh prefill requests or preempted requests that need
    to be recomputed from scratch.
    """
    # Selected sequences for prefill.
    seq_groups: List[SequenceGroup]
    kv_free_seq_groups: List[SequenceGroup]
    # Ignored sequence groups.
    ignored_seq_groups: List[SequenceGroup]
    num_lookahead_slots: int

    @classmethod
    def create_empty(cls) -> "SchedulerPrefillOutputs":
        return SchedulerPrefillOutputs(
            seq_groups=[],
            kv_free_seq_groups=[],
            ignored_seq_groups=[],
            num_lookahead_slots=0,
        )

@dataclass
class SchedulerMetric:
    scheduler_start_time: float = 0.0
    scheduler_end_time: float = 0.0
    total_swap_out_blocks: int = 0
    total_swap_in_blocks: int = 0
    total_swap_out_seqs: int = 0
    total_swap_in_seqs: int = 0
    total_low_eff_swap_out: int = 0
    total_low_eff_swap_out_diff: int = 0
    schedule_running_time: float = 0.0
    schedule_waiting_time: float = 0.0
    schedule_swapped_time: float = 0.0
    prefill_token_num: int = 0
    decode_token_num: int = 0
    gpu_memory_occupy: float = 0
    gpu_computation_occupy: float = 0
    running_seq_nums: int = 0
    waiting_seq_nums: int = 0
    pending_seq_nums: int = 0
    total_count: int = 0
    execution_time: float = 0.0
    schedule_time: float = 0.0
    swap_time: float = 0.0
    handle_output_time: float = 0.0
    scheduler_index: int = 0

    @classmethod
    def to_dataframe(cls, metrics_list: list['SchedulerMetric'] = None) -> pd.DataFrame:
        """将类变量或实例列表转换为DataFrame
        
        Args:
            metrics_list: 如果传入实例列表，则生成多行DataFrame
                        如果为None，则用当前类变量生成单行DataFrame
        """
        if metrics_list is None:
            data = {k: [v] for k, v in cls.__dict__.items() 
                   if not k.startswith('__') and isinstance(v, (int, float))}
        else:
            data = [asdict(m) for m in metrics_list]
        
        return pd.DataFrame(data)


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        pipeline_parallel_size: int = 1,
        batch_solver: Optional[BatchSolver] = None,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        # Note for LoRA scheduling: the current policy is extremely
        # simple and NOT fair. It can lead to starvation of some
        # LoRAs. This should be improved in the future.
        self.lora_config = lora_config

        version = "v1"
        if self.scheduler_config.use_v2_block_manager:
            version = "v2"
        if self.scheduler_config.embedding_mode:
            version = "embedding"

        BlockSpaceManagerImpl = BlockSpaceManager.get_block_space_manager_class(
            version)
        self.ddl = None
        self.reach_ddl = False

        if self.scheduler_config.policy == 'tfittradeoff':
            # self.num_shared_blocks = min(int(self.scheduler_config.max_num_batched_tokens 
            # // self.cache_config.block_size 
            # // self.scheduler_config.waiting_iter_base),      
            # int(self.cache_config.num_gpu_blocks*0.1)+1)
            self.num_shared_blocks = self.cache_config.num_shared_blocks
        else:
            self.num_shared_blocks = 0
        
        
        # Create the block space manager.
        num_gpu_blocks = cache_config.num_gpu_blocks
        if num_gpu_blocks:
            num_gpu_blocks //= pipeline_parallel_size

        num_cpu_blocks = cache_config.num_cpu_blocks
        if num_cpu_blocks:
            num_cpu_blocks //= pipeline_parallel_size

        # Create the block space manager.
        self.block_manager = BlockSpaceManagerImpl(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            num_shared_blocks=self.num_shared_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching)

        # Sequence groups in the WAITING state.
        # Contain new prefill or preempted requests.
        self.waiting: Deque[SequenceGroup] = deque()
        # Sequence groups in the RUNNING state.
        # Contain decode requests.
        self.running: Deque[SequenceGroup] = deque()
        # Sequence groups in the SWAPPED state.
        # Contain decode requests that are swapped out.
        self.swapped: Deque[SequenceGroup] = deque()

        self.high_priority_seq_groups: List[SequenceGroup] = [] # cache seqs with high priority in the waiting 
                                                             #queue and swap them into the running queue. 
        self.low_priority_seq_groups: List[SequenceGroup] = [] # cache seqs with low priority in the running queue  
                                                            #and swap them out to the waiting queue. 


        self.kv_free_seq_groups: List[str] = []

        # Sequence groups finished requests ids since last step iteration.
        # It lets the model know that any state associated with these requests
        # can and must be released after the current step.
        self._finished_requests_ids: List[str] = list()
        # Time at previous scheduling step
        self.prev_time = 0.0
        # Did we schedule a prompt at previous step?
        self.prev_prompt = False
        # Latency of the last prompt step
        self.last_prompt_latency = 0.0
        # preemption mode, RECOMPUTE or SWAP
        self.user_specified_preemption_mode = scheduler_config.preemption_mode

        # The following field is test-only. It is used to inject artificial
        # preemption.
        self.enable_artificial_preemption = ENABLE_ARTIFICIAL_PREEMPT
        self.artificial_preempt_cnt = (ARTIFICIAL_PREEMPTION_MAX_CNT
                                       if self.enable_artificial_preemption
                                       else 0)
        self.num_cumulative_preemption: int = 0
        self.preemption_mode: PreemptionMode = PreemptionMode.RECOMPUTE

        if self.scheduler_config.preemption_mode == "swap":
            self.preemption_mode = PreemptionMode.SWAP
        elif self.scheduler_config.preemption_mode == "recompute":
            self.preemption_mode = PreemptionMode.RECOMPUTE
        # partial swapped dict: key is sequence group, value is a tuple of (remaining block sizes, preempted_seq_group)
        self.partial_swapped: Dict[str, Tuple[int, SequenceGroup]] = {}

        self.partial_swapped_values: List[Tuple[int, str]] = []
        self.seq_group_for_preempted: Tuple[SequenceGroup, int] = (None, 0)


        self.seq_group_for_pre_swap_in: List[SequenceGroup] = []
        self.seq_group_for_pre_swap_out: List[SequenceGroup] = []        

        
        self.has_finished_seqs = False
        self.partial_swap_out_flag = self.scheduler_config.swap_out_tokens_policy == "partial"
        self.partial_swapped_rate = self.scheduler_config.swap_out_partial_rate
        self.scheduler_metric=SchedulerMetric()
        self.batch_solver = batch_solver
        self.aux_model = None
        self.need_score = True
        self.tbound = -1
        self.starv = -1 
        self.period = -1  
        self.fake_allocate = self.scheduler_config.fake_allocate
        self.prompt_limit = min(self.scheduler_config.max_model_len,
                               self.scheduler_config.max_num_batched_tokens)
        self.virtual_engine = 0

    @property
    def lora_enabled(self) -> bool:
        return bool(self.lora_config)

    @property
    def num_decoding_tokens_per_seq(self) -> int:
        """The number of new tokens."""
        return 1

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)
        seq_group.idle = 0
        seq_group.runs = 0
        seq_group.pri = 0


    def set_virtual_engine(self, virtual_engine: int):
        self.virtual_engine= virtual_engine

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a sequence group with the given ID.

        Check if the sequence group with the given ID
            is present in any of the state queue.
        If present, remove the sequence group from the state queue.
            Also, if any of the sequences in the sequence group is not finished,
                free the sequence with status `FINISHED_ABORTED`.
        Otherwise, do nothing.

        Args:
            request_id: The ID(s) of the sequence group to abort.
        """
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running, self.swapped]:
            aborted_groups: List[SequenceGroup] = []
            for seq_group in state_queue:
                if not request_ids:
                    # Using 'break' here may add two extra iterations,
                    # but is acceptable to reduce complexity.
                    break
                if seq_group.request_id in request_ids:
                    # Appending aborted group into pending list.
                    aborted_groups.append(seq_group)
                    request_ids.remove(seq_group.request_id)
            for aborted_group in aborted_groups:
                # Remove the sequence group from the state queue.
                state_queue.remove(aborted_group)
                for seq in aborted_group.get_seqs():
                    if seq.is_finished():
                        continue
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.free_seq(seq)

    def has_unfinished_seqs(self) -> bool:
        return len(self.waiting) != 0 or len(self.running) != 0 or len(
            self.swapped) != 0

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def _insert_seq_group_into_partial_swapped(
            self, remaining_block_sizes: int, seq_group_request_id: str,
            seq_group: SequenceGroup) -> None:
        """Insert a sequence group into the partial_swapped queue.
        """
        self.partial_swapped[seq_group_request_id] = (remaining_block_sizes,
                                                      seq_group)
        self.partial_swapped_values.append(
            (remaining_block_sizes, seq_group_request_id))

    def _get_seq_group_from_partial_swapped(
            self, seq_group_reqeust_id: str) -> Tuple[int, SequenceGroup]:
        """Get a sequence group from the partial_swapped queue.
        """
        (left_block_size,
         seq_group) = self.partial_swapped.pop(seq_group_reqeust_id)
        self.partial_swapped_values.remove(
            (left_block_size, seq_group_reqeust_id))
        return (left_block_size, seq_group)

    def _swap_out_partial(
            self, seq_group_request_id: str, seq_group: SequenceGroup,
            budget: SchedulingBudget,
            num_running_tokens: int) -> Tuple[bool, Dict[SequenceGroup, int]]:
        """Swap out a sequence group partially.

        Args:
            seq_group: The sequence group to swap out.
            budget: The scheduling budget.
            block_size: The size of the block to swap out.
        Returns:
            A tuple of (is_swap_out, 
                        swapped_out_seq_groups, 
                        swap_out_block_nums).
            is_swap_out: True if the sequence group is swapped out, 
                        False otherwise.
        """
        swapped_out_seq_groups: Dict[SequenceGroup, int] = {}
        seq_group_token_num = num_running_tokens
        if len(self.waiting) > 0:
            # swap out all blocks for the prefill request
            budget.subtract_num_batched_tokens(seq_group_request_id,
                                               seq_group_token_num)
            swapped_out_seq_groups[
                seq_group] = -1  # swap out the whole seq_group
            return True, swapped_out_seq_groups
        swap_out_rate = self.scheduler_config.swap_out_partial_rate
        # Swap out a partial block.
        seq_group_block_size = seq_group.total_token_block_size
        block_unit = max(ceil(seq_group_block_size * swap_out_rate), 1)
        if len(self.partial_swapped) == 0:
            # swap out part of the current seq_group for decode request
            swap_out_block_num = block_unit
            budget.subtract_num_batched_tokens(seq_group_request_id,
                                               seq_group_token_num)
            r_bs = seq_group_block_size - swap_out_block_num
            if r_bs > 0:
                self._insert_seq_group_into_partial_swapped(
                    r_bs, seq_group_request_id, seq_group)
            swapped_out_seq_groups[seq_group] = swap_out_block_num
            return True, swapped_out_seq_groups
        else:
            # swap out left part of the sequence group in the
            # partial_swapped queue prior to the current seq_group
            partial_swapped_values = self.partial_swapped_values
            partial_swapped_values.sort(key=lambda x: x[0])
            partial_swapped_bn, partial_swapped_sgs = map(
                list, zip(*partial_swapped_values))

            # potential bug
            selected_swapped_sg_index = self.min_numbers_sum_at_least(
                partial_swapped_bn, seq_group_block_size)
            if selected_swapped_sg_index == -1:
                # swap out part of the current seq_group for decode request
                # due to the lack of free blocks from partial_swapped queue
                block_unit = max(int(seq_group_block_size * swap_out_rate), 1)
                swap_out_block_num = block_unit
                budget.subtract_num_batched_tokens(seq_group_request_id,
                                                   seq_group_token_num)
                r_bs = seq_group_block_size - swap_out_block_num
                self._insert_seq_group_into_partial_swapped(
                    r_bs, seq_group_request_id, seq_group)
                # return seq_group as the swap out seq group
                swapped_out_seq_groups[seq_group] = swap_out_block_num
                return True, swapped_out_seq_groups
            # potential bug

            else:
                total_swap_block = 0
                last_swap_block = 0
                selected_partial_swapped_sg = partial_swapped_sgs[:
                                                                  selected_swapped_sg_index]
                for selected_seq_group in selected_partial_swapped_sg:
                    r_bs,  partial_swapped_sg = \
                        self._get_seq_group_from_partial_swapped(selected_seq_group)
                    last_swap_block = total_swap_block
                    total_swap_block += r_bs
                    if total_swap_block > seq_group_block_size:
                        block_unit = max(
                            ceil(partial_swapped_sg.total_token_block_size *
                                 swap_out_rate), 1)
                        swap_out_block_size = ceil(
                            (seq_group_block_size - last_swap_block) /
                            block_unit) * block_unit
                        left_block_size = r_bs - swap_out_block_size
                        if left_block_size > 0:
                            self._insert_seq_group_into_partial_swapped(
                                left_block_size, selected_seq_group,
                                partial_swapped_sg)
                        if swap_out_block_size > 0:
                            swapped_out_seq_groups[
                                partial_swapped_sg] = swap_out_block_size
                    else:
                        swapped_out_seq_groups[partial_swapped_sg] = r_bs
                return False, swapped_out_seq_groups

    def _append_seq_group(self,
                          seq_group: SequenceGroup,
                          blocks_to_copy: List[Tuple[int, int]],
                          num_running_tokens: int,
                          prefill_seq_groups: List[ScheduledSequenceGroup],
                          decode_seq_groups: List[ScheduledSequenceGroup],
                          budget: SchedulingBudget,
                          curr_loras: Optional[Set[int]],
                          enable_chunking: bool = False) -> None:
        total_block_size = seq_group.total_token_block_size
        self._append_slots(seq_group, blocks_to_copy)
        is_prefill = seq_group.is_prefill()
        if is_prefill:
            prefill_seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_running_tokens))
            self.scheduler_metric.prefill_token_num += num_running_tokens
        else:
            decode_seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=1))

        if self.seq_group_for_preempted == (
                None, 0) or total_block_size > self.seq_group_for_preempted[1]:
            self.seq_group_for_preempted = (seq_group, total_block_size)

        budget.add_num_batched_tokens(seq_group.request_id, num_running_tokens)
        seq_group.reset_waiting_iter_nums()
        # seq_group.update_execution_iter_nums()
        # OPTIMIZATION:  Note that get_max_num_running_seqs is
        # expensive. For the default scheduling chase where
        # enable_chunking is False, num_seqs are updated before running
        # this method, so we don't have to update it again here.
        if enable_chunking:
            num_running_seqs = seq_group.get_max_num_running_seqs()
            budget.add_num_seqs(seq_group.request_id, num_running_seqs)
        if curr_loras is not None and seq_group.lora_int_id > 0:
            curr_loras.add(seq_group.lora_int_id)
    
    def _get_ordered_requests(self):

        need_aux_scores = []
        for r in self.waiting:
            if r.need_aux_model_score():
                need_aux_scores.append(r)
        if need_aux_scores: 
            if int(os.environ.get('OPT_TIME', 0)):
                t0 = time.time()
            self.aux_model.obtain_aux_scores(need_aux_scores, self.virtual_engine)
            if int(os.environ.get('OPT_TIME', 0)):
                t1 = time.time()       
                print("OPT-TIME: ", t1 - t0)
        if len(self.running) > 0:
            print(f"[Scheduler] self.running {self.running[0].aux_model_score}")
        if self.starv != -1:
            for r in list(self.waiting) + list(self.running) + list(self.swapped):
                if r.idle >= self.starv:
                    r.pri = -1
                    r.idle = 0
                    r.runs = self.period
                    #print('[promote] ', r.request_id)
                elif r.pri == -1 and r.runs <= 0:
                    r.pri = 0
                    #print('[demote] ', r.request_id)

            ret = list(sorted(list(self.waiting) + list(self.running) + list(self.swapped), key=lambda req: (req.pri, -req.aux_model_score )))
        else:
            ret = list(sorted(list(self.waiting) + list(self.running) + list(self.swapped), key=lambda req:  -req.aux_model_score ))

        return ret 

    def _general_schedule(self):
        st = time.time()
        ordered_requests = self._get_ordered_requests()
        original_len = len(self.swapped) + len(self.running) + len(self.waiting)

        #print("budget: ", self.scheduler_config.max_num_batched_tokens, self.scheduler_config.max_num_seqs, len(self.running), len(self.waiting), len(self.swapped))
            
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )

        final_budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        curr_loras: Set[int] = set()

        remaining_waiting, prefills = (self.waiting,
                                       SchedulerPrefillOutputs.create_empty())
        remaining_running, running_scheduled = (
            self.running, SchedulerRunningOutputs.create_empty())
        remaining_swapped, swapped_in = (
            self.swapped, SchedulerSwappedInOutputs.create_empty())

        enable_chunking = True 
        selected_seq_groups = []
        exe_waiting = []
        exe_swapped_prefill_seq_groups = []
        exe_swapped_decode_seq_groups = []
        exe_running_prefill_seq_groups = []
        exe_running_decode_seq_groups = []
        gpu_block_required = 0
        

        for seq_group in ordered_requests:            
            seq = seq_group.get_seqs()[0]
            if seq_group in remaining_running:
                num_new_tokens = self._get_num_new_tokens(
                seq_group, SequenceStatus.RUNNING, enable_chunking, budget)
                if num_new_tokens == 0:
                    #print(seq_group.get_seqs())
                    assert budget.remaining_token_budget() == 0
                    break

                assert seq_group not in remaining_swapped, f" runs {seq_group}"
                assert seq_group not in remaining_waiting, f" wait {seq_group}"
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_new_tokens == 0
                        or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                                num_new_seqs=num_new_seqs)):
                    break
                budget.add_num_batched_tokens(seq_group.request_id,
                                              num_new_tokens)
                budget.add_num_seqs(seq_group.request_id, num_new_seqs)

                seq_group.num_new_tokens = num_new_tokens
                seq_group.num_new_seqs = num_new_seqs

                selected_seq_groups.append(seq_group)
                gpu_block_required += num_new_seqs
                if seq_group.is_prefill():
                    self.scheduler_metric.prefill_token_num += num_new_tokens
                else:
                    self.scheduler_metric.decode_token_num += seq_group.seq_len

                #x1.append(seq_group)

            elif seq_group in remaining_swapped:
                num_new_seqs = seq_group.get_max_num_running_seqs()
                num_new_tokens = self._get_num_new_tokens(seq_group,
                                                        SequenceStatus.SWAPPED,
                                                        enable_chunking, budget)
                num_swapped_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
                if (num_new_tokens == 0
                        or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                                num_new_seqs=num_new_seqs)):
                    break


                seq_group.num_new_tokens = num_new_tokens
                seq_group.num_new_seqs = num_new_seqs

                budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
                budget.add_num_seqs(seq_group.request_id, num_new_seqs)
                selected_seq_groups.append(seq_group)
                gpu_block_required += (len(self.block_manager._get_physical_blocks(seq_group)) + num_swapped_seqs)
                if seq_group.is_prefill():
                    self.scheduler_metric.prefill_token_num += num_new_tokens
                else:
                    self.scheduler_metric.decode_token_num += seq_group.seq_len
                    self.scheduler_metric.total_swap_in_blocks += seq_group.total_token_block_size
                    self.scheduler_metric.total_swap_in_seqs += 1

                
            elif seq_group in remaining_waiting:
                waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
                num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.WAITING,
                                                      enable_chunking, budget)
                if num_new_tokens > self.prompt_limit:
                    assert False, "req exceed prompt limit"
                #can allocate later
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_new_tokens == 0
                        or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                                num_new_seqs=num_new_seqs)):
                    #print('not appending')
                    break

                seq_group.num_new_tokens = num_new_tokens
                seq_group.num_new_seqs = num_new_seqs
                #print("append seq group: ", seq_group)
                selected_seq_groups.append(seq_group)
                budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
                budget.add_num_seqs(seq_group.request_id, num_new_seqs)
                gpu_block_required += seq_group.total_token_block_size
                if seq_group.is_prefill():
                    self.scheduler_metric.prefill_token_num += num_new_tokens
                else:
                    self.scheduler_metric.decode_token_num += seq_group.seq_len

            else:
                
                assert False, "seqgroup not in all lists"

        for seq_group in selected_seq_groups:
            ordered_requests.remove(seq_group)
        
        #print("before remain: ", len(selected_seq_groups), len(ordered_requests), budget.num_curr_seqs)

        _, execute_pinned_requests, preempted, swapped_out, blocks_to_swap_out, blocks_to_swap_in = self.reserve_free_blocks(gpu_block_required, selected_seq_groups, ordered_requests, remaining_running, budget)
        blocks_to_copy = []

        for seq_group in execute_pinned_requests:
            if seq_group in remaining_waiting:
                remaining_waiting.remove(seq_group)
                if self.block_manager.can_allocate(seq_group) == AllocStatus.OK:
                    self._allocate_and_set_running(seq_group)
                    exe_waiting.append(ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=seq_group.num_new_tokens))
                    
                    #final_budget.add_num_batched_tokens(seq_group.request_id, seq_group.num_new_tokens)
                    #final_budget.add_num_seqs(seq_group.request_id, seq_group.num_new_seqs)
                    del seq_group.num_new_tokens
                    del seq_group.num_new_seqs
                else:
                    assert False, "can not append new req"
            elif seq_group in remaining_running:
                remaining_running.remove(seq_group)
                if self.block_manager.can_append_slots(seq_group):
                    self._append_slots(seq_group, blocks_to_copy)

                    is_prefill = seq_group.is_prefill()
                    #print("prefill run: ", is_prefill)
                    if is_prefill:
                        exe_running_prefill_seq_groups.append(
                            ScheduledSequenceGroup(
                                seq_group=seq_group,
                                token_chunk_size=seq_group.num_new_tokens))
                    else:
                        exe_running_decode_seq_groups.append(
                            ScheduledSequenceGroup(seq_group=seq_group,
                                                token_chunk_size=1))

                    #final_budget.add_num_batched_tokens(seq_group.request_id, seq_group.num_new_tokens)
                    #final_budget.add_num_seqs(seq_group.request_id, seq_group.num_new_seqs)
                    del seq_group.num_new_tokens
                    del seq_group.num_new_seqs

                else:
                    raise AssertionError()

            elif seq_group in remaining_swapped:
                remaining_swapped.remove(seq_group)
                if self.block_manager.can_append_slots(seq_group):
                    self._append_slots(seq_group, blocks_to_copy)

                    is_prefill = seq_group.is_prefill()
                    #print("swapped: ", seq_group, is_prefill)
                    if is_prefill:
                        exe_swapped_prefill_seq_groups.append(
                            ScheduledSequenceGroup(seq_group,
                                                token_chunk_size=seq_group.num_new_tokens))
                    else:
                        assert seq_group.num_new_tokens == 1
                        exe_swapped_decode_seq_groups.append(
                            ScheduledSequenceGroup(seq_group, token_chunk_size=1))

                    #final_budget.add_num_batched_tokens(seq_group.request_id, seq_group.num_new_tokens)
                    #final_budget.add_num_seqs(seq_group.request_id, seq_group.num_new_seqs)
                    del seq_group.num_new_tokens
                    del seq_group.num_new_seqs

                else:
                    raise AssertionError()
            else:
                raise AssertionError() 
        #print("prefill decode: ", len(exe_running_decode_seq_groups), len(exe_running_prefill_seq_groups), len(remaining_running))
        #assert len(remaining_running) == 0
        prefills = SchedulerPrefillOutputs(
            seq_groups=exe_waiting,
            ignored_seq_groups=[],
            kv_free_seq_groups=[],
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill=True))
        swapped_in = SchedulerSwappedInOutputs(
            decode_seq_groups=exe_swapped_decode_seq_groups,
            prefill_seq_groups=exe_swapped_prefill_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False), 
            infeasible_seq_groups=[])
        running_scheduled = SchedulerRunningOutputs(
            decode_seq_groups=exe_running_decode_seq_groups,
            prefill_seq_groups=exe_running_prefill_seq_groups,
            preempted=preempted,
            swapped_out=swapped_out,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False))

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs, f" num req: {budget.num_curr_seqs} {self.scheduler_config.max_num_seqs}"


        #print("extend: ", len(remaining_running), len(prefills.seq_groups), len(running_scheduled.decode_seq_groups), len(running_scheduled.prefill_seq_groups), len(running_scheduled.prefill_seq_groups), len(swapped_in.decode_seq_groups), len(swapped_in.prefill_seq_groups))
        # Update waiting requests.
        self.waiting = remaining_waiting
        self.waiting.extendleft(running_scheduled.preempted)
        # Update new running requests.
        self.running = remaining_running
        self.running.extend([s.seq_group for s in prefills.seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.prefill_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.prefill_seq_groups])
        # Update swapped requests.
        self.swapped = remaining_swapped
        self.swapped.extend(running_scheduled.swapped_out)
        all_pri = list(self.swapped) + list(self.running) + list(self.waiting)
        assert len(self.swapped) + len(self.running) + len(self.waiting) == original_len
        ret = SchedulerOutputs(
            scheduled_seq_groups=(prefills.seq_groups +
                                  running_scheduled.prefill_seq_groups +
                                  swapped_in.prefill_seq_groups +
                                  running_scheduled.decode_seq_groups +
                                  swapped_in.decode_seq_groups),
            num_prefill_groups=(len(prefills.seq_groups) +
                                len(swapped_in.prefill_seq_groups) +
                                len(running_scheduled.prefill_seq_groups)),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=running_scheduled.blocks_to_copy + swapped_in.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=0,
            preempted=0,
            num_waiting_to_running=0,
            num_running_to_waiting=0,
            recomputed_token_nums=0,
            need_score=self.need_score,
            allow_both_swap=True
        )

        block_generated = len(ret.scheduled_seq_groups)
        block_used = self.scheduler_metric.decode_token_num + block_generated
        # Normalization
        self.scheduler_metric.gpu_memory_occupy = block_used/ self.cache_config.num_gpu_blocks

        # 2) GPU computation:

        prefills_seq_groups = (prefills.seq_groups +
                                prefills.kv_free_seq_groups +
                                running_scheduled.prefill_seq_groups +
                                swapped_in.prefill_seq_groups )
        computation_tokens = 0

        for seq_group in [s.seq_group for s in prefills_seq_groups]:
            computation_tokens += seq_group.seq_len

        computation_tokens += block_generated

        self.scheduler_metric.gpu_computation_occupy = computation_tokens

        self.scheduler_metric.running_seq_nums += len(ret.scheduled_seq_groups)
        self.scheduler_metric.waiting_seq_nums += len(self.swapped)
        self.scheduler_metric.pending_seq_nums += len(self.waiting)
        
        running_this_step = [r.seq_group for r in ret.scheduled_seq_groups]
        for seq in all_pri:
            if seq in running_this_step:
                if seq.pri == -1:
                    seq.runs -= 1
                seq.idle = 0
            else:
                seq.idle += 1
        
        return ret


    def reserve_free_blocks(self, num_blocks_needed, pinned_requests: List[SequenceGroup], priority_requests, remaining_running, final_budget):

        blocks_to_swap_out: List[Tuple[int, int]]  = []
        blocks_to_swap_in: List[Tuple[int, int]] = []
        
        preempted = []
        swapped_out = []

        num_swap_out_blocks_needed = (
            num_blocks_needed
            - self.block_manager.gpu_allocator.get_num_free_blocks() \
            + self.block_manager.watermark_blocks
        )
        swap_out_needed = num_swap_out_blocks_needed > 0

        # the pinned requests we really execute
        execute_pinned_requests = pinned_requests.copy()
        # the pinned requests we put back due to swapped out
        swapped_pinned_requests: List[SequenceGroup] = []

        # swap out low priority requests if GPU blocks are not enough
        if swap_out_needed:
            pinned_request_ids = set(
                [request.request_id for request in pinned_requests]
            )
            # swap out from the lowest priority request
            for request in reversed(priority_requests): 
                # pinned request must have already been popped from MLFQ,
                assert request.request_id not in pinned_request_ids
                if num_swap_out_blocks_needed <= 0:
                    break
                if (len(request.get_seqs(status=SequenceStatus.RUNNING))):
                    num_swap_out_blocks_needed -= len(self.block_manager._get_physical_blocks(request))
                    preempted_mode = self._preempt(request, blocks_to_swap_out, preemption_mode = PreemptionMode.SWAP)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(request)
                        self.scheduler_metric.total_swap_out_blocks += request.total_token_block_size
                        self.scheduler_metric.total_swap_out_seqs+= 1
                        request.update_swap_times()
                    else:
                        self.scheduler_metric.total_swap_out_blocks += request.total_token_block_size
                        self.scheduler_metric.total_swap_out_seqs += 1
                        request.update_swap_times()
                        swapped_out.append(request)
                    if request in remaining_running:
                        assert request not in execute_pinned_requests
                        remaining_running.remove(request)
                    else:
                        execute_pinned_requests.remove(request)

            if num_swap_out_blocks_needed > 0:
                # if we still need to swap out blocks, swap out pinned requests
                # location of pinned requests may be in CPU/GPU or none now
                while num_swap_out_blocks_needed > 0 and len(execute_pinned_requests) > 0:
                    request = execute_pinned_requests.pop(-1)
                    swapped_pinned_requests.append(request)
                    if (len(request.get_seqs(status=SequenceStatus.RUNNING))):
                        num_swap_out_blocks_needed -= request.num_seqs(status=SequenceStatus.RUNNING)
                        num_swap_out_blocks_needed -= len(self.block_manager._get_physical_blocks(request))
                        preempted_mode = self._preempt(request, blocks_to_swap_out, preemption_mode = PreemptionMode.SWAP)
                        
                        remaining_running.remove(request)
                        if preempted_mode == PreemptionMode.RECOMPUTE:
                            preempted.append(request)
                            self.scheduler_metric.total_swap_out_blocks += request.total_token_block_size
                            self.scheduler_metric.total_swap_out_seqs += 1
                            request.update_swap_times()
                            request.update_waiting_iter_nums()
                        else:
                            self.scheduler_metric.total_swap_out_blocks += request.total_token_block_size
                            self.scheduler_metric.total_swap_out_seqs += 1
                            request.update_swap_times()
                            request.update_waiting_iter_nums()
                            swapped_out.append(request)

                    elif (len(request.get_seqs(status=SequenceStatus.SWAPPED))):
                        num_swap_out_blocks_needed -= (len(self.block_manager._get_physical_blocks(request)) + request.num_seqs(status=SequenceStatus.SWAPPED))
                    else:
                        num_swap_out_blocks_needed -= request.get_seqs()[0].n_blocks  
                        

            # swap block is required by waiting request and we already put it back
            assert num_swap_out_blocks_needed <= 0

        # swap in pinned requests if needed
        for seq_group in execute_pinned_requests:
            if (len(seq_group.get_seqs(status=SequenceStatus.SWAPPED))):
                self._swap_in(seq_group, blocks_to_swap_in)

            final_budget.add_num_batched_tokens(seq_group.request_id, seq_group.num_new_tokens)
            final_budget.add_num_seqs(seq_group.request_id, seq_group.num_new_seqs)

        if not swap_out_needed:
            for request in priority_requests:                 

                if (len(request.get_seqs(status=SequenceStatus.SWAPPED))):

                    num_new_seqs = request.get_max_num_running_seqs()
                    num_new_tokens = self._get_num_new_tokens(request,
                                                        SequenceStatus.SWAPPED,
                                                        enable_chunking=True, budget=final_budget)


                    # swap in the request if there are enough free blocks
                    if (
                        self.block_manager.can_swap_in(request)
                    ) and (num_swap_out_blocks_needed + len(self.block_manager._get_physical_blocks(request)) + request.num_seqs(status=SequenceStatus.SWAPPED)) < 0 \
                        and (num_new_tokens > 0 and final_budget.can_schedule(num_new_tokens=num_new_tokens, num_new_seqs=num_new_seqs)):
                        
                        #print("Xswap in : ", request, request.is_prefill(), request.get_max_num_running_seqs(), sum(seq.get_num_new_tokens() for seq in request.get_seqs(status=SequenceStatus.SWAPPED)))
                        request.num_new_seqs = request.get_max_num_running_seqs()
                        request.num_new_tokens = sum(seq.get_num_new_tokens() for seq in request.get_seqs(status=SequenceStatus.SWAPPED))
                        self._swap_in(request, blocks_to_swap_in)
                        self.scheduler_metric.total_swap_in_blocks += request.total_token_block_size
                        self.scheduler_metric.total_swap_in_seqs += request.num_seqs(status=SequenceStatus.RUNNING)

                        final_budget.add_num_batched_tokens(seq_group.request_id, seq_group.num_new_tokens)
                        final_budget.add_num_seqs(seq_group.request_id, seq_group.num_new_seqs)

                        execute_pinned_requests.append(request)
                        num_swap_out_blocks_needed += (len(self.block_manager._get_physical_blocks(request)) + request.num_seqs(status=SequenceStatus.SWAPPED))
                    else:
                        break
                # reduce the quata no matter if the request needs swapping in
                #swap_quata -= 1


        return swapped_pinned_requests, execute_pinned_requests, preempted, swapped_out, blocks_to_swap_out, blocks_to_swap_in


    def _schedule_preemption(
        self,
        running_queue: Deque[SequenceGroup],
        swapped_queue: Deque[SequenceGroup],
        waiting_queue: Deque[SequenceGroup],
        budget: SchedulingBudget,
        policy: Policy,
        policy_info: PolicyInfo,
        enable_chunking: bool = False,
    ) -> Tuple[deque, deque, deque, SchedulerRunningOutputs,
               SchedulerSwappedInOutputs, SchedulerPrefillOutputs, int]:
        """Schedule sequence groups that are in inference stage.

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            running_queue: The queue that contains running requests.
                The given arguments are NOT in-place modified.
            swapped_queue: The queue that contains swapped out requests.
                The given arguments are NOT in-place modified.
            waiting_queue: The queue that contains waiting requests.
                The given arguments are NOT in-place modified.
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.
            policy: The scheduling policy.
        Returns:
            A tuple of (
            running_queue, swapped_queue, waiting_queue,
            SchedulerRunningOutputs, SchedulerSwappedInOutputs, SchedulerPrefillOutputs, recomputed_token_nums).
        
        """
        # create a copy of queues to avoid in-place modification
        self.enable_chunking = enable_chunking

        decode_seq_groups_running: List[SequenceGroup] = []
        decode_seq_groups_swapped: List[SequenceGroup] = []
        prefill_seq_groups_running: List[SequenceGroup] = []
        prefill_seq_groups_swapped: List[SequenceGroup] = []
        preempted_running: List[SequenceGroup] = []
        swapped_out_running: List[SequenceGroup] = []
        blocks_to_swap_in: List[Tuple[int, int]] = []
        blocks_to_swap_out: List[Tuple[int, int]] = []
        blocks_to_copy_running: List[Tuple[int, int]] = []
        blocks_to_copy_swapped: List[Tuple[int, int]] = []
        infeasible_seq_groups: List[SequenceGroup] = []
        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups_prefill: List[SequenceGroup] = []
        num_lookahead_slots_running: int = 0
        num_lookahead_slots_swapped: int = 0
        num_lookahead_slots_prefill: int = 0
        recomputed_token_nums: int = 0

        scheduler_preemtion = SchedulerPreemption(
            decode_seq_groups_running=decode_seq_groups_running,
            decode_seq_groups_swapped=decode_seq_groups_swapped,
            prefill_seq_groups_running=prefill_seq_groups_running,
            prefill_seq_groups_swapped=prefill_seq_groups_swapped,
            preempted_running=preempted_running,
            swapped_out_running=swapped_out_running,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy_running=blocks_to_copy_running,
            blocks_to_copy_swapped=blocks_to_copy_swapped,
            infeasible_seq_groups=infeasible_seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            seq_groups_prefill=seq_groups_prefill,
            num_lookahead_slots_running=num_lookahead_slots_running,
            num_lookahead_slots_swapped=num_lookahead_slots_swapped,
            num_lookahead_slots_prefill=num_lookahead_slots_prefill,
        )
        gpu_block_capacity = self.block_manager.gpu_block_capacity
  
        
        tmp_total_block_size = 0
        selected_running_seq_groups: List[SequenceGroup] = []
        selected_swapped_seq_groups: List[SequenceGroup] = []
        now = time.time()
        num_new_tokens = 0
        all_seq_group_queue = running_queue + swapped_queue + waiting_queue
        if self.scheduler_config.policy != 'tfittradeoff':
            total_queue = policy.sort_by_priority(now, all_seq_group_queue)
        else:
            if self.scheduler_config.phase!='decode':
                total_waiting_queue = swapped_queue + waiting_queue
                total_waiting_queue = policy.sorted_by_priority(total_waiting_queue, "waiting",policy_info)
                running_queue = policy.sorted_by_priority(running_queue, "running",policy_info)
                total_queue = running_queue + total_waiting_queue
            else:
                swapped_queue  = policy.sorted_by_priority(swapped_queue,"waiting",policy_info)
                running_queue = policy.sorted_by_priority(running_queue, "running",policy_info)
                total_queue = waiting_queue + running_queue + swapped_queue
    
        swapped_queue_set = set(swapped_queue)
        running_queue_set = set(running_queue)
        waiting_queue_set = set(waiting_queue)
        if budget.max_num_seqs != 2048:
            budget.update_max_num_seqs(4096)
        for sg in total_queue:
            if sg in swapped_queue_set:
                seq_status= SequenceStatus.SWAPPED
            elif sg in running_queue_set:
                seq_status= SequenceStatus.RUNNING
            elif sg in waiting_queue_set:
                seq_status= SequenceStatus.WAITING
            if sg not in running_queue_set and not self.batch_solver.is_opt(self.scheduler_config.policy, sg) and self.scheduler_config.phase!='decode':
                sg.update_waiting_iter_nums()
                continue
            num_new_tokens = self._get_num_new_tokens(sg,
                                                        seq_status,
                                                        self.enable_chunking,
                                                        budget)
            total_token_block_size = sg.total_token_block_size
            if seq_status == SequenceStatus.WAITING:
                block_size = min(total_token_block_size, gpu_block_capacity - tmp_total_block_size)
                tmp_total_block_size += block_size
            else:
                block_size =total_token_block_size  
                tmp_total_block_size += block_size
            num_new_seqs = sg.get_max_num_running_seqs()
            if tmp_total_block_size <= gpu_block_capacity and num_new_tokens > 0 \
                and budget.remaining_token_budget() >= 0 \
                and budget.can_schedule(
                    num_new_tokens=num_new_tokens,
                    num_new_seqs=num_new_seqs
                ):
                sg.reset_waiting_iter_nums()
                selected_running_seq_groups.append(sg)
                sg.token_chunk_size = num_new_tokens
                if sg.is_prefill() and num_new_tokens < sg.seq_len:
                    sg.is_chunk_prefill=True

                budget.add_num_batched_tokens(sg.request_id, num_new_tokens)
                budget.add_num_seqs(sg.request_id, num_new_seqs)
            else:
                if sg in waiting_queue_set or sg in swapped_queue_set:
                    sg.update_waiting_iter_nums()
                tmp_total_block_size -= block_size
                selected_swapped_seq_groups.append(sg)
        self.batch_solver.reset_opt()
        for seq_group in selected_swapped_seq_groups:
            self._preempt_seq(
                seq_group=seq_group,
                budget=budget,
                schedule_preemption=scheduler_preemtion,
                running_queue=running_queue,
            )
        for seq_group in selected_running_seq_groups:
            _, _, recomputed_token_nums = self._allocate_seq(
                seq_group=seq_group,
                budget=budget,
                schedule_preemption=scheduler_preemtion,
                running_queue=running_queue,
                waiting_queue=waiting_queue,
                swapped_queue=swapped_queue,
                recomputed_token_nums=recomputed_token_nums,
            )
        running_scheduler_output = SchedulerRunningOutputs(
            decode_seq_groups=scheduler_preemtion.decode_seq_groups_running,
            prefill_seq_groups=scheduler_preemtion.prefill_seq_groups_running,
            preempted=scheduler_preemtion.preempted_running,
            swapped_out=scheduler_preemtion.swapped_out_running,
            blocks_to_swap_out=scheduler_preemtion.blocks_to_swap_out,
            blocks_to_copy=scheduler_preemtion.blocks_to_copy_running,
            num_lookahead_slots=scheduler_preemtion.num_lookahead_slots_running
        )
        swapped_scheduler_output = SchedulerSwappedInOutputs(
            decode_seq_groups=scheduler_preemtion.decode_seq_groups_swapped,
            prefill_seq_groups=scheduler_preemtion.prefill_seq_groups_swapped,
            blocks_to_swap_in=scheduler_preemtion.blocks_to_swap_in,
            blocks_to_copy=scheduler_preemtion.blocks_to_copy_swapped,
            num_lookahead_slots=scheduler_preemtion.
            num_lookahead_slots_swapped,
            infeasible_seq_groups=scheduler_preemtion.infeasible_seq_groups)
        waiting_scheduler_output = SchedulerPrefillOutputs(
            seq_groups=scheduler_preemtion.seq_groups_prefill,
            kv_free_seq_groups=[],
            num_lookahead_slots=scheduler_preemtion.
            num_lookahead_slots_prefill,
            ignored_seq_groups=scheduler_preemtion.ignored_seq_groups)

        return (running_queue, swapped_queue, waiting_queue,
                running_scheduler_output, swapped_scheduler_output,
                waiting_scheduler_output, recomputed_token_nums)

    def _allocate_seq(self, seq_group: SequenceGroup, budget: SchedulingBudget,
                      schedule_preemption: SchedulerPreemption,
                      running_queue: deque, waiting_queue: deque,
                      swapped_queue: deque, recomputed_token_nums: int):
        seq_group.reset_waiting_iter_nums()
        if seq_group in running_queue:
            num_running_tokens = seq_group.token_chunk_size 
            self._append_slots(seq_group,
                               schedule_preemption.blocks_to_copy_running)
            is_prefill = seq_group.is_prefill()
            if is_prefill:
                schedule_preemption.prefill_seq_groups_running.append(
                    ScheduledSequenceGroup(
                        seq_group=seq_group,
                        token_chunk_size=num_running_tokens))
                recomputed_token_nums += num_running_tokens
                self.scheduler_metric.prefill_token_num += num_running_tokens
            else:
                schedule_preemption.decode_seq_groups_running.append(
                    ScheduledSequenceGroup(seq_group=seq_group,
                                           token_chunk_size=1))
            running_queue.remove(seq_group)
        elif seq_group in swapped_queue:
            is_prefill = seq_group.is_prefill()
            alloc_status = self.block_manager.can_swap_in(
                seq_group, self._get_num_lookahead_slots(is_prefill))
            if alloc_status == AllocStatus.NEVER:
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                schedule_preemption.infeasible_seq_groups.append(seq_group)
                swapped_queue.remove(seq_group)
            num_new_tokens = seq_group.token_chunk_size

            swapped_queue.remove(seq_group)
            self._swap_in(seq_group, schedule_preemption.blocks_to_swap_in)
            self.scheduler_metric.total_swap_in_blocks += seq_group.total_token_block_size
            self.scheduler_metric.total_swap_in_seqs += 1
            self._append_slots(seq_group,
                               schedule_preemption.blocks_to_copy_swapped)
            if is_prefill:
                schedule_preemption.prefill_seq_groups_swapped.append(
                    ScheduledSequenceGroup(seq_group,
                                           token_chunk_size=num_new_tokens))
                self.scheduler_metric.prefill_token_num += num_new_tokens
            else:
                schedule_preemption.decode_seq_groups_swapped.append(
                    ScheduledSequenceGroup(seq_group, token_chunk_size=1))
            # budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            # budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        elif seq_group in waiting_queue:
            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            prompt_limit = self._get_prompt_limit(seq_group)
            num_new_tokens = seq_group.token_chunk_size
            if num_new_tokens > prompt_limit:
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                schedule_preemption.ignored_seq_groups.append(seq_group)
                waiting_queue.remove(seq_group)

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(seq_group)
            if can_allocate == AllocStatus.NEVER:
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                schedule_preemption.ignored_seq_groups.append(seq_group)
                waiting_queue.remove(seq_group)
            elif can_allocate == AllocStatus.LATER:
                return False, "Cannot allocate sequence group in the waiting queue.", recomputed_token_nums
            self._allocate_and_set_running(seq_group)
            waiting_queue.remove(seq_group)
            schedule_preemption.seq_groups_prefill.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            self.scheduler_metric.prefill_token_num += num_new_tokens
        return True, None, recomputed_token_nums

    def _preempt_seq(self, seq_group: SequenceGroup, budget: SchedulingBudget,
                     schedule_preemption: SchedulerPreemption,
                     running_queue: deque):
        preemption_mode = self.preemption_mode
        if seq_group in running_queue:
            if not self.block_manager.can_swap_out(seq_group):
                preemption_mode = PreemptionMode.RECOMPUTE
            preempted_mode = self._preempt(
                seq_group, schedule_preemption.blocks_to_swap_out,
                preemption_mode)
            if preempted_mode == PreemptionMode.RECOMPUTE:
                schedule_preemption.preempted_running.append(seq_group)
            else:
                schedule_preemption.swapped_out_running.append(seq_group)
            running_queue.remove(seq_group)
            self.scheduler_metric.total_swap_out_seqs += 1
            self.scheduler_metric.total_swap_out_blocks += seq_group.total_token_block_size
            seq_group.update_swap_times()
            seq_group.update_waiting_iter_nums()
            # Queue requests that couldn't be scheduled.
        return True, None

    def handle_victim(self, required_block_size,vic_block_size, vic_request_id=None, vic_group=None):
        swap_out_unit = ceil(vic_block_size * self.partial_swapped_rate)
        if vic_block_size <= required_block_size:
            # 全量交换
            swap_out_nums = -1 if vic_request_id is None else vic_block_size
            required_block_size -= vic_block_size
            return SequenceStatus.SWAPPED, swap_out_nums
        else:
            # 部分交换
            swap_out_nums = max(ceil(required_block_size / swap_out_unit) * swap_out_unit, 1)
            left_size = vic_block_size - swap_out_nums
            if left_size > 0 and vic_request_id is not None:
                self.partial_swapped[vic_request_id] = (left_size, vic_group)
            required_block_size = max(0, required_block_size - swap_out_nums)
            return SequenceStatus.PARTIAL_SWAPPED, swap_out_nums


    def _schedule_running_partial(
        self,
        running_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        policy: Policy,
        policy_info: PolicyInfo,
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerRunningOutputs, int]:
        '''
        Schedule sequence groups that are running and do not use LoRa.
        '''
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_out: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []

        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        recomputed_token_nums: int = 0
        preempted: Set[SequenceGroup] = set()
        swapped_out: Set[SequenceGroup] = set()
        partial_swapped_flag = self.partial_swap_out_flag
        partial_swapped_rate = self.scheduler_config.swap_out_partial_rate

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        scheduler_policy= self.scheduler_config.policy
        now = time.time()
        if scheduler_policy == "tfittradeoff":
            running_queue = policy.sorted_by_priority(
                running_queue, queue_type='running', policy_info=policy_info)
        else:
            running_queue = policy.sort_by_priority(now, running_queue)
        self.has_preempted_seq = False

        while running_queue:
            seq_group: SequenceGroup = running_queue[0]
                 
            num_running_tokens = self._get_num_new_tokens(
                seq_group, SequenceStatus.RUNNING, enable_chunking, budget)

            if num_running_tokens == 0:
                break

            running_queue.popleft()
            required_block_size = seq_group.total_token_block_size

            while not self._can_append_slots(seq_group):
                swap_out_block_nums = -1
                budget.subtract_num_batched_tokens(seq_group.request_id,
                                                   num_running_tokens)
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(seq_group.request_id,
                                         num_running_seqs)

                if running_queue:
                    # Preempt the lowest-priority sequence groups.
                    self.has_preempted_seq = True
                    if not partial_swapped_flag:
                        victim_seq_group = running_queue.pop()
                        preempted_mode = self._preempt(victim_seq_group,
                                                       blocks_to_swap_out,
                                                       self.preemption_mode)
                        self.scheduler_metric.total_swap_out_blocks += victim_seq_group.total_token_block_size
                        self.scheduler_metric.total_swap_out_seqs += 1
                        if preempted_mode == PreemptionMode.RECOMPUTE:
                            preempted.add(victim_seq_group)
                        elif preempted_mode == PreemptionMode.KV_FREE_RECOMPUTE and scheduler_policy == 'tfittradeoff':
                            self.kv_free_seq_groups.append(victim_seq_group)
                        else:
                            swapped_out.add(victim_seq_group)
                    else:
                        if len(self.partial_swapped) == 0:
                            victim_seq_group = running_queue.pop()  # Debug
                            vic_seq_group_block_size = victim_seq_group.total_token_block_size
                            seq_group_status, swap_out_block_nums = self.handle_victim(
                                required_block_size, 
                                vic_seq_group_block_size, 
                                victim_seq_group.request_id, 
                                victim_seq_group)
                        else:
                            victim_seq_group_request_id = list(
                                self.partial_swapped.keys())[0]
                            left_victim_block_size, victim_seq_group = \
                                                self.partial_swapped.pop(
                                                    victim_seq_group_request_id)
                            seq_group_status, swap_out_block_nums = self.handle_victim(
                                required_block_size, 
                                left_victim_block_size, 
                                victim_seq_group_request_id, 
                                victim_seq_group
                                )

                        preempted_mode = self._preempt(
                            victim_seq_group,
                            blocks_to_swap_out,
                            self.preemption_mode,
                            swap_out_block_nums,
                            seq_group_status=seq_group_status)

                        if preempted_mode == PreemptionMode.RECOMPUTE:
                            preempted.add(victim_seq_group)
                        elif preempted_mode == PreemptionMode.KV_FREE_RECOMPUTE:
                            self.kv_free_seq_groups.append(victim_seq_group)
                        else:
                            swapped_out.add(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self.has_preempted_seq = True
                    if not partial_swapped_flag:
                        preempted_mode = self._preempt(seq_group,
                                                       blocks_to_swap_out,
                                                       self.preemption_mode)

                        if preempted_mode == PreemptionMode.RECOMPUTE:
                            preempted.add(seq_group)
                        elif preempted_mode == PreemptionMode.KV_FREE_RECOMPUTE:
                            self.kv_free_seq_groups.append(seq_group)
                        else:
                            swapped_out.add(seq_group)
                    else:
                        victim_seq_group = seq_group
                        vic_seq_group_block_size = victim_seq_group.total_token_block_size
                        swap_out_block_unit = ceil(vic_seq_group_block_size *
                                                   partial_swapped_rate)
                        swap_out_block_nums = max(swap_out_block_unit, 1)
                        left_victim_block_size = vic_seq_group_block_size - \
                                                 swap_out_block_nums
                        victim_seq_group_request_id = victim_seq_group.request_id
                        self.partial_swapped[victim_seq_group_request_id] = (
                            left_victim_block_size, victim_seq_group)
                        seq_group_status = SequenceStatus.PARTIAL_SWAPPED
                        preempted_mode = self._preempt(
                            victim_seq_group,
                            blocks_to_swap_out,
                            self.preemption_mode,
                            swap_out_block_nums,
                            seq_group_status=seq_group_status)

                        if preempted_mode == PreemptionMode.RECOMPUTE:
                            preempted.add(victim_seq_group)
                        elif preempted_mode == PreemptionMode.KV_FREE_RECOMPUTE:
                            self.kv_free_seq_groups.append(victim_seq_group)
                        else:
                            swapped_out.add(victim_seq_group)
                    break
            else:
                # if len(running_queue) == 0 and len(swapped_out) > 0:
                    # append the seq_group into the low priority queue

                self._append_seq_group(seq_group, blocks_to_copy,
                                       num_running_tokens, prefill_seq_groups,
                                       decode_seq_groups, budget, curr_loras,
                                       enable_chunking)

        if len(swapped_out) > 0:
            total_swapped_out = set(self.swapped)
            swapped_out = swapped_out.difference(total_swapped_out)
            # self.low_priority_seq_groups.append()

        self.scheduler_metric.schedule_running_time += time.time() - now

        return running_queue, SchedulerRunningOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            preempted=list(preempted),
            swapped_out=list(swapped_out),
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False)), recomputed_token_nums
    def get_and_reset_finished_requests_ids(self) -> List[str]:
        """Flushes the list of request ids of previously finished seq_groups."""
        finished_requests_ids = self._finished_requests_ids
        self._finished_requests_ids = list()
        return finished_requests_ids

    def _schedule_running(
        self,
        running_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        policy: Policy,
        policy_info: PolicyInfo,
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerRunningOutputs, int]:
        """Schedule sequence groups that are running.

        Running queue should include decode and chunked prefill requests.

        Args:
            running_queue: The queue that contains running requests (i.e.,
                decodes). The given arguments are NOT in-place modified.
            budget: The scheduling budget. The argument is in-place updated
                when any decodes are preempted.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any decodes are preempted.
            policy: The sorting policy to sort running_queue.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.
    
        Returns:
            A tuple of remaining running queue (should be always 0) after
            scheduling and SchedulerRunningOutputs.
        """
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_out: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []

        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        recomputed_token_nums: int = 0
        preempted: List[SequenceGroup] = []
        swapped_out: List[SequenceGroup] = []

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        now = time.time()
        if self.scheduler_config.policy == "tfittradeoff":
            running_queue = policy.sorted_by_priority(running_queue, queue_type='running',policy_info=policy_info)
        else:
            running_queue = policy.sort_by_priority(now, running_queue)

        while running_queue:
            seq_group: SequenceGroup = running_queue[0]

            num_running_tokens = self._get_num_new_tokens(
                seq_group, SequenceStatus.RUNNING, enable_chunking, budget)

            if num_running_tokens == 0:
                break

            running_queue.popleft()

            while not self._can_append_slots(seq_group):
                budget.subtract_num_batched_tokens(seq_group.request_id,
                                                   num_running_tokens)
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(seq_group.request_id,
                                         num_running_seqs)

                if (curr_loras is not None and seq_group.lora_int_id > 0
                        and seq_group.lora_int_id in curr_loras):
                    curr_loras.remove(seq_group.lora_int_id)

                if running_queue:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = running_queue.pop()

                    if self.preemption_mode:
                        preempted_mode = self._preempt(victim_seq_group,
                                                       blocks_to_swap_out,
                                                       self.preemption_mode)
                    else:
                        preempted_mode = self._preempt(victim_seq_group,
                                                       blocks_to_swap_out)

                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(victim_seq_group)
                    else:
                        swapped_out.append(victim_seq_group)
                    victim_seq_group.swap_out_moment = time.time()
                    self.scheduler_metric.total_swap_out_blocks += victim_seq_group.total_token_block_size
                    self.scheduler_metric.total_swap_out_seqs += 1
                    victim_seq_group.update_swap_times()

                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.

                    if self.preemption_mode:
                        preempted_mode = self._preempt(seq_group,
                                                       blocks_to_swap_out,
                                                       self.preemption_mode)
                    else:
                        preempted_mode = self._preempt(seq_group,
                                                       blocks_to_swap_out)

                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(seq_group)
                    else:
                        swapped_out.append(seq_group)
                    self.scheduler_metric.total_swap_out_blocks += seq_group.total_token_block_size
                    self.scheduler_metric.total_swap_out_seqs += 1
                    seq_group.swap_out_moment = time.time()
                    break
            else:
                self._append_slots(seq_group, blocks_to_copy)
                seq_group.reset_waiting_iter_nums()
                is_prefill = seq_group.is_prefill()
                if is_prefill:
                    prefill_seq_groups.append(
                        ScheduledSequenceGroup(
                            seq_group=seq_group,
                            token_chunk_size=num_running_tokens))
                    recomputed_token_nums += num_running_tokens
                    self.scheduler_metric.prefill_token_num += num_running_tokens
                else:
                    decode_seq_groups.append(
                        ScheduledSequenceGroup(seq_group=seq_group,
                                               token_chunk_size=1))
                budget.add_num_batched_tokens(seq_group.request_id,
                                              num_running_tokens)
                # OPTIMIZATION:  Note that get_max_num_running_seqs is
                # expensive. For the default scheduling chase where
                # enable_chunking is False, num_seqs are updated before running
                # this method, so we don't have to update it again here.
                if enable_chunking:
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.request_id, num_running_seqs)
                if curr_loras is not None and seq_group.lora_int_id > 0:
                    curr_loras.add(seq_group.lora_int_id)

        return running_queue, SchedulerRunningOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            preempted=preempted,
            swapped_out=swapped_out,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False)), recomputed_token_nums

    def _schedule_swapped(
        self,
        swapped_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        policy: Policy,
        policy_info: PolicyInfo,
        enable_chunking: bool = False,
    ) -> Tuple[deque, SchedulerSwappedInOutputs]:
        """Schedule sequence groups that are swapped out.

        It schedules swapped requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            swapped_queue: The queue that contains swapped out requests.
                The given arguments are NOT in-place modified.
            budget: The scheduling budget. The argument is in-place updated
                when any requests are swapped in.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are swapped in.
            policy: The sorting policy to sort swapped_queue.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            A tuple of remaining swapped_queue after scheduling and
            SchedulerSwappedInOutputs.
        """
        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_in: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []
        decode_seq_groups: List[ScheduledSequenceGroup] = []
        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        now = time.time()
        if self.scheduler_config.policy == "tfittradeoff":
            # avg_priorities = -1
            swapped_queue = policy.sorted_by_priority(
                swapped_queue,
                queue_type='swapped',policy_info=policy_info)
        else:
            swapped_queue = policy.sort_by_priority(now, swapped_queue)
        infeasible_seq_groups: List[SequenceGroup] = []
        leftover_swapped: Deque[SequenceGroup] = deque()
        while swapped_queue:
            seq_group: SequenceGroup = swapped_queue[0]
            # if (self.scheduler_config.policy in ["infer", "tfittradeoff"]
            #         and seq_group.request_id in self.partial_swapped):
            #     swapped_queue.popleft()
            #     leftover_swapped.appendleft(seq_group)
            #     continue
            # If the sequence group cannot be swapped in, stop.
            is_prefill = seq_group.is_prefill()
            alloc_status = self.block_manager.can_swap_in(
                seq_group, self._get_num_lookahead_slots(is_prefill))
            if alloc_status == AllocStatus.LATER:
                if self.scheduler_config.policy == "tfittradeoff":  # Debug
                    # seq_group.update_waiting_iter_nums()
                    # swapped_queue.popleft()
                    # leftover_swapped.appendleft(seq_group)
                    # continue
                    # break
                    for seq_group in swapped_queue:
                        seq_group.update_waiting_iter_nums()
                    break
                else:
                    for seq_group in swapped_queue:
                        seq_group.update_waiting_iter_nums()
                    break
                # swapped_queue.popleft()
                # leftover_swapped.appendleft(seq_group)
            elif alloc_status == AllocStatus.NEVER:
                logger.warning(
                    "Failing the request %s because there's not enough kv "
                    "cache blocks to run the entire sequence.",
                    seq_group.request_id)
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                infeasible_seq_groups.append(seq_group)
                swapped_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (lora_int_id > 0 and (lora_int_id not in curr_loras)
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_swapped.appendleft(seq_group)
                    swapped_queue.popleft()
                    continue

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.SWAPPED,
                                                      enable_chunking, budget)
            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                if self.scheduler_config.policy == "tfittradeoff":  # Debug
                    seq_group.update_waiting_iter_nums()
                    swapped_queue.popleft()
                    leftover_swapped.appendleft(seq_group)
                    continue
                else:
                    for seq_group in swapped_queue:
                        seq_group.update_waiting_iter_nums()
                    break

            if lora_int_id > 0 and curr_loras is not None:
                curr_loras.add(lora_int_id)
            swapped_queue.popleft()

            if seq_group.metrics.waiting_iter_nums < seq_group.seq_len:
                self.scheduler_metric.total_low_eff_swap_out += 1
                self.scheduler_metric.total_low_eff_swap_out_diff += seq_group.seq_len - \
                                seq_group.metrics.waiting_iter_nums


            self._swap_in(seq_group, blocks_to_swap_in)

            self.scheduler_metric.total_swap_in_blocks += seq_group.total_token_block_size
            self.scheduler_metric.total_swap_in_seqs += 1

            seq_group_request_id = seq_group.request_id
            if seq_group_request_id in self.partial_swapped:
                _, _ = self.partial_swapped.pop(seq_group_request_id)

            self._append_slots(seq_group, blocks_to_copy)
            is_prefill = seq_group.is_prefill()
            if is_prefill:
                prefill_seq_groups.append(
                    ScheduledSequenceGroup(seq_group,
                                           token_chunk_size=num_new_tokens))
                self.scheduler_metric.prefill_token_num += num_new_tokens
            else:
                decode_seq_groups.append(
                    ScheduledSequenceGroup(seq_group, token_chunk_size=1))
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)
            seq_group.reset_waiting_iter_nums()
        swapped_queue.extendleft(leftover_swapped)
        self.scheduler_metric.schedule_swapped_time += time.time() - now

        return swapped_queue, SchedulerSwappedInOutputs(
            decode_seq_groups=decode_seq_groups,
            prefill_seq_groups=prefill_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_copy=blocks_to_copy,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=False),
            infeasible_seq_groups=infeasible_seq_groups,
        )

    def _get_prompt_limit(self, seq_group: SequenceGroup) -> int:
        if self.scheduler_config.chunked_prefill_enabled:
            prompt_limit = self.scheduler_config.max_model_len
        else:
            prompt_limit = min(self.scheduler_config.max_model_len,
                               self.scheduler_config.max_num_batched_tokens)

        # Model is fine tuned with long context. Return the fine tuned max_len.
        if (seq_group.lora_request
                and seq_group.lora_request.long_lora_max_len):
            assert prompt_limit <= seq_group.lora_request.long_lora_max_len
            return seq_group.lora_request.long_lora_max_len
        else:
            return prompt_limit

    def _schedule_prefills(
        self,
        waiting_queue: deque,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        policy_info: PolicyInfo,
        enable_chunking: bool = False,
        policy: Optional[Policy] = None,
    ) -> Tuple[deque, SchedulerPrefillOutputs]:
        """Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            waiting_queue: The queue that contains prefill requests.
                The given arguments are NOT in-place modified.
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            A tuple of remaining waiting_queue after scheduling and
            SchedulerSwappedInOutputs.
        """
        now = time.time()
        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[ScheduledSequenceGroup] = []
        kv_free_seq_groups:List[ScheduledSequenceGroup] = []
        # We don't sort waiting queue because we assume it is sorted.
        # Copy the queue so that the input queue is not modified.
        waiting_queue = deque([s for s in waiting_queue])
        if policy is not None:
            if self.scheduler_config.policy == "tfittradeoff":
                # priorities = [
                #     seq_group.priority_rate for seq_group in self.running 
                #     if seq_group.priority_rate> 0
                # ]
                # avg_priorities = float(
                    # np.max(priorities) if len(priorities) > 0 else 1)
                waiting_queue = policy.sorted_by_priority(
                    waiting_queue, queue_type='waiting', policy_info=policy_info)
            else:
                waiting_queue = policy.sort_by_priority(
                    time.time(), waiting_queue)

        waiting_queue = deque([s for s in self.kv_free_seq_groups]+ [s for s in waiting_queue])

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()

        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]
            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt ",
                f"sequence {seq_group.request_id}, but got {[s.status for s in seq_group.get_seqs()]}")
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.WAITING,
                                                      enable_chunking, budget)
            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d", num_new_tokens, prompt_limit)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            if self.fake_allocate:
                can_allocate = AllocStatus.OK
            else:
                # If the sequence group cannot be allocated, stop.
                can_allocate = self.block_manager.can_allocate(seq_group)
                if can_allocate == AllocStatus.LATER:
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue
                elif can_allocate == AllocStatus.NEVER:
                    logger.warning(
                        "Input prompt (%d tokens) is too long"
                        " and exceeds the capacity of block_manager",
                        num_new_tokens)
                    for seq in waiting_seqs:
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    waiting_queue.popleft()
                    continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                break

            # Can schedule this request.
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()

            if self.fake_allocate:
                self._fake_allocate_and_set_running(seq_group)
            else:
                self._allocate_and_set_running(seq_group)


            seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            self.scheduler_metric.prefill_token_num += num_new_tokens
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)
        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        self.scheduler_metric.schedule_waiting_time += time.time() - now

        for scheduled_seq_group in seq_groups:
            scheduled_seq_group.seq_group.reset_waiting_iter_nums()
        for scheduled_seq_group in kv_free_seq_groups:
            scheduled_seq_group.seq_group.reset_waiting_iter_nums()

        for seq_group in waiting_queue:
            seq_group.update_waiting_iter_nums()
        return waiting_queue, SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            kv_free_seq_groups=kv_free_seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill=True))

    def _schedule_default(self) -> SchedulerOutputs:
        """Schedule queued requests.
        
        The current policy is designed to optimize the throughput. First,
        it batches as many prefill requests as possible. And it schedules
        decodes. If there's a pressure on GPU memory, decode requests can
        be swapped or preempted.
        """
        # Include running requests to the budget.
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        # Make sure we include num running seqs before scheduling prefill,
        # so that we don't schedule beyond max_num_seqs for prefill.
        for seq_group in self.running:
            budget.add_num_seqs(seq_group.request_id,
                                seq_group.get_max_num_running_seqs())
        curr_loras = set(
            seq_group.lora_int_id for seq_group in self.running
            if seq_group.lora_int_id > 0) if self.lora_enabled else None

        remaining_waiting, prefills = (self.waiting,
                                       SchedulerPrefillOutputs.create_empty())
        remaining_running, running_scheduled = (
            self.running, SchedulerRunningOutputs.create_empty())
        remaining_swapped, swapped_in = (
            self.swapped, SchedulerSwappedInOutputs.create_empty())

        # If any requests are swapped, prioritized swapped requests.
        if not self.swapped:
            remaining_waiting, prefills = self._schedule_prefills(
                self.waiting, budget, curr_loras,None, enable_chunking=False)

        policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Don't schedule decodes if prefills are scheduled.
        # NOTE: If `_schedule_prefills` doesn't enable chunking, self.running
        # only contains decode requests, not chunked prefills.
        if len(prefills.seq_groups) == 0:
            remaining_running, running_scheduled, _ = self._schedule_running(
                self.running,
                budget,
                curr_loras,
                policy,
                policy_info=None,
                enable_chunking=False)

            # If any sequence group is preempted, do not swap in any sequence
            # group. because it means there's no slot for new running requests.
            if len(running_scheduled.preempted) + len(
                    running_scheduled.swapped_out) == 0:
                remaining_swapped, swapped_in = self._schedule_swapped(
                    self.swapped, budget, curr_loras, policy, None)

        # assert (budget.num_batched_tokens <=
                # self.scheduler_config.max_num_batched_tokens)
        # assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting = remaining_waiting
        self.waiting.extendleft(running_scheduled.preempted)
        # Update new running requests.
        self.running = remaining_running
        self.running.extend([s.seq_group for s in prefills.seq_groups])
        self.running.extend(
            [s.seq_group for s in running_scheduled.decode_seq_groups])
        self.running.extend(
            [s.seq_group for s in swapped_in.decode_seq_groups])
        # Update swapped requests.
        self.swapped = remaining_swapped
        self.swapped.extend(running_scheduled.swapped_out)
        preempted = (len(running_scheduled.preempted) +
                     len(running_scheduled.swapped_out))

        # There should be no prefill from running queue because this policy
        # doesn't allow chunked prefills.
        assert len(running_scheduled.prefill_seq_groups) == 0
        assert len(swapped_in.prefill_seq_groups) == 0
        return SchedulerOutputs(
            scheduled_seq_groups=(prefills.seq_groups +
                                  running_scheduled.decode_seq_groups +
                                  swapped_in.decode_seq_groups),
            num_prefill_groups=len(prefills.seq_groups),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=running_scheduled.blocks_to_copy +
            swapped_in.blocks_to_copy,
            ignored_seq_groups=prefills.ignored_seq_groups +
            swapped_in.infeasible_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=preempted,
            num_running_to_waiting=0,
            num_waiting_to_running=0,
            recomputed_token_nums=0,
            need_score=False,
            allow_both_swap=self.allow_both_swap
        )

    def _reallocate_shared_block_size(self,num_shared_blocks):
        if len(self.waiting)+len(self.kv_free_seq_groups) ==0:
            self.block_manager.reset_shared_blocks() 

        elif len(self.waiting) > 0 and len(self.kv_free_seq_groups) == 0:
            shared_blocks = self.block_manager.add_shared_blocks(num_shared_blocks)
            print(f"adding shared block size {shared_blocks}")
        else:
            print("skipping reallocate shared block size")

    def _schedule_chunked_prefill(self):
        """Schedule queued requests.
        
        Chunked prefill allows to chunk prefill requests, batch them together
        with decode requests. This policy 1. schedule as many decoding requests
        as possible. 2. schedule chunked prefill requests that are not
        finished. 3. schedule swapped request. 4. schedule new prefill
        requests.

        The policy can sustain the high GPU utilization because it can put
        prefill and decodes requests to the same batch, while it improves
        inter token latency because decodes requests don't need to blocked
        by prefill requests.
        """

        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=max(self.scheduler_config.max_num_seqs, 1024),
        )
        now = time.time()
        # swap_execution_time = [s.get_last_execute_time() for s in self.swapped]
        # waiting_execution_time = [s.get_last_execute_time() for s in self.waiting]
        # max_pending_time = max(swap_execution_time) if len(swap_execution_time) > 0 else 1
        # max_waiting_time = max(waiting_execution_time) if len(waiting_execution_time) > 0 else 1
        policy_info = PolicyInfo(
            waiting_queue_size=len(self.waiting),
            running_queue_size = len(self.running),
            swapped_queue_size = len(self.swapped),
            # max_pending_time=max_pending_time,
            # max_waiting_time=max_waiting_time,
            now=now,
        )
        curr_loras: Set[int] = set()
        if not self.reach_ddl:
            remaining_waiting, prefills = (
                self.waiting, SchedulerPrefillOutputs.create_empty())
            remaining_running, running_scheduled = (
                self.running, SchedulerRunningOutputs.create_empty())
            remaining_swapped, swapped_in = (
                self.swapped, SchedulerSwappedInOutputs.create_empty())
            policy = PolicyFactory.get_policy(
                policy_name=self.scheduler_config.policy)


            if self.scheduler_config.policy in ["sjmlfq", "tfittradeoff","las",'sjf']:
                (remaining_running, remaining_swapped, remaining_waiting,
                running_scheduled, swapped_in,
                prefills, recomputed_token_nums) = \
                    self._schedule_preemption(
                    running_queue=self.running,
                    swapped_queue=self.swapped,
                    waiting_queue=self.waiting,
                    budget=budget,
                    policy=policy,
                    policy_info=policy_info,
                    enable_chunking=True,
                )
            elif self.scheduler_config.policy == "_tfittradeoff":
                remaining_running, running_scheduled, recomputed_token_nums = \
                    self._schedule_running_partial(self.running,
                                        budget,
                                        curr_loras,
                                        policy,
                                        policy_info=policy_info,
                                        enable_chunking=True)
                # Schedule swapped out requests.
                # If preemption happens, it means we don't have space for swap-in.
                # if len(running_scheduled.preempted) + len(
                        # running_scheduled.swapped_out) == 0:
                # if not self.has_preempted_seq:
                # Schedule new prefills.
                remaining_waiting, prefills = self._schedule_prefills(
                    self.waiting,
                    budget,
                    curr_loras,
                    policy=policy,
                    policy_info=policy_info,
                    enable_chunking=True)
                if len(remaining_waiting)>0:
                    self.partial_swap_out_flag=False
                else:
                    self.partial_swap_out_flag=True
                remaining_swapped, swapped_in, = self._schedule_swapped(
                    self.swapped,
                    budget,
                    curr_loras,
                    policy,
                    policy_info=policy_info,
                        enable_chunking=True)
                
            else:
                remaining_running, running_scheduled, recomputed_token_nums = \
                    self._schedule_running(self.running,
                                        budget,
                                        curr_loras,
                                        policy,
                                        policy_info=policy_info,
                                        enable_chunking=True)

                # Schedule swapped out requests.
                # If preemption happens, it means we don't have space for swap-in.
                if len(running_scheduled.preempted) + len(
                        running_scheduled.swapped_out) == 0:
                    remaining_swapped, swapped_in, = self._schedule_swapped(
                        self.swapped,
                        budget,
                        curr_loras,
                        policy,
                        policy_info,
                        enable_chunking=True)

                # Schedule new prefills.
                remaining_waiting, prefills = self._schedule_prefills(
                    self.waiting,
                    budget,
                    curr_loras,
                    policy=policy,
                    policy_info=policy_info,
                    enable_chunking=True)

            # if self.scheduler_config.policy != "tfittradeoff":
            #     assert (budget.num_batched_tokens <=
            #             self.scheduler_config.max_num_batched_tokens)
            #     assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

            # Update waiting requests.
            self.waiting = remaining_waiting
            self.waiting.extendleft(running_scheduled.preempted)
            # Update new running requests.
            self.running = remaining_running
            self.running.extend([s.seq_group for s in prefills.seq_groups])
            self.running.extend([s.seq_group for s in prefills.kv_free_seq_groups])
            self.running.extend(
                [s.seq_group for s in running_scheduled.decode_seq_groups])
            self.running.extend(
                [s.seq_group for s in running_scheduled.prefill_seq_groups])
            self.running.extend(
                [s.seq_group for s in swapped_in.decode_seq_groups])
            self.running.extend(
                [s.seq_group for s in swapped_in.prefill_seq_groups])

            self.kv_free_seq_groups = [s.seq_group.request_id for s in prefills.kv_free_seq_groups]
            # Update swapped requests.
            self.swapped = remaining_swapped
            self.swapped.extend(running_scheduled.swapped_out)
            
            
            self.scheduler_metric.decode_token_num = 0
            for s in (running_scheduled.decode_seq_groups +
                                    swapped_in.decode_seq_groups):
                self.scheduler_metric.decode_token_num += s.seq_group.seq_len

            # Motivation:
            # 1) GPU Memory:

            scheduled_seq_groups = (prefills.seq_groups +
                                    prefills.kv_free_seq_groups +
                                    running_scheduled.prefill_seq_groups +
                                    swapped_in.prefill_seq_groups +
                                    running_scheduled.decode_seq_groups +
                                    swapped_in.decode_seq_groups)

            block_generated = len(scheduled_seq_groups)

            block_used = self.scheduler_metric.decode_token_num + block_generated

            # Normalization
            self.scheduler_metric.gpu_memory_occupy = block_used/ self.cache_config.num_gpu_blocks

            # 2) GPU computation:

            prefills_seq_groups = (prefills.seq_groups +
                                    prefills.kv_free_seq_groups +
                                    running_scheduled.prefill_seq_groups +
                                    swapped_in.prefill_seq_groups )
            computation_tokens = 0

            for seq_group in [s.seq_group for s in prefills_seq_groups]:
                computation_tokens += seq_group.seq_len

            computation_tokens += block_generated

            self.scheduler_metric.gpu_computation_occupy = computation_tokens

            # 3) batch size:


            running_seq_nums = len(scheduled_seq_groups)
            waiting_seq_nums = len(self.swapped)
            pending_seq_nums= len(self.waiting)
            self.scheduler_metric.running_seq_nums += running_seq_nums
            self.scheduler_metric.waiting_seq_nums += waiting_seq_nums
            self.scheduler_metric.pending_seq_nums += pending_seq_nums


            return SchedulerOutputs(
                scheduled_seq_groups=scheduled_seq_groups,
                num_prefill_groups=(len(prefills.seq_groups) +
                                    len(prefills.kv_free_seq_groups)+
                                    len(swapped_in.prefill_seq_groups) +
                                    len(running_scheduled.prefill_seq_groups)),
                num_batched_tokens=budget.num_batched_tokens,
                blocks_to_swap_in=swapped_in.blocks_to_swap_in,
                blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
                blocks_to_copy=running_scheduled.blocks_to_copy +
                swapped_in.blocks_to_copy,
                ignored_seq_groups=prefills.ignored_seq_groups +
                swapped_in.infeasible_seq_groups,
                num_lookahead_slots=running_scheduled.num_lookahead_slots,
                running_queue_size=len(self.running),
                preempted=(len(running_scheduled.preempted) +
                           len(running_scheduled.swapped_out)),
                num_running_to_waiting=len(running_scheduled.preempted),
                num_waiting_to_running=len(
                    running_scheduled.prefill_seq_groups),
                recomputed_token_nums=recomputed_token_nums,
                need_score=False,
                allow_both_swap=self.allow_both_swap
            )
        else:
            ignored_seq_groups: List[SequenceGroup] = []
            for seq_group in (self.waiting + self.running + self.swapped):
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_STOPPED
                # num_running_tokens = self._get_num_new_tokens(
                #     seq_group, SequenceStatus.RUNNING, True, budget)
                ignored_seq_groups.append(seq_group)
            self.waiting = deque()
            self.running = deque()
            self.swapped = deque()
            # self.partial_swapped = deque()
            return SchedulerOutputs(
                scheduled_seq_groups=[],
                num_prefill_groups=0,
                num_batched_tokens=budget.num_batched_tokens,
                blocks_to_swap_in=[],
                blocks_to_swap_out=[],
                blocks_to_copy=[],
                ignored_seq_groups=ignored_seq_groups,
                num_lookahead_slots=0,
                running_queue_size=0,
                preempted=0,
                num_running_to_waiting=0,
                num_waiting_to_running=0,
                recomputed_token_nums=0,
                need_score=False,
                allow_both_swap=self.allow_both_swap
            )

    def _schedule(self) -> SchedulerOutputs:
        """Schedule queued requests."""
        if self.scheduler_config.max_num_seqs != 4096:
            self.scheduler_config.max_num_seqs = 4096 
        if self.scheduler_config.policy == "opt":
            return self._general_schedule()
        if self.scheduler_config.chunked_prefill_enabled :
            self.allow_both_swap = True
            return self._schedule_chunked_prefill()
        else:
            self.allow_both_swap = True
            return self._schedule_default()

    def _can_append_slots(self, seq_group: SequenceGroup) -> bool:
        """Determine whether or not we have enough space in the KV cache to
        continue generation of the sequence group.
        """
        # It is True only for testing case to trigger artificial preemption.
        if (self.enable_artificial_preemption
                and random.uniform(0, 1) < ARTIFICIAL_PREEMPTION_PROB
                and self.artificial_preempt_cnt > 0):
            self.artificial_preempt_cnt -= 1
            return False

        # Appending slots only occurs in decoding.
        is_prefill = False

        return self.block_manager.can_append_slots(
            seq_group=seq_group,
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill),
        )

    def _can_append_slots_prefill(self, seq_group: SequenceGroup) -> bool:
        return self.block_manager.can_append_slots(
            seq_group=seq_group,
            num_lookahead_slots=self._get_num_lookahead_slots(is_prefill=True),
        )

    def _can_allocate_seq(self, block_size: int) -> bool:
        return self.block_manager.can_allocate_infer(block_size)


    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_outputs = self._schedule()
        now = time.time()
        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []

        for i, scheduled_seq_group in enumerate(
                scheduler_outputs.scheduled_seq_groups):
            seq_group = scheduled_seq_group.seq_group
            token_chunk_size = scheduled_seq_group.token_chunk_size
            seq_group.maybe_set_first_scheduled_time(now)
            seq_group.update_last_execute_time()
            # seq_id -> SequenceData
            seq_data: Dict[int, SequenceData] = {}
            # seq_id -> physical block numbers
            block_tables: Dict[int, List[int]] = {}
            shared=False
            if seq_group.request_id in self.kv_free_seq_groups:
                shared=True
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                if self.fake_allocate:
                    block_tables[seq_id] = self.block_manager.get_fake_block_table_and_delete(seq)
                else:
                    block_tables[seq_id] = self.block_manager.get_block_table(seq,shared)
                self.block_manager.access_all_blocks_in_seq(seq, now)

            common_computed_block_nums = (
                self.block_manager.get_common_computed_block_ids(
                    seq_group.get_seqs(status=SequenceStatus.RUNNING)))

            do_sample = True
            if seq_group.is_prefill():
                seqs = seq_group.get_seqs()
                # Prefill has only 1 sequence.
                assert len(seqs) == 1
                # In the next iteration, all prompt tokens are not computed.
                # It means the prefill is chunked, and we don't need sampling.
                # NOTE: We use get_len instead of get_prompt_len because when
                # a sequence is preempted, prefill includes previous generated
                # output tokens.
                if (token_chunk_size + seqs[0].data.get_num_computed_tokens() <
                        seqs[0].data.get_len()):
                    do_sample = False

            # It assumes the scheduled_seq_groups is ordered by
            # prefill < decoding.
            is_prompt = seq_group.is_prefill()
            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=is_prompt,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
                do_sample=do_sample,
                pooling_params=seq_group.pooling_params,
                token_chunk_size=token_chunk_size,
                lora_request=seq_group.lora_request,
                computed_block_nums=common_computed_block_nums,
                state=seq_group.state,
                # `multi_modal_data` will only be present for the 1st comm
                # between engine and worker.
                # the subsequent comms can still use delta, but
                # `multi_modal_data` will be None.
                multi_modal_data=seq_group.multi_modal_data
                if scheduler_outputs.num_prefill_groups > 0 else None,
                eos_token_id=seq_group.eos_token_id,
                prompt_adapter_request=seq_group.prompt_adapter_request,
            )
            seq_group_metadata_list.append(seq_group_metadata)

        # Now that the batch has been created, we can assume all blocks in the
        # batch will have been computed before the next scheduling invocation.
        # This is because the engine assumes that a failure in model execution
        # will crash the vLLM instance / will not retry.
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            self.block_manager.mark_blocks_as_computed(
                scheduled_seq_group.seq_group)
        
        return seq_group_metadata_list, scheduler_outputs

    def reset_schedule_metric(self):
        self.scheduler_metric = SchedulerMetric()
        return
    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        """Free a sequence from a block table."""
        self.block_manager.free(seq)

    def free_finished_seq_groups(self) -> None:
        running_queue_length = len(self.running)
        for queue in [self.running, self.swapped, self.waiting]:
            self._finished_requests_ids += [
                seq_group.request_id for seq_group in queue
                if seq_group.is_finished()
            ]
        self.running = deque(seq_group for seq_group in self.running
                             if not seq_group.is_finished())
        if len(self.running) < running_queue_length:
            self.has_finished_seqs = True

    def _allocate_and_set_running(self, seq_group: SequenceGroup, shared=False) -> None:
        self.block_manager.allocate(seq_group, shared=shared)
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.status = SequenceStatus.RUNNING
            if shared:
                seq.set_seq_type(SequenceType.TEMP)
            else:
                seq.set_seq_type(SequenceType.NORMAL)
            seq.status_transmit = SequenceStatus.WAITING_TO_RUNNING


    def _fake_allocate_and_set_running(self, seq_group: SequenceGroup)-> None:
        self.block_manager.fake_allocate(seq_group)
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.status = SequenceStatus.RUNNING

    def _append_slots(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: List[Tuple[int, int]],
    ) -> None:
        """Appends new slots to the sequences in the given sequence group.

        Args:
            seq_group (SequenceGroup): The sequence group containing the
                sequences to append slots to.
            blocks_to_copy (List[Tuple[int, int]]): A list of tuple of two
                ints, the first int is the source block index, and the second
                int is the destination block index. This list is updated with
                the new source and destination block indices for the appended
                slots.
        """
        num_lookahead_slots = self._get_num_lookahead_slots(is_prefill=False)

        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            cows = self.block_manager.append_slots(seq, num_lookahead_slots)
            blocks_to_copy.extend(cows)

    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
        preemption_mode: Optional[PreemptionMode] = None,
        swap_out_block_nums: int = -1,
        seq_group_status: SequenceStatus = SequenceStatus.SWAPPED
    ) -> PreemptionMode:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if preemption_mode is None:
            if self.user_specified_preemption_mode is None:
                # if self.block_manager.can_swap_in_shared_blocks(seq_group):
                #     print(f"{seq_group.request_id} can swap in shared blocks")
                #     preemption_mode = PreemptionMode.KV_FREE_RECOMPUTE
                if seq_group.get_max_num_running_seqs(
                ) == 1 or not self.block_manager.can_swap_out(seq_group):
                    preemption_mode = PreemptionMode.RECOMPUTE
                else:
                    preemption_mode = PreemptionMode.SWAP

            elif self.user_specified_preemption_mode == "swap" and self.block_manager.can_swap_out(
                    seq_group):
                preemption_mode = PreemptionMode.SWAP
            else:
                preemption_mode = PreemptionMode.RECOMPUTE
        
        # elif self.block_manager.can_swap_in_shared_blocks(seq_group):
        #     print(f"{seq_group.request_id} can swap in shared blocks")
        #     preemption_mode = PreemptionMode.KV_FREE_RECOMPUTE
        else:
            if not self.block_manager.can_swap_out(seq_group) and seq_group_status != SequenceStatus.PARTIAL_SWAPPED:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = preemption_mode

        if (self.num_cumulative_preemption % 50 == 0
                and self.num_cumulative_preemption > 0):
            logger.debug(
                "Sequence group %s is preempted by %s mode because there is "
                "not enough KV cache space. This can affect the end-to-end "
                "performance. Increase gpu_memory_utilization or "
                "tensor_parallel_size to provide more KV cache memory. "
                "total_num_cumulative_preemption=%d", seq_group.request_id,
                preemption_mode, self.num_cumulative_preemption + 1)
        self.num_cumulative_preemption += 1
        if preemption_mode == PreemptionMode.RECOMPUTE or preemption_mode == PreemptionMode.KV_FREE_RECOMPUTE:
            # preemption_mode = PreemptionMode.RECOMPUTE
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group,
                                  blocks_to_swap_out,
                                  swap_out_block_nums,
                                  seq_group_status=seq_group_status)
        return preemption_mode

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:

        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        # assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            seq.status_transmit = SequenceStatus.RUNNING_TO_WAITING
            self.free_seq(seq)
            seq.reset_state_for_recompute()


    def _preempt_by_swap(
            self,
            seq_group: SequenceGroup,
            blocks_to_swap_out: List[Tuple[int, int]],
            swap_out_block_nums: int = -1,
            seq_group_status: SequenceStatus = SequenceStatus.SWAPPED) -> None:
        self._swap_out(seq_group,
                       blocks_to_swap_out,
                       swap_out_block_nums,
                       seq_group_status=seq_group_status)

    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: List[Tuple[int, int]],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.extend(mapping)
        seqs = seq_group.get_seqs(
            status=SequenceStatus.SWAPPED) + seq_group.get_seqs(
                status=SequenceStatus.PARTIAL_SWAPPED)


        for seq in seqs:
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
            self,
            seq_group: SequenceGroup,
            blocks_to_swap_out: List[Tuple[int, int]],
            swap_out_block_nums: int = -1,
            seq_group_status: SequenceStatus = SequenceStatus.SWAPPED) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            # self._preempt_by_recompute(seq_group)
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
            # return
        mapping = self.block_manager.swap_out(
            seq_group, swap_out_block_nums=swap_out_block_nums)
        blocks_to_swap_out.extend(mapping)

        # asyncio.run(self._async_swap([],blocks_to_swap_out,[]))
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = seq_group_status

    def _passed_delay(self, now: float) -> bool:
        if self.prev_prompt:
            self.last_prompt_latency = now - self.prev_time
        self.prev_time, self.prev_prompt = now, False
        # Delay scheduling prompts to let waiting queue fill up
        if self.scheduler_config.delay_factor > 0 and self.waiting:
            earliest_arrival_time = min(
                [e.metrics.arrival_time for e in self.waiting])
            passed_delay = (
                (now - earliest_arrival_time) >
                (self.scheduler_config.delay_factor * self.last_prompt_latency)
                or not self.running)
        else:
            passed_delay = True
        return passed_delay

    def _get_num_lookahead_slots(self, is_prefill: bool) -> int:
        """The number of slots to allocate per sequence per step, beyond known
        token ids. Speculative decoding uses these slots to store KV activations
        of tokens which may or may not be accepted.

        Speculative decoding does not yet support prefill, so we do not perform
        lookahead allocation for prefill.
        """
        if is_prefill:
            return 0

        return self.scheduler_config.num_lookahead_slots

    def _get_num_new_tokens(self, seq_group: SequenceGroup,
                            status: SequenceStatus, enable_chunking: bool,
                            budget: SchedulingBudget) -> int:
        """Get the next new tokens to compute for a given sequence group
            that's in a given `status`.

        The API could chunk the number of tokens to compute based on `budget`
        if `enable_chunking` is True. If a sequence group has multiple
        sequences (e.g., running beam search), it means it is in decoding
        phase, so chunking doesn't happen.

        Returns 0 if the new token cannot be computed due to token budget.
        """
        num_new_tokens = 0
        if status == SequenceStatus.SWAPPED:
            seqs = seq_group.get_seqs(status=status) + seq_group.get_seqs(
                status=SequenceStatus.PARTIAL_SWAPPED)
        else:
            seqs = seq_group.get_seqs(status=status)
        for seq in seqs:
            num_new_tokens += seq.get_num_new_tokens()

        assert num_new_tokens > 0, f"{seq_group.get_seqs()}, \
                                    {seq_group.request_id}"

        # Chunk if a running request cannot fit in.
        # If number of seq > 1, it means it is doing beam search in a
        # decode phase. Do not chunk in that case.
        if enable_chunking and len(seqs) == 1:
            num_new_tokens = min(num_new_tokens,
                                 budget.remaining_token_budget())
        return num_new_tokens

    def max_numbers_sum_at_most(self, numbers: List[int], target: int) -> int:
        prefix_sum = list(accumulate(numbers))

        # Use bisect_right for binary search
        index = bisect.bisect_right(prefix_sum, target)

        if index >= len(prefix_sum):
            return -1
        return index

    def min_numbers_sum_at_least(self, numbers: List[int], target: int) -> int:
        """
        Find the minimum sum of numbers that is at least `target`.
        """
        # Calculate prefix sums using accumulate
        prefix_sum = list(accumulate(numbers))

        # Use bisect_left for binary search
        index = bisect.bisect_left(prefix_sum, target)

        if index >= len(prefix_sum):
            return -1

        return index + 1  # return the number of elements needed

    def is_opt(self, seq_group: SequenceGroup):
        pass