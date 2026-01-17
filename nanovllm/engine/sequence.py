from copy import copy
from enum import Enum, auto
from itertools import chain, count

from nanovllm.sampling_params import SamplingParams

COT_PAD_TOKEN_ID = -1
END_OF_THINK_TOKEN_ID = -1

INVALID_TOKEN_ID = -1

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.uncompressed_token_ids_by_block: list[list[int]] = [
            copy(token_ids[i:i+self.block_size]) 
            for i in range(0, len(token_ids), self.block_size)
        ]
        self.next_input_cot_ids: tuple[int, ...] = ()
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table: list[int] = []
        self.sampling_params: SamplingParams = sampling_params

        # When the MTP module generates the EOT token, it will set this flag to True
        # Since we still need to process the token from the NTP module, (and soft embed it with COT_PAD)
        # We need to set this flag so that after we do another step of soft_mtp generation,
        # EOT will be the "last token" when we switch back to the normal decoding.
        self.eot_from_mtp_module = False

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size
    
    @property
    def uncompressed_token_ids(self):
        return list(chain.from_iterable(self.uncompressed_token_ids_by_block))

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        if self.num_tokens % self.block_size == 0:
            self.uncompressed_token_ids_by_block.append([token_id])
        else:
            self.uncompressed_token_ids_by_block[-1].append(token_id)
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def uncompressed_block(self, i):
        assert 0 <= i < self.num_blocks
        return copy(self.uncompressed_token_ids_by_block[i])

    def append_soft_mtp_tokens(self, token_ids: tuple[int, ...]):
        assert len(token_ids) == 2
        assert not self.eot_from_mtp_module, "After reaching this state, we should not append more tokens"

        if self.num_tokens % self.block_size == 0:
            self.uncompressed_token_ids_by_block.append(list(token_ids))
        else:
            self.uncompressed_token_ids_by_block[-1].extend(token_ids)

        if token_ids[-1] == END_OF_THINK_TOKEN_ID:
            self.eot_from_mtp_module = True
            token_ids = (token_ids[0], COT_PAD_TOKEN_ID)
        self.next_input_cot_ids = token_ids

        if token_ids[-1] == COT_PAD_TOKEN_ID:
            self.token_ids.append(token_ids[0])
        elif token_ids[0] == END_OF_THINK_TOKEN_ID:
            # This is a normal case, where the ntp module generated the EOT token
            self.token_ids.append(END_OF_THINK_TOKEN_ID)
        else:
            self.token_ids.append(INVALID_TOKEN_ID)

        self.last_token = self.token_ids[-1]
        self.num_tokens += 1

    def apply_eot_from_mtp_module(self):
        # Add the EOT token and increment num_tokens for the soft token from this step
        self.token_ids.append(END_OF_THINK_TOKEN_ID)
        self.last_token = END_OF_THINK_TOKEN_ID
        self.num_tokens += 1
        self.eot_from_mtp_module = False

    # TODO: add uncompressed_token_ids to state
    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
