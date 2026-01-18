from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


INVALID_BLOCK_HASH = -1

class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = INVALID_BLOCK_HASH
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = INVALID_BLOCK_HASH
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, max_soft_mtp_tokens: int = 1):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.max_soft_mtp_tokens = max_soft_mtp_tokens
        self.soft_mtp_enabled = max_soft_mtp_tokens > 1
        assert max_soft_mtp_tokens <= 2, "Current Soft MTP prefill_allocate only supports 2 tokens"

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        soft_mtp_special_case = self.soft_mtp_enabled and seq.last_block_num_tokens == self.block_size
        if soft_mtp_special_case:
            return len(self.free_block_ids) >= seq.num_blocks + 1
        return len(self.free_block_ids) >= seq.num_blocks

    def prefill_allocate(self, seq: Sequence):
        assert not seq.block_table
        h = INVALID_BLOCK_HASH
        cache_miss = False
        # When we are in soft MTP mode and the last block is full, we need to allocate an
        # additional block for the MTP module, since it is off by 1.
        soft_mtp_special_case = self.soft_mtp_enabled and seq.last_block_num_tokens == self.block_size
        num_blocks = seq.num_blocks if not soft_mtp_special_case else seq.num_blocks + 1
        for i in range(num_blocks):
            if soft_mtp_special_case and i == num_blocks - 1:
                # in this case, we are allocating an additional block for the MTP module.
                # Use -1 just as a random placeholder. The actual number doesn't matter
                token_ids = [-1]
                block_len = len(token_ids)
            else:
                token_ids = seq.block(i)
                block_len = len(token_ids)
                if self.soft_mtp_enabled:
                    # for soft mtp, we use the uncompressed token ids to compute the hash
                    token_ids = seq.uncompressed_block(i)

            h = self.compute_hash(token_ids, h) if block_len == self.block_size else INVALID_BLOCK_HASH
            # h != INVALID_BLOCK_HASH means we need to assign a new block to this sequence

            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != INVALID_BLOCK_HASH:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= ((len(seq) + self.max_soft_mtp_tokens - 1) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]

        if len(seq) % self.block_size == 0:
            assert last_block.hash == INVALID_BLOCK_HASH
            if self.soft_mtp_enabled:
                token_ids = seq.uncompressed_block(seq.num_blocks-1)
            else:
                token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else INVALID_BLOCK_HASH
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id

        # we need to allocate a new block if the last MTP modules will need the next block
        if (len(seq) + self.max_soft_mtp_tokens - 1) % self.block_size == 1:
            assert last_block.hash != INVALID_BLOCK_HASH
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
