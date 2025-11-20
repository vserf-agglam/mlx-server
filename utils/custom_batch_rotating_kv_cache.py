from typing import List, Optional, Tuple
import logging
import mlx.core as mx
from mlx_lm.models.cache import _BaseCache

logger = logging.getLogger(__name__)


class CustomBatchRotatingKVCache(_BaseCache):
    """
    A Rotating KV Cache that supports 'Attention Sinks' (Keep).

    This cache maintains the first `keep` tokens (sinks) indefinitely while
    treating the remaining `max_size - keep` slots as a circular buffer.

    References:
    - Efficient Streaming Language Models with Attention Sinks (Xiao et al., 2023)
    """

    step = 256

    def __init__(
            self,
            max_size: int,
            left_padding: List[int],
            keep: int = 4  # Default sink size typically 4
    ):
        self.keys = None
        self.values = None
        self.keep = keep
        self.max_size = max_size

        # Padding is critical for batch alignment, but irrelevant once we rotate
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])

        self._idx = 0
        self._offset = 0
        self.rotated = False
        
        # Statistics tracking
        self._rotation_count = 0
        self._total_tokens = 0
        self._trim_count = 0
        self._in_place_updates = 0
        self._concat_updates = 0
        
        logger.info(
            "RotatingKVCache initialized | max_size=%d, keep=%d, batch_size=%d",
            self.max_size,
            self.keep,
            len(left_padding)
        )

    def _trim(self, trim_size, v, append=None):
        """
        Trims the cache while preserving the 'keep' region.
        Removes the oldest tokens from the ring buffer segment.
        """
        to_cat = []
        if trim_size > 0:
            # Keep the sinks, trim the start of the ring
            to_cat = [
                v[..., : self.keep, :],
                v[..., self.keep + trim_size:, :]
            ]
            self._trim_count += 1
            logger.info(
                "Trim operation | removing %d tokens from ring buffer (trim_count=%d)",
                trim_size,
                self._trim_count
            )
        else:
            to_cat = [v]

        if append is not None:
            to_cat.append(append)

        return mx.concatenate(to_cat, axis=2)

    def _temporal_order(self):
        """
        Rearrange the cache into linear temporal order (Logical Layout).
        Necessary when resizing or debugging, but expensive to call frequently.
        """
        if self.rotated:
            # The buffer is physically [Sinks, Oldest_Ring ... Newest_Ring]
            # We need to unroll the ring part.
            if self.keep > 0:
                k_keep = self.keys[..., :self.keep, :]
                v_keep = self.values[..., :self.keep, :]

                # The ring buffer effectively starts at self.keep
                k_ring = self.keys[..., self.keep:, :]
                v_ring = self.values[..., self.keep:, :]

                # Logic: _idx points to the *next* write slot.
                # So _idx is the "head" (oldest) of the ring history,
                # and _idx - 1 is the "tail" (newest).
                # We roll so that _idx moves to index 0 of the ring.
                shift = -(self._idx - self.keep)
                k_ring = mx.roll(k_ring, shift, axis=2)
                v_ring = mx.roll(v_ring, shift, axis=2)

                self.keys = mx.concatenate([k_keep, k_ring], axis=2)
                self.values = mx.concatenate([v_keep, v_ring], axis=2)
            else:
                # Standard full rotation
                self.keys = mx.roll(self.keys, -self._idx, axis=2)
                self.values = mx.roll(self.values, -self._idx, axis=2)

            self._idx = self.keys.shape[2]
            self.rotated = False

    def _update_concat(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            # To resize via concat, we must linearize first
            self._temporal_order()

            # Trim overshoot if generation exceeds known limits
            if self.keys.shape[2] > self._idx:
                self.keys = self.keys[..., : self._idx, :]
                self.values = self.values[..., : self._idx, :]

            trim_size = self._idx - self.max_size + 1
            if trim_size > 0:
                self.left_padding = mx.maximum(self.left_padding - trim_size, 0)

            self.keys = self._trim(trim_size, self.keys, keys)
            self.values = self._trim(trim_size, self.values, values)

        self.offset += keys.shape[2]
        self._offset += keys.shape[2]
        self._idx = self.keys.shape[2]
        return self.keys, self.values

    def _update_in_place(self, keys, values):
        """
        Optimized circular buffer update.
        Writes new tokens to self._idx, wrapping around 'keep' if full.
        """
        B, n_kv_heads, S, k_head_dim = keys.shape
        prev = self._offset

        # 1. Grow if not yet at max_size
        if self.keys is None or (
                prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            v_head_dim = values.shape[3]
            # Grow by step, but don't exceed max_size
            new_size = min(self.step, self.max_size - prev)
            k_shape = (B, n_kv_heads, new_size, k_head_dim)
            v_shape = (B, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)

            if self.keys is not None:
                old_size = self.keys.shape[2]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
                logger.info(
                    "Cache growing | %d → %d tokens (%.1f%% full)",
                    old_size,
                    self.keys.shape[2],
                    (self.keys.shape[2] / self.max_size) * 100
                )
            else:
                self.keys, self.values = new_k, new_v
                logger.info(
                    "Cache initialized | size=%d tokens (%.1f%% of max)",
                    new_size,
                    (new_size / self.max_size) * 100
                )
            self._idx = prev

        # 2. Handle Trim (if we somehow switched from a larger cache state)
        trim_size = self.keys.shape[2] - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size
            self.left_padding = mx.maximum(self.left_padding - trim_size, 0)

        # 3. Check for Rotation (Buffer Full)
        # If we are at the end, wrap around to `keep`
        if self._idx == self.max_size:
            self.rotated = True
            self._idx = self.keep
            self._rotation_count += 1
            logger.info(
                "Cache rotation started | token=%d, entering circular mode (rotation #%d)",
                self._offset,
                self._rotation_count
            )

        # Once rotated, left_padding is semantically irrelevant (we have full context)
        # We zero it out to prevent mask artifacts.
        if self.rotated:
            self.left_padding = mx.zeros_like(self.left_padding)

        # 4. Assign new tokens
        # Note: This implementation assumes S (tokens to add) fits within the remaining space
        # before a wrap is needed. For S=1 (generation), this is always true.
        self.keys[..., self._idx: self._idx + S, :] = keys
        self.values[..., self._idx: self._idx + S, :] = values

        self._offset += S
        self.offset += S
        self._idx += S

        # Return valid slice
        if self._offset < self.max_size:
            return (
                self.keys[..., : self._offset, :],
                self.values[..., : self._offset, :],
            )
        return self.keys, self.values

    def update_and_fetch(self, keys, values):
        # S=1 implies generation phase -> use optimized in-place ring buffer
        self._total_tokens += keys.shape[2]
        if keys.shape[2] == 1:
            self._in_place_updates += 1
            if self._in_place_updates % 100 == 0:  # Log every 100 in-place updates
                logger.info(
                    "In-place update | idx=%d, rotated=%s, fill=%.1f%%, total_tokens=%d",
                    self._idx,
                    self.rotated,
                    (min(self._offset, self.max_size) / self.max_size) * 100,
                    self._total_tokens
                )
            return self._update_in_place(keys, values)
        else:
            self._concat_updates += 1
            logger.info(
                "Concat update | adding %d tokens, method=concat (update #%d)",
                keys.shape[2],
                self._concat_updates
            )
            return self._update_concat(keys, values)

    def make_mask(
            self, N: int, window_size: Optional[int] = None, return_array: bool = False
    ):
        """
        Constructs an attention mask that respects the 'keep' + 'ring' layout.
        """
        # Default window is the full cache size
        window_size = window_size or self.max_size

        # We need a mask of shape (L, R) where L is current tokens, R is history
        # For generation (N=1), R is the current cache size.

        current_cache_len = min(self.max_size, self._offset)
        offset = current_cache_len - N if not self.rotated else current_cache_len - 1

        # 1. Generate standard Causal Mask / Window Mask
        # Create indices [0, 1, ... current_cache_len]
        rinds = mx.arange(current_cache_len)
        # For N=1 (generation), linds is just a single scalar [current_pos]
        # But we need the mask to match the cache shape.
        linds = mx.arange(offset, offset + N) if offset >= 0 else rinds

        linds = linds[:, None]
        rinds = rinds[None]

        # Standard Causal Logic: L >= R
        mask = linds >= rinds
        # Window Logic: L < R + window
        mask &= linds < rinds + window_size

        # 2. Handle Left Padding
        # If we are rotated, self.left_padding is 0, so this effect vanishes.
        if not self.rotated:
            mask &= (rinds >= mx.expand_dims(self.left_padding, (1, 2, 3)))

        # 3. Handle Rotation (The "Split-and-Roll")
        # If rotated, the physical cache is not in causal order.
        # Layout: [Keep (0..k)] + [Ring (k..max)]
        # The Ring is rotated such that `_idx` is the split point.

        # We only need to adjust the mask if we are reading from a rotated cache.
        # For N=1 generation, we usually query against the full cache.
        is_gen_step = (N == 1)
        if self.rotated and is_gen_step:
            if self.keep > 0:
                # Split the mask into Keep and Ring regions
                mask_keep = mask[..., :self.keep]
                mask_ring = mask[..., self.keep:]

                # We need to align the mask's causal diagonal with the physical data.
                # The physical data at `_idx` corresponds to the "oldest" ring data.
                # The physical data at `_idx - 1` is the "newest".
                # Standard mask assumes newest is at the end.
                # So we roll the RING part of the mask to match the buffer rotation.

                # Shift calculation:
                # The write pointer is at self._idx.
                # We want the mask's "end" to align with self._idx - 1.
                shift = self._idx - self.keep

                mask_ring = mx.roll(mask_ring, shift, axis=-1)

                # Reassemble
                mask = mx.concatenate([mask_keep, mask_ring], axis=-1)
            else:
                # Standard Full Rotation (no keep)
                mask = mx.roll(mask, self._idx, axis=-1)

        return mask

    @property
    def state(self):
        k, v = self.keys, self.values
        # If not full, return only the filled portion
        if self._offset < k.shape[2]:
            k, v = k[..., : self._offset, :], v[..., : self._offset, :]
        return k, v, self.offset, self.left_padding

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.left_padding = v
        self.rotated = (self.keys.shape[2] == self.max_size)
        # Re-calculate _idx and _offset based on loaded state
        # This is an approximation; exact reconstruction requires meta_state
        self._idx = self.keys.shape[2]
        self._offset = int(mx.max(self.offset).item())

    @property
    def meta_state(self):
        return tuple(map(str, (self.max_size, self._offset, self._idx, self.rotated, self.keep)))
    
    def get_stats(self):
        """Return cache statistics for monitoring"""
        return {
            "total_tokens": self._total_tokens,
            "rotation_count": self._rotation_count,
            "trim_count": self._trim_count,
            "in_place_updates": self._in_place_updates,
            "concat_updates": self._concat_updates,
            "fill_rate": (min(self._offset, self.max_size) / self.max_size) * 100,
            "rotated": self.rotated,
            "current_idx": self._idx,
            "max_size": self.max_size,
            "keep_size": self.keep
        }

    @meta_state.setter
    def meta_state(self, v):
        self.max_size, self._offset, self._idx = map(int, v[:3])
        self.rotated = bool(v[3])
        if len(v) > 4:
            self.keep = int(v[4])