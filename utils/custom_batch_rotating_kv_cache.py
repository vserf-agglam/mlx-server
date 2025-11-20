from typing import List, Optional, Tuple, Any
import logging
import mlx.core as mx
from mlx_lm.models.cache import _BaseCache

logger = logging.getLogger(__name__)


class CustomBatchRotatingKVCache(_BaseCache):
    """
    A Rotating KV Cache that supports 'Attention Sinks' (Keep).
    """

    step = 256

    def __init__(
            self,
            max_size: int,
            left_padding: List[int],
            keep: int = 4
    ):
        self.max_size = max_size

        # SAFETY 1: Sanitize 'keep'.
        # A rotating cache MUST have at least 1 slot for the ring buffer.
        # If user sets keep >= max_size, we force it down.
        if keep >= max_size:
            safe_keep = max(0, max_size - 64)  # Reserve 64 slots for rotation
            logger.warning(
                "⚠️ CONFIG ERROR: kv_keep (%d) >= max_kv_size (%d). "
                "This leaves no room for rotation. Forcing keep=%d.",
                keep, max_size, safe_keep
            )
            self.keep = safe_keep
        else:
            self.keep = keep

        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None

        self.left_padding = mx.array(left_padding) if left_padding is not None else mx.array([])

        if self.left_padding.size > 0:
            self.offset = mx.array([-l for l in left_padding])
        else:
            self.offset = mx.array([0])

        self._idx = 0
        self._offset = 0
        self.rotated = False

        # Statistics
        self._rotation_count = 0
        self._total_tokens = 0
        self._trim_count = 0
        self._in_place_updates = 0
        self._concat_updates = 0

        logger.debug(
            "RotatingKVCache initialized | max_size=%d, keep=%d",
            self.max_size, self.keep
        )

    def _trim(self, trim_size: int, v: mx.array, append: Optional[mx.array] = None) -> mx.array:
        if trim_size > 0:
            to_cat = [
                v[..., : self.keep, :],
                v[..., self.keep + trim_size:, :]
            ]
            self._trim_count += 1
        else:
            to_cat = [v]

        if append is not None:
            to_cat.append(append)

        return mx.concatenate(to_cat, axis=2)

    def _temporal_order(self):
        if self.rotated:
            if self.keep > 0:
                k_keep = self.keys[..., :self.keep, :]
                v_keep = self.values[..., :self.keep, :]

                k_ring = self.keys[..., self.keep:, :]
                v_ring = self.values[..., self.keep:, :]

                shift = -(self._idx - self.keep)
                k_ring = mx.roll(k_ring, shift, axis=2)
                v_ring = mx.roll(v_ring, shift, axis=2)

                self.keys = mx.concatenate([k_keep, k_ring], axis=2)
                self.values = mx.concatenate([v_keep, v_ring], axis=2)
            else:
                self.keys = mx.roll(self.keys, -self._idx, axis=2)
                self.values = mx.roll(self.values, -self._idx, axis=2)

            self._idx = self.keys.shape[2]
            self.rotated = False

    def _update_concat(self, keys: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        # 1. Linearize
        if self.keys is not None:
            self._temporal_order()
            k_cat = mx.concatenate([self.keys, keys], axis=2)
            v_cat = mx.concatenate([self.values, values], axis=2)
        else:
            k_cat, v_cat = keys, values

        # 2. Strict Size Enforcement
        total_tokens = k_cat.shape[2]

        if total_tokens > self.max_size:
            logger.warning(
                "✂️ CACHE TRUNCATION | Total (%d) > Max (%d) | Dropping %d tokens",
                total_tokens, self.max_size, total_tokens - self.max_size
            )

            # A) Sinks (0 .. keep)
            k_sinks = k_cat[..., :self.keep, :]
            v_sinks = v_cat[..., :self.keep, :]

            # B) Ring/History
            needed_ring = self.max_size - self.keep

            # SAFETY 2: Handle negative slicing bug.
            # If needed_ring is 0, we want an empty slice, NOT the whole array.
            if needed_ring > 0:
                # Take the last N tokens
                k_ring = k_cat[..., -needed_ring:, :]
                v_ring = v_cat[..., -needed_ring:, :]

                self.keys = mx.concatenate([k_sinks, k_ring], axis=2)
                self.values = mx.concatenate([v_sinks, v_ring], axis=2)
            else:
                # Edge case: No ring buffer space at all (keep ~= max_size)
                self.keys = k_sinks
                self.values = v_sinks

            if self.left_padding is not None:
                self.left_padding = mx.zeros_like(self.left_padding)
        else:
            self.keys, self.values = k_cat, v_cat

        # 3. Update State
        added_len = keys.shape[2]
        self.offset += added_len
        self._offset += added_len

        self._idx = self.keys.shape[2]
        self.rotated = False

        # 4. Paranoid Check
        if self.keys.shape[2] > self.max_size:
            # This should theoretically never happen now with SAFETY 2 fix
            logger.error("CRITICAL: Truncation math error. Hard clamping.")
            self.keys = self.keys[..., :self.max_size, :]
            self.values = self.values[..., :self.max_size, :]
            self._idx = self.max_size

        return self.keys, self.values

    def _update_in_place(self, keys: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        B, n_kv_heads, S, k_head_dim = keys.shape
        prev = self._offset

        if self.keys is None or (
                prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            v_head_dim = values.shape[3]
            new_size = min(self.step, self.max_size - self.keys.shape[2] if self.keys is not None else self.max_size)

            k_shape = (B, n_kv_heads, new_size, k_head_dim)
            v_shape = (B, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)

            if self.keys is not None:
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        if self._idx + S > self.max_size:
            self.rotated = True
            # Only log infrequently to avoid spamming loops
            if self._rotation_count % 10 == 0:
                logger.warning(
                    "🔄 CACHE ROTATION | Capacity Full (%d) | Wrapping to %d",
                    self.max_size, self.keep
                )
            self._idx = self.keep
            self._rotation_count += 1
            self.left_padding = mx.zeros_like(self.left_padding)

        self.keys[..., self._idx: self._idx + S, :] = keys
        self.values[..., self._idx: self._idx + S, :] = values

        self._offset += S
        self.offset += S
        self._idx += S

        if self._offset < self.max_size:
            return (
                self.keys[..., : self._offset, :],
                self.values[..., : self._offset, :],
            )
        return self.keys, self.values

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        self._total_tokens += keys.shape[2]
        if keys.shape[2] == 1:
            self._in_place_updates += 1
            return self._update_in_place(keys, values)
        else:
            self._concat_updates += 1
            return self._update_concat(keys, values)

    def make_mask(
            self, N: int, window_size: Optional[int] = None, return_array: bool = False
    ):
        window_size = window_size or self.max_size

        if N > 1:
            post_update_offset = self._offset + N
            current_cache_len = min(self.max_size, post_update_offset)

            if self._offset == 0:
                mask = mx.tril(mx.ones((N, N), dtype=mx.bool_))
                if self.left_padding.size > 0 and self.left_padding.min() > 0:
                    rinds = mx.arange(N)[None, :]
                    mask &= (rinds >= mx.expand_dims(self.left_padding, (1, 2)))
                return mask

            mask_local = mx.tril(mx.ones((N, N), dtype=mx.bool_))

            if current_cache_len > N:
                history_size = current_cache_len - N
                mask_history = mx.ones((N, history_size), dtype=mx.bool_)
                mask = mx.concatenate([mask_history, mask_local], axis=1)
            else:
                mask = mask_local

            if not self.rotated and self.left_padding.size > 0:
                rinds = mx.arange(current_cache_len)[None, :]
                mask &= (rinds >= mx.expand_dims(self.left_padding, (1, 2)))

            return mask

        current_cache_len = min(self.max_size, self._offset + 1)
        offset = current_cache_len - 1
        rinds = mx.arange(current_cache_len)
        linds = mx.arange(offset, offset + N)
        linds = linds[:, None]
        rinds = rinds[None]
        mask = linds >= rinds
        mask &= linds < rinds + window_size

        if self.rotated:
            if self.keep > 0:
                mask_keep = mask[..., :self.keep]
                mask_ring = mask[..., self.keep:]
                shift = self._idx - self.keep
                mask_ring = mx.roll(mask_ring, shift, axis=-1)
                mask = mx.concatenate([mask_keep, mask_ring], axis=-1)
            else:
                mask = mx.roll(mask, self._idx, axis=-1)

        return mask

    @property
    def state(self):
        k, v = self.keys, self.values
        valid_len = self.keys.shape[2] if self.keys is not None else 0
        if self._offset < valid_len:
            k, v = k[..., : self._offset, :], v[..., : self._offset, :]
        return k, v, self.offset, self.left_padding

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.left_padding = v
        # Don't override max_size here, trust __init__
        self.rotated = (self.keys.shape[2] == self.max_size)
        self._offset = int(mx.max(self.offset).item())

        self._idx = self.keys.shape[2]
        if self._idx == self.max_size:
            self.rotated = True
            self._idx = self.keep

    @property
    def meta_state(self):
        return tuple(map(str, (self.max_size, self._offset, self._idx, self.rotated, self.keep)))

    @meta_state.setter
    def meta_state(self, v):
        self.max_size, self._offset, self._idx = map(int, v[:3])
        self.rotated = bool(v[3])
        if len(v) > 4:
            self.keep = int(v[4])

    def get_stats(self):
        return {
            "total_tokens": self._total_tokens,
            "rotation_count": self._rotation_count,
            "in_place_updates": self._in_place_updates,
            "concat_updates": self._concat_updates,
            "fill_rate": (min(self._offset, self.max_size) / self.max_size) * 100,
            "rotated": self.rotated,
            "current_idx": self._idx
        }