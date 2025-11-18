import hashlib
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List

from api.types import OutputTextContentItem, OutputToolContentItem

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from token_parser.base_token_parser import BaseTokenParser


def make_tool_call_id(raw_tool_json: str) -> str:
    """
    Generate a stable tool_use id from the raw JSON payload.

    The same tool JSON will always produce the same id, which allows
    streaming tool events and the final MessagesResponse to stay in sync.
    """
    raw = raw_tool_json.strip()
    if not raw:
        # Fallback to a fixed suffix; the exact value is not important,
        # it just needs to be non-empty.
        logger.debug("make_tool_call_id: empty raw_tool_json, returning 'toolu_empty'")
        return "toolu_empty"

    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    tool_id = f"toolu_{digest}"
    logger.debug(
        "make_tool_call_id: raw_len=%d, digest=%s, tool_id=%s",
        len(raw),
        digest,
        tool_id,
    )
    return tool_id


class ToolStreamHelper:
    """
    Incrementally parses streaming text chunks and extracts tool calls,
    delegating interpretation to the configured token parser.

    As text is fed in via `feed`, this helper emits a sequence of
    high-level events:
      - {"kind": "text_delta", "text": "..."}
      - {
            "kind": "tool_call",
            "id": "toolu_...",
            "name": "...",
            "arguments": {...},
        }

    For parsers that expose `tool_call_pattern` and `tool_call_open_tag`,
    partial tool blocks at the end of the buffer are held back until they
    are complete. For other parsers, the helper falls back to treating
    the entire buffer as plain text and lets the parser interpret it
    best-effort.
    """

    def __init__(
        self,
        token_parser: "BaseTokenParser | None" = None,
        enable_tools: bool = True,
    ):
        self.enable_tools = enable_tools
        self._parser = token_parser
        self._buffer: str = ""

        # Optional parser hints to avoid emitting partial tool blocks.
        self._pattern: re.Pattern[str] | None = None
        self._open_tag: str | None = None

        if self.enable_tools and self._parser is not None:
            pattern_str = getattr(self._parser, "tool_call_pattern", None)
            open_tag = getattr(self._parser, "tool_call_open_tag", None)
            if pattern_str is not None and open_tag is not None:
                try:
                    self._pattern = re.compile(pattern_str, re.DOTALL)
                    self._open_tag = open_tag
                except re.error:
                    logger.exception(
                        "Failed to compile tool_call_pattern %r", pattern_str
                    )

    def feed(self, text_chunk: str) -> List[Dict[str, Any]]:
        """
        Consume a new piece of generated text and return a list of events.
        """
        events: List[Dict[str, Any]] = []
        if not text_chunk:
            return events

        logger.debug(
            "feed: received text_chunk_len=%d, text_chunk=%r",
            len(text_chunk),
            text_chunk[:200] if len(text_chunk) > 200 else text_chunk,
        )

        self._buffer += text_chunk
        logger.debug(
            "feed: buffer_len=%d after append, buffer_preview=%r",
            len(self._buffer),
            self._buffer[:200] if len(self._buffer) > 200 else self._buffer,
        )

        # Determine how much of the buffer is safe to parse now.
        safe_end = self._compute_safe_prefix_end()
        logger.debug(
            "feed: safe_end=%d (out of buffer_len=%d)",
            safe_end,
            len(self._buffer),
        )
        if safe_end <= 0:
            return events

        safe_text = self._buffer[:safe_end]
        self._buffer = self._buffer[safe_end:]
        logger.debug(
            "feed: processing safe_text_len=%d, remaining_buffer_len=%d",
            len(safe_text),
            len(self._buffer),
        )

        events.extend(self._process_text(safe_text))
        logger.debug(
            "feed: generated %d events",
            len(events),
        )

        return events

    def flush(self) -> List[Dict[str, Any]]:
        """
        Flush any remaining buffered text as final events.
        """
        if not self._buffer:
            logger.debug("flush: buffer is empty, returning empty events list")
            return []

        logger.debug(
            "flush: flushing buffer_len=%d, buffer=%r",
            len(self._buffer),
            self._buffer[:200] if len(self._buffer) > 200 else self._buffer,
        )
        events = self._process_text(self._buffer)
        self._buffer = ""
        logger.debug(
            "flush: generated %d events, cleared buffer",
            len(events),
        )
        return events

    def _compute_safe_prefix_end(self) -> int:
        """
        Return the index in `self._buffer` up to which text can be safely
        parsed without including an incomplete tool-call block.
        """
        if not (self.enable_tools and self._parser and self._pattern and self._open_tag):
            # No parser hints; treat everything as safe.
            logger.debug(
                "_compute_safe_prefix_end: no parser hints, "
                "returning entire buffer_len=%d",
                len(self._buffer),
            )
            return len(self._buffer)

        text = self._buffer
        matches = list(self._pattern.finditer(text))
        logger.debug(
            "_compute_safe_prefix_end: found %d complete tool_call matches",
            len(matches),
        )

        if not matches:
            # No complete tool blocks yet. If there is an opening tag, only
            # emit text before the first opening tag.
            idx_open = text.find(self._open_tag)
            if idx_open == -1:
                logger.debug(
                    "_compute_safe_prefix_end: no matches and no open_tag, "
                    "returning entire buffer_len=%d",
                    len(text),
                )
                return len(text)
            logger.debug(
                "_compute_safe_prefix_end: no matches but found open_tag at idx=%d, "
                "returning up to open_tag",
                idx_open,
            )
            return idx_open

        # We have at least one complete tool block.
        last_end = matches[-1].end()
        tail = text[last_end:]
        logger.debug(
            "_compute_safe_prefix_end: last_match_end=%d, tail_len=%d",
            last_end,
            len(tail),
        )
        if not tail:
            return len(text)

        idx_open_tail = tail.find(self._open_tag)
        if idx_open_tail == -1:
            # No new opening tag; everything is safe.
            logger.debug(
                "_compute_safe_prefix_end: no new open_tag in tail, "
                "returning entire buffer_len=%d",
                len(text),
            )
            return len(text)

        # Emit up to just before the first new opening tag in the tail.
        safe_end = last_end + idx_open_tail
        logger.debug(
            "_compute_safe_prefix_end: found new open_tag in tail at idx=%d, "
            "returning safe_end=%d",
            idx_open_tail,
            safe_end,
        )
        return safe_end

    def _process_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Run the token parser (if enabled) on the given text and map the
        resulting content items to streaming events.

        For tool calls, this now emits a sequence of events:
        - tool_start: Signals the beginning of a tool block
        - input_json_delta: Streams the tool input JSON in chunks
        - tool_stop: Signals the end of a tool block
        """
        logger.debug(
            "_process_text: processing text_len=%d, text_preview=%r",
            len(text),
            text[:200] if len(text) > 200 else text,
        )
        events: List[Dict[str, Any]] = []
        if not text:
            return events

        if not (self.enable_tools and self._parser):
            logger.debug("_process_text: tools disabled or no parser, emitting text_delta")
            events.append({"kind": "text_delta", "text": text})
            return events

        content_items, _ = self._parser.parse_tool_calls(text)
        logger.debug(
            "_process_text: parser returned %d content_items",
            len(content_items),
        )
        for item in content_items:
            if isinstance(item, OutputTextContentItem):
                if item.text:
                    logger.debug(
                        "_process_text: emitting text_delta, text_len=%d",
                        len(item.text),
                    )
                    events.append({"kind": "text_delta", "text": item.text})
            elif isinstance(item, OutputToolContentItem):
                # Emit tool_start event
                logger.debug(
                    "_process_text: emitting tool_start, id=%s, name=%s",
                    item.id,
                    item.name,
                )
                events.append({
                    "kind": "tool_start",
                    "id": item.id,
                    "name": item.name,
                })

                # Stream the tool input JSON in chunks (by complete field values)
                json_chunks = self._chunk_json_by_fields(item.input)
                logger.debug(
                    "_process_text: chunked JSON into %d chunks",
                    len(json_chunks),
                )
                for i, chunk in enumerate(json_chunks):
                    logger.debug(
                        "_process_text: emitting input_json_delta [%d/%d], chunk=%r",
                        i + 1,
                        len(json_chunks),
                        chunk[:100] if len(chunk) > 100 else chunk,
                    )
                    events.append({
                        "kind": "input_json_delta",
                        "partial_json": chunk,
                    })

                # Emit tool_stop event
                logger.debug("_process_text: emitting tool_stop")
                events.append({
                    "kind": "tool_stop",
                })

        logger.debug(
            "_process_text: returning %d events total",
            len(events),
        )
        return events

    def _chunk_json_by_fields(self, json_obj: dict[str, Any]) -> List[str]:
        """
        Chunk a JSON object into partial JSON strings, emitting complete field values.

        This provides a balance between streaming granularity and overhead.
        For simple objects, this might emit:
        - '{"location": '
        - '"San Francisco, CA"'
        - ', "unit": '
        - '"celsius"'
        - '}'

        Args:
            json_obj: The JSON object to chunk

        Returns:
            List of partial JSON string chunks
        """
        import json

        if not json_obj:
            return ['{}']

        chunks: List[str] = []
        chunks.append('{')

        items = list(json_obj.items())
        for i, (key, value) in enumerate(items):
            # Add the key
            chunks.append(f'"{key}": ')

            # Add the value (properly JSON-encoded)
            value_json = json.dumps(value)
            chunks.append(value_json)

            # Add comma if not the last item
            if i < len(items) - 1:
                chunks.append(', ')

        chunks.append('}')

        return chunks
