import logging
import queue
import random
import time
from typing import LiteralString, Literal

from mlx_lm.generate import BatchGenerator, wired_limit, generation_stream
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load

from api.types import ToolType, MessagesBody

logger = logging.getLogger(__name__)

class Server:


    def __init__(self, model_name: str, token_parser_name: Literal["qwen3_moe"]):
        self.loaded = False
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.queue_prompts = queue.Queue(maxsize=0)
        self.queue_map = queue.Queue(maxsize=0)
        self.token_parser_name = token_parser_name
        self.queue_results = queue.Queue(maxsize=0)



    def load(self):
        before_loading_time = time.time()
        model, tokenizer = load(self.model_name)
        after_loading_time = time.time()
        loading_time = after_loading_time - before_loading_time
        logger.debug(f"Loading time: {loading_time}")
        self.model = model
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.loaded = True

    def unload(self):
        self.model = None
        self.tokenizer = None
        self.loaded = False

    def chat_template(self, messages_body: MessagesBody, **kwargs):
        """Apply chat template with OpenAI-compatible format"""
        openai_messages = messages_body.get_openai_compatible_messages()
        openai_tools = messages_body.get_openai_compatible_tools()

        return self.tokenizer.apply_chat_template(
            openai_messages,
            tools=openai_tools,
            add_generation_prompt=True,
            **kwargs
        )
    def add_to_batch(self, prompt, max_tokens):
        prompt_id = random.randint(1, 999999999)
        self.queue_prompts.put_nowait(
            (prompt, max_tokens, prompt_id)
        )

        
        detokonizer = self.tokenizer.detokenizer
        while not self.queue_results.empty():
            while not self.queue_map.empty():




    def stream_continue_batch_generate(self, batch_uid: int):
        first_item = self.queue_prompts.get_nowait()
        prompts = [first_item[0]]  # Single prompt
        max_tokens = [first_item[1]]
        prompt_ids = first_item[2]
        gen = BatchGenerator(self.model, stop_tokens=set(self.tokenizer.eos_token_ids))
        finished_count = 0
        finished_uids = []
        with wired_limit(self.model, [generation_stream]):
            uids = gen.insert(prompts, max_tokens)
            self.queue_map.put_nowait({f"batch_{uids}": prompt_ids})
            while responses := gen.next():
                interim_tokens = []  # Initialize texts for all active uids
                for r in responses:
                    if r.finish_reason is not None:
                        finished_count += 1
                        finished_uids.append(r.uid)
                        interim_tokens.append(r.token)
                    if r.finish_reason != "stop":
                        interim_tokens.append(r.token)

                self.queue_results.put_nowait([interim_tokens, [f"{batch_uid}_{fin_uid}" for fin_uid in finished_uids]])
                if self.queue_prompts is not None and not self.queue_prompts.empty():
                    new_prompts = []
                    new_max_tokens = []
                    new_prompt_ids = []
                    while not self.queue_prompts.empty():
                        new_queue_item = self.queue_prompts.get_nowait()
                        prompt_tokens, max_tokens, prompt_ids = new_queue_item[0], new_queue_item[1], new_queue_item[2]
                        new_prompts.append(prompt_tokens)
                        new_max_tokens.append(max_tokens)
                        new_prompt_ids.append(prompt_ids)

                    if new_prompts:
                        print(f"INFERENCE: Adding {len(new_prompts)} samples...")
                        new_uids = gen.insert(new_prompts, new_max_tokens)
                        for idx, uid in enumerate(new_uids):
                            self.queue_map.put_nowait({f"{batch_uid}_{uid}": new_prompt_ids[idx]})
                        uids.extend(new_uids)
                        for uid in fin_uids:
                            uids.remove(uid)
                        fin_uids = []  # Reset finished uids list






