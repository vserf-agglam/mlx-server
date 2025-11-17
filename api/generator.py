import logging
import queue

from mlx_lm.generate import BatchGenerator, BatchStats, wired_limit, generation_stream
from mlx_lm.tokenizer_utils import TokenizerWrapper


def stream_continuous_batches(
    model,
    tokenizer,
    batch_uid: int,
    queue_prompts: queue.Queue,
    queue_map: queue.Queue,
    queue_results: queue.Queue,
    queue_stats: queue.Queue,
    verbose: bool = False
    ):
    """
    Generate responses for continuous batches of prompts.
    
    This function implements the same logic as stream_continuous_batch_generate
    from server.py but adapted for the API server use case.
    """
    
    if verbose:
        print("INFERENCE: Thread started - beginning continuous batch generation")
    
    gen = BatchGenerator(model, stop_tokens=set(tokenizer.eos_token_ids))
    
    # Ensure tokenizer is wrapped
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    
    # Get first prompt to start the batch
    first_item = queue_prompts.get_nowait()
    prompts = [first_item[0]]  # Single prompt to start
    max_tokens = [first_item[1]]
    prompt_ids = first_item[2]
    
    num_samples = len(prompts)
    fin = 0
    fin_uids = []
    
    if verbose:
        print(f"INFERENCE: Starting processing 0/{num_samples} ...")
        
    with wired_limit(model, [generation_stream]):
        uids = gen.insert(prompts, max_tokens)
        queue_map.put_nowait({f"{batch_uid}_0": prompt_ids})  # Map first prompt immediately
        
        detokenizers = {uid: tokenizer.detokenizer for uid in uids}  # Create detokenizer for each uid
        
        while responses := gen.next():
            interim_texts = {f"{batch_uid}_{uid}": "" for uid in uids}  # Initialize texts for all active uids
            
            for r in responses:
                if r.finish_reason is not None:
                    fin += 1
                    fin_uids.append(r.uid)
                    detokenizers[r.uid].finalize()  # Finalize to flush remaining tokens
                    interim_texts[f"{batch_uid}_{r.uid}"] = detokenizers[r.uid].last_segment
                    
                    if verbose:
                        print(f"INFERENCE: Finished processing {fin}/{num_samples} - Prompt id {r.uid + 1}")
                
                if r.finish_reason != "stop":
                    detokenizers[r.uid].add_token(r.token)
                    interim_texts[f"{batch_uid}_{r.uid}"] = detokenizers[r.uid].last_segment
            
            # Send results with finished UIDs
            queue_results.put_nowait([interim_texts, [f"{batch_uid}_{fin_uid}" for fin_uid in fin_uids]])
            
            # Check for new prompts to add to the batch
            if queue_prompts is not None and not queue_prompts.empty():
                new_prompts = []
                new_max_tokens = []
                new_prompt_ids = []
                
                while not queue_prompts.empty():
                    try:
                        new_queue_item = queue_prompts.get_nowait()
                        prompt_tokens, max_tokens_item, prompt_id = new_queue_item[0], new_queue_item[1], new_queue_item[2]
                        new_prompts.append(prompt_tokens)
                        new_max_tokens.append(max_tokens_item)
                        new_prompt_ids.append(prompt_id)
                    except queue.Empty:
                        break
                
                if new_prompts:
                    if verbose:
                        print(f"INFERENCE: Adding {len(new_prompts)} samples...")
                    
                    num_samples += len(new_prompts)
                    new_uids = gen.insert(new_prompts, new_max_tokens)
                    
                    # Map new UIDs to prompt IDs
                    for idx, uid in enumerate(new_uids):
                        queue_map.put_nowait({f"{batch_uid}_{uid}": new_prompt_ids[idx]})
                    
                    uids.extend(new_uids)
                    
                    # Remove finished UIDs
                    for uid in fin_uids:
                        if uid in uids:
                            uids.remove(uid)
                        detokenizers.pop(uid, None)
                    
                    fin_uids = []  # Reset finished list
                    
                    # Add detokenizers for new UIDs
                    detokenizers.update({uid: tokenizer.detokenizer for uid in new_uids})
    
    if verbose:
        print(f"INFERENCE: Finished processing {fin}/{num_samples}")
    
    # Send batch statistics
    stats = gen.stats()
    queue_stats.put_nowait(stats)
    
    if verbose:
        print(f"INFERENCE: Prompt: {stats.prompt_tokens} tokens, {stats.prompt_tps:.3f} tokens-per-sec")
        print(f"INFERENCE: Generation: {stats.generation_tokens} tokens, {stats.generation_tps:.3f} tokens-per-sec")
        print(f"INFERENCE: Peak memory: {stats.peak_memory:.3f} GB")