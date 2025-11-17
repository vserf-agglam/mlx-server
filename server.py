#!/usr/bin/env python3

import threading
import queue
import time
import curses
import sys
import io
from contextlib import redirect_stdout

from mlx_lm.utils import load
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.generate import BatchGenerator, BatchStats, wired_limit, generation_stream
from typing import List, Union, Generator, Optional, Dict
from dataclasses import dataclass
import mlx.core as mx

# Global queues for capturing different types of print statements
main_queue = queue.Queue()
inference_queue = queue.Queue()
io_queue = queue.Queue()

class CapturingPrint:
    """Custom print function that captures output to both console and categorized queues"""
    def __init__(self, original_print):
        self.original_print = original_print
    
    def __call__(self, *args, **kwargs):
        # Get the printed text
        text = ' '.join(str(arg) for arg in args)
        # Send to original print
        self.original_print(*args, **kwargs)
        
        # Categorize and send to appropriate queue
        if text.startswith("MAIN:"):
            main_queue.put(text)
        elif text.startswith("INFERENCE:"):
            inference_queue.put(text)
        elif text.startswith("INPUT QUEUE:") or text.startswith("OUTPUT:"):
            io_queue.put(text)
        else:
            # Default to main queue if no specific prefix
            main_queue.put(text)

# Replace the built-in print with our capturing version
original_print = print
print = CapturingPrint(original_print)


def stream_continuous_batch_generate(
    model,
    tokenizer,
    batch_uid: int,
    queue_prompts: queue.Queue,
    queue_map: queue.Queue,
    queue_results: queue.Queue,
    queue_stats: queue.Queue,
    verbose: bool = False,
    **kwargs,
) -> None:
    """
    Generate responses for the given batch of prompts.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (List[List[int]]): The input prompts.
       verbose (bool): If ``True``, print tokens and timing information.
          Default: ``False``.
       max_tokens (Union[int, List[int]): Maximum number of output tokens. This
          can be per request if a list is provided.
       kwargs: The remaining options get passed to :obj:`BatchGenerator`.
          See :obj:`BatchGenerator` for more details.
    """
    if verbose:
        print("INFERENCE: Thread started - beginning continuous batch generation")
    gen = BatchGenerator(model, stop_tokens=set(tokenizer.eos_token_ids)) 
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    # Get first prompt immediately for continuous batching
    first_item = queue_prompts.get_nowait()
    prompts = [first_item[0]]  # Single prompt 
    max_tokens = [first_item[1]]
    prompt_ids = first_item[2]
    num_samples = len(prompts)
    fin = 0
    fin_uids = []
    if verbose:
        print(f"INFERENCE: Starting processing 0/{num_samples} ...")
    
    with wired_limit(model, [generation_stream]):
        uids = gen.insert(prompts, max_tokens)
        queue_map.put_nowait({f"{batch_uid}_0":prompt_ids}) #put map for first prompt immediately
        detokenizers = {uid: tokenizer.detokenizer for uid in uids}  # Create a detokenizer for each uid
        while responses := gen.next():
            interim_texts = {f"{batch_uid}_{uid}": "" for uid in uids}  # Initialize interim texts for all uids
            for r in responses:
                if r.finish_reason != None:
                    fin += 1
                    fin_uids.append(r.uid)
                    detokenizers[r.uid].finalize()  # Finalize detokenizer to flush any remaining tokens
                    interim_texts[f"{batch_uid}_{r.uid}"] = detokenizers[r.uid].last_segment  # Update interim text at correct index
                    
                    if verbose:
                        print(f"INFERENCE: Finished processing {fin}/{num_samples} - Prompt id {r.uid + 1}")
                if r.finish_reason != "stop":
                    detokenizers[r.uid].add_token(r.token)
                    interim_texts[f"{batch_uid}_{r.uid}"] = detokenizers[r.uid].last_segment  # Update interim text at correct index

            queue_results.put_nowait([interim_texts, [f"{batch_uid}_{fin_uid}" for fin_uid in fin_uids]])

            # Add new prompts from the queue if available for continuous processing
            if queue_prompts is not None and not queue_prompts.empty():
                new_prompts = []
                new_max_tokens = []
                new_prompt_ids = [] 
                while not queue_prompts.empty():
                    new_queue_item = queue_prompts.get_nowait()
                    prompt_tokens, max_tokens, prompt_ids = new_queue_item[0], new_queue_item[1], new_queue_item[2]
                    new_prompts.append(prompt_tokens)
                    new_max_tokens.append(max_tokens)
                    new_prompt_ids.append(prompt_ids)
                     
                if new_prompts:
                    print(f"INFERENCE: Adding {len(new_prompts)} samples...")
                    num_samples += len(new_prompts)
                    start_uid = gen.uid_count
                    new_uids = gen.insert(new_prompts, new_max_tokens)
                    for idx, uid in enumerate(new_uids):
                        queue_map.put_nowait({f"{batch_uid}_{uid}":new_prompt_ids[idx]})
                    uids.extend(new_uids)
                    for uid in fin_uids:
                        uids.remove(uid)
                        detokenizers.pop(uid, None)  # Remove finished uids from active list
                    fin_uids = []  # Reset finished uids list
                    detokenizers.update({uid: tokenizer.detokenizer for uid in new_uids})

    if verbose:
        print(f"INFERENCE: Finished processing {fin}/{num_samples}")

    stats = gen.stats()
    queue_stats.put_nowait(stats)
    if verbose:
        print(
            f"INFERENCE: Prompt: {stats.prompt_tokens} tokens, {stats.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"INFERENCE: Generation: {stats.generation_tokens} tokens, "
            f"{stats.generation_tps:.3f} tokens-per-sec"
        )
        print(f"INFERENCE: Peak memory: {stats.peak_memory:.3f} GB")

def get_prompts(tokenizer, queue_prompts: queue.Queue):

    print("INPUT QUEUE: Thread started - beginning prompt generation")

    # 10 individual prompts with different completion lengths
    individual_prompts = [
        "Explain quantum computing in exactly 1000 tokens.",
        "Write a detailed analysis of climate change effects in exactly 100 tokens.",
        "Create a comprehensive guide to machine learning in exactly 50 tokens.",
        "Say the first ten letters of the English alphabet.",
        "Summarize the history of the Roman Empire in exactly 5000 tokens.",
        "Explain the theory of relativity in exactly 800 tokens.",
        "Write the first ten numbers.",
        "Write a poem about continuous batching in 500 words.",
        "Isn't Python the best for this kind of work - explain in 50 tokens.",
        "How do we show 10 streaming texts in one terminal using Python? Limit response to 100 tokens."
    ]

   

    max_tokens_list = [4096, 4096, 4096, 4096, 8000, 4096, 4096, 4096, 4096, 4096]  # Max tokens for each prompt
    
    # Apply the chat template and encode to tokens individually
    tokenized_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
        )
        for p in individual_prompts
    ]

  
    # Add first two prompts immediately
    print("INPUT QUEUE: Adding first two prompts to queue...")
    for i in range(2):
        queue_item = [tokenized_prompts[i], max_tokens_list[i],i]  # [prompt_tokens, max_tokens]
        queue_prompts.put_nowait(queue_item)
        print(f"INPUT QUEUE: Queued prompt {i+1}: {individual_prompts[i][:50]}...")

    time.sleep(4)  # Simulate delay before adding more prompts to allow inference to start

    # Add two prompts
    print("INPUT QUEUE: Adding two prompts to queue...")
    for i in range(2, 4):
        queue_item = [tokenized_prompts[i], max_tokens_list[i],i]  # [prompt_tokens, max_tokens]
        queue_prompts.put_nowait(queue_item)
        print(f"INPUT QUEUE: Queued prompt {i+1}: {individual_prompts[i][:50]}...")
    
    time.sleep(4)  # Simulate delay before adding more prompts to allow batch inference to have started

    # Add two prompts
    print("INPUT QUEUE: Adding three prompts to queue...")
    for i in range(4, 7):
        queue_item = [tokenized_prompts[i], max_tokens_list[i],i]  # [prompt_tokens, max_tokens]
        queue_prompts.put_nowait(queue_item)
        print(f"INPUT QUEUE: Queued prompt {i+1}: {individual_prompts[i][:50]}...")

    time.sleep(30)  # Simulate delay before adding batched prompts
    # Add batched prompts
    print("INPUT QUEUE: Adding last three prompts to queue...")
    for i in range(7, 10):
        queue_item = [tokenized_prompts[i], max_tokens_list[i],i]  # [prompt_tokens, max_tokens]
        queue_prompts.put_nowait(queue_item)
        print(f"INPUT QUEUE: Queued prompt {i+1}: {individual_prompts[i][:50]}...")

    time.sleep(30)  # Simulate delay before adding batched prompts
    print("INPUT QUEUE: All prompts queued - thread finishing")

def curses_display(stdscr, queue_map, queue_results, stop_event):
    curses.curs_set(0)  # Hide cursor
    stdscr.nodelay(1)   # Non-blocking input
    stdscr.timeout(100) # Refresh every 100ms
    
    # Initialize colors
    curses.start_color()
    curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)    # Red for headers
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Green for responses
    curses.init_pair(3, curses.COLOR_WHITE, curses.COLOR_BLACK)  # White for separators
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)   # Cyan for console
    curses.init_pair(5, curses.COLOR_YELLOW, curses.COLOR_BLACK) # Yellow for completion message
    
    # Get terminal dimensions
    height, width = stdscr.getmaxyx()
    
    # Reserve top 10 rows for console output
    console_height = 10
    display_start = console_height
    display_height = height - console_height
    
    # Initialize display data
    map = {0:"None"}
    results = {}
    log_messages = []
    max_log_messages = 100
    processing_complete = False
    
    # Original prompts for display
    original_prompts = [
        "Explain quantum computing in exactly 1000 tokens.",
        "Write a detailed analysis of climate change effects in exactly 100 tokens.",
        "Create a comprehensive guide to machine learning in exactly 50 tokens.",
        "Say the first ten letters of the English alphabet.",
        "Summarize the history of the Roman Empire in exactly 5000 tokens.",
        "Explain the theory of relativity in exactly 800 tokens.",
        "Write the first ten numbers.",
        "Write a poem about continuous batching in 500 words.",
        "Isn't Python the best for this kind of work - explain in 50 tokens.",
        "How do we show 10 streaming texts in one terminal using Python? Limit response to 100 tokens."
    ]
    
    # Add initial log message
    log_messages.append(" curses_display initialized - waiting for results...")
    
    while not stop_event.is_set() or processing_complete:
        # Process any new results
        results_updated = False
        while not queue_results.empty():
            while not queue_map.empty():
                new_map = queue_map.get_nowait()
                map.update(new_map)
                # Don't add "Updated map" messages to console - only show script prints
            
            interim_results = queue_results.get_nowait()
            interim_texts = interim_results[0]
            finished_uids = interim_results[1]
            
            for key, value in interim_texts.items():
                if(key in map):
                    if(map[key] in results):
                        results[map[key]] += value
                    else:
                        results.update({map[key]:value})
                    results_updated = True
            
            if finished_uids:
                # Don't add "Finished UIDs" to console - only show script prints
                pass
        
        # Check if processing is complete (stop_event set and no more results)
        if stop_event.is_set() and queue_results.empty() and queue_map.empty():
            processing_complete = True
            # Change to blocking mode to wait for user key press
            stdscr.nodelay(0)
            stdscr.timeout(-1)  # Block indefinitely for user input
        
        # Process any new print statements from the categorized queues
        # Collect new messages without clearing the queues
        new_main_messages = []
        new_inference_messages = []
        new_io_messages = []
        
        # Get new messages from each queue
        while not main_queue.empty():
            try:
                new_main_messages.append(main_queue.get_nowait())
            except queue.Empty:
                break
                
        while not inference_queue.empty():
            try:
                new_inference_messages.append(inference_queue.get_nowait())
            except queue.Empty:
                break
                
        while not io_queue.empty():
            try:
                new_io_messages.append(io_queue.get_nowait())
            except queue.Empty:
                break
        
        # Add new messages to stored lists
        log_messages.extend(new_main_messages + new_inference_messages + new_io_messages)
        if len(log_messages) > max_log_messages:
            log_messages = log_messages[-max_log_messages:]  # Keep only recent messages
        
        # Clear screen
        stdscr.clear()
        
        # Calculate column widths for 3-column layout
        col_width = (width - 8) // 3  # 3 columns with spacing
        if col_width < 20:  # Minimum column width
            col_width = 20
        
        # Column positions
        main_col_x = 1
        inference_col_x = col_width + 3
        io_col_x = (col_width + 3) * 2
        
        # Display 3-column log area at top
        try:
            # MAIN column header
            stdscr.addstr(0, main_col_x, "MAIN", curses.color_pair(1) | curses.A_BOLD)
            
            # INFERENCE column header  
            stdscr.addstr(0, inference_col_x, "INFERENCE", curses.color_pair(1) | curses.A_BOLD)
            
            # I/O column header
            stdscr.addstr(0, io_col_x, "INPUT/OUTPUT", curses.color_pair(1) | curses.A_BOLD)
        except curses.error:
            pass  # Ignore display errors
        
        # Filter messages by category for display
        main_messages = [msg for msg in log_messages if msg.startswith("MAIN:")]
        inference_messages = [msg for msg in log_messages if msg.startswith("INFERENCE:")]
        io_messages = [msg for msg in log_messages if msg.startswith("INPUT QUEUE:") or msg.startswith("OUTPUT:")]
        
        # Display MAIN messages
        for i, msg in enumerate(main_messages[-(console_height-2):]):
            if i + 2 < console_height:
                try:
                    truncated_msg = msg[:col_width-2] + ".." if len(msg) > col_width-2 else msg
                    stdscr.addstr(i + 2, main_col_x + 1, truncated_msg, curses.color_pair(4))
                except curses.error:
                    pass
        
        # Display INFERENCE messages
        for i, msg in enumerate(inference_messages[-(console_height-2):]):
            if i + 2 < console_height:
                try:
                    truncated_msg = msg[:col_width-2] + ".." if len(msg) > col_width-2 else msg
                    stdscr.addstr(i + 2, inference_col_x + 1, truncated_msg, curses.color_pair(4))
                except curses.error:
                    pass
        
        # Display I/O messages
        for i, msg in enumerate(io_messages[-(console_height-2):]):
            if i + 2 < console_height:
                try:
                    truncated_msg = msg[:col_width-2] + ".." if len(msg) > col_width-2 else msg
                    stdscr.addstr(i + 2, io_col_x + 1, truncated_msg, curses.color_pair(4))
                except curses.error:
                    pass
        
        # Add separator line below 3-column console in white
        try:
            stdscr.hline(console_height, 0, '=', width, curses.color_pair(3))
        except curses.error:
            pass  # Ignore display errors
        
        # Display results - first 5 as columns in first row, next 5 as columns in second row
        if results and display_height > 0:
            prompt_ids = sorted(results.keys())
            
            # Debug: Show how many results we have
            if len(results) != 10:
                debug_msg = f"DEBUG: Found {len(results)} results: {list(results.keys())}"
                try:
                    stdscr.addstr(console_height + 1, 1, debug_msg, curses.color_pair(5))
                except curses.error:
                    pass
            
            # Calculate column width (more compact layout with minimal spacing)
            col_width = (width - 4) // 5  # Minimal spacing between columns
            if col_width < 8:  # Smaller minimum column width
                col_width = 8
            
            # Calculate available height for each row (excluding headers and separators)
            row_height = (display_height - 3) // 2  # Split available height between 2 rows
            if row_height < 2:  # Minimum height per row
                row_height = 2
            
            # Adjust positions to account for console at top
            first_row_pos = display_start + 1
            second_row_start = display_start + row_height + 3
            
            # Ensure we have all 10 prompt IDs (0-9)
            all_prompt_ids = list(range(10))
            
            # First 5 results - first row (prompts 0-4)
            for prompt_id in range(5):
                x_pos = prompt_id * col_width + 1  # Tight spacing
                text = results.get(prompt_id, "")  # Get response or empty string
                
                # Compact prompt header (shortened) - show prompt ID + 1
                if prompt_id < len(original_prompts):
                    prompt_text = original_prompts[prompt_id][:col_width-4] + ".." if len(original_prompts[prompt_id]) > col_width-2 else original_prompts[prompt_id]
                    header = f"{prompt_id + 1}.{prompt_text}"
                else:
                    header = f"P{prompt_id + 1}:"
                
                try:
                    # Display header at top of row in red
                    stdscr.addstr(first_row_pos, x_pos, header[:col_width], curses.color_pair(1))
                    
                    # Display response text with better wrapping in green (showing last words)
                    if len(text) > 0:
                        lines = []
                        # Smart word wrapping
                        words = text.split()
                        current_line = ""
                        for word in words:
                            if len(current_line + word) <= col_width:
                                current_line += word + " "
                            else:
                                if current_line:
                                    lines.append(current_line.rstrip())
                                current_line = word + " "
                        if current_line:
                            lines.append(current_line.rstrip())
                        
                        # Show the last lines (most recent content) instead of first lines
                        if len(lines) > row_height:
                            # Display the last row_height lines to show newest content
                            recent_lines = lines[-row_height:]
                            for line_idx, line in enumerate(recent_lines):
                                stdscr.addstr(first_row_pos + 1 + line_idx, x_pos, line[:col_width], curses.color_pair(2))
                        else:
                            # Display all lines if fewer than row_height
                            for line_idx, line in enumerate(lines):
                                stdscr.addstr(first_row_pos + 1 + line_idx, x_pos, line[:col_width], curses.color_pair(2))
                except curses.error:
                    pass  # Ignore display errors
            
            # Next 5 results - second row (prompts 5-9)
            if second_row_start < height:
                for prompt_id in range(5, 10):
                    x_pos = (prompt_id - 5) * col_width + 1  # Tight spacing
                    text = results.get(prompt_id, "")  # Get response or empty string
                    
                    # Compact prompt header (shortened) - show prompt ID + 1
                    if prompt_id < len(original_prompts):
                        prompt_text = original_prompts[prompt_id][:col_width-4] + ".." if len(original_prompts[prompt_id]) > col_width-2 else original_prompts[prompt_id]
                        header = f"{prompt_id + 1}.{prompt_text}"
                    else:
                        header = f"P{prompt_id + 1}:"
                    
                    try:
                        # Display header at top of second row in red
                        stdscr.addstr(second_row_start, x_pos, header[:col_width], curses.color_pair(1))
                        
                        # Display response text with better wrapping in green (showing last words)
                        if len(text) > 0:
                            lines = []
                            # Smart word wrapping
                            words = text.split()
                            current_line = ""
                            for word in words:
                                if len(current_line + word) <= col_width:
                                    current_line += word + " "
                                else:
                                    if current_line:
                                        lines.append(current_line.rstrip())
                                    current_line = word + " "
                            if current_line:
                                lines.append(current_line.rstrip())
                            
                            # Show the last lines (most recent content) instead of first lines
                            if len(lines) > row_height:
                                # Display the last row_height lines to show newest content
                                recent_lines = lines[-row_height:]
                                for line_idx, line in enumerate(recent_lines):
                                    if second_row_start + 1 + line_idx < height:
                                        stdscr.addstr(second_row_start + 1 + line_idx, x_pos, line[:col_width], curses.color_pair(2))
                            else:
                                # Display all lines if fewer than row_height
                                for line_idx, line in enumerate(lines):
                                    if second_row_start + 1 + line_idx < height:
                                        stdscr.addstr(second_row_start + 1 + line_idx, x_pos, line[:col_width], curses.color_pair(2))
                    except curses.error:
                        pass  # Ignore display errors
        
        # Draw separator between rows in white
        if display_height > 0:
            row_separator = display_start + row_height + 2
            if row_separator < height:
                try:
                    stdscr.hline(row_separator, 0, '-', width, curses.color_pair(3))
                except curses.error:
                    pass  # Ignore display errors
        
        # Display completion message if processing is complete
        if processing_complete:
            try:
                completion_msg = "PROCESSING COMPLETE! Press any key to exit..."
                msg_x = (width - len(completion_msg)) // 2
                msg_y = height - 2
                stdscr.addstr(msg_y, msg_x, completion_msg, curses.color_pair(5) | curses.A_BOLD)
            except curses.error:
                pass  # Ignore display errors
        
        # Refresh display
        stdscr.refresh()
        
        # Handle user input when processing is complete
        if processing_complete:
            try:
                # Wait for any key press
                key = stdscr.getch()
                if key != -1:  # Any key pressed
                    break
            except curses.error:
                pass  # Ignore display errors
        else:
            # Small delay to prevent excessive CPU usage during processing
            time.sleep(0.1)
    
    # Cleanup before exiting - wait for final key press
    stdscr.clear()
    stdscr.addstr(0, 0, "Curses display complete. Press any key to exit...", curses.color_pair(5) | curses.A_BOLD)
    stdscr.refresh()
    
    # Wait for final key press before exiting
    stdscr.nodelay(0)
    stdscr.timeout(-1)
    try:
        stdscr.getch()
    except curses.error:
        pass

def results_stream(queue_map, queue_results, stop_event = threading.Event()):
    time.sleep(4) #to give enough time for inference to start streaming some results
    
    try:
        # Initialize curses and run the display
        curses.wrapper(curses_display, queue_map, queue_results, stop_event)
    except Exception as e:
        # Fallback to console output if curses fails
        print(f"Curses display failed: {e}")
        print("Falling back to console output...")
        
        map = {0:"None"}
        results = {}
        
        while not stop_event.is_set(): 
            while not queue_results.empty():
                while not queue_map.empty():
                    new_map = queue_map.get_nowait()
                    map.update(new_map)
                interim_results = queue_results.get_nowait()
                interim_texts = interim_results[0]
                finished_uids = interim_results[1]

                for key, value in interim_texts.items():
                    if(key in map):
                        if(map[key] in results):
                            results[map[key]] += value
                        else:
                            results.update({map[key]:value})
            
            time.sleep(0.2)
        
        # Print final results
        for key, value in results.items():
            print(f"OUTPUT: Response for prompt {key} was: {value}")
    
    print("OUTPUT: Output thread quitting...")

     
    
if __name__ == "__main__":
    
    print("MAIN: Loading model...")
    model, tokenizer = load("/Users/vahit/.lmstudio/models/az13770129/Qwen3-Coder-REAP-25B-A3B-mlx-4Bit")
    stop_event = threading.Event()

    queue_prompts = queue.Queue(maxsize=0)
    queue_results = queue.Queue(maxsize=0)
    queue_map = queue.Queue(maxsize=0)
    queue_stats = queue.Queue(maxsize=0)
    batch_uid = -1

    
    # Create and start the Queue thread
    print("MAIN: Starting IO queues threads")
    i_thread = threading.Thread(target=get_prompts, args=(tokenizer,queue_prompts))
    i_thread.start()

    o_thread = threading.Thread(
        target=results_stream,
        args=(queue_map, queue_results,stop_event)
    )
    o_thread.start()
    # Main processing loop - always monitoring queue
    idle_start_time = time.time()
    idle_timeout = 10.0  # Shutdown after 60 seconds of no activity
    
    print(f"MAIN: Starting main loop - monitoring queue for requests (timeout: {idle_timeout}s)")
    
    while True:
        # Check if queue has items immediately
        if not queue_prompts.empty():
            batch_uid += 1
            print(f"\nMAIN: Starting BATCH {batch_uid} processing cycle...")
            batch_start_time = time.time()
            # Create and start the inference thread for this batch
            inference_thread = threading.Thread(
                target=stream_continuous_batch_generate,
                args=(model, tokenizer, batch_uid, queue_prompts, queue_map, queue_results,queue_stats, True)
            )
            inference_thread.start()
            inference_thread.join()
            total_time = time.time() - batch_start_time
            print(f"\nMAIN: BATCH {batch_uid} COMPLETED in {total_time:.1f}s!")
            # Extract batch stats if available

            batch_stats = queue_stats.get_nowait()
            print(f"MAIN: BATCH {batch_uid} STATS:")
            print(f"MAIN:   Prompt tokens: {batch_stats.prompt_tokens}")
            print(f"MAIN:   Generation tokens: {batch_stats.generation_tokens}")
            print(f"MAIN:   Prompt TPS: {batch_stats.prompt_tps:.3f}")
            print(f"MAIN:   Generation TPS: {batch_stats.generation_tps:.3f}")
            print(f"MAIN:   Peak memory: {batch_stats.peak_memory:.3f} GB")
      
            print("="*80) 

            # Reset idle timer when batch completes (not when we get requests)
            idle_start_time = time.time()
            print(f"MAIN: BATCH {batch_uid} complete - returning to queue monitoring...")
            continue  # Go back to top of loop immediately
        
        # Check for idle timeout
        current_idle_time = time.time() - idle_start_time
        if current_idle_time > idle_timeout:
            print(f"MAIN: No requests for {idle_timeout}s - checking shutdown conditions...")
            
            # Check if Input thread is still alive and if queue_prompts is truly empty
            if not i_thread.is_alive() and queue_prompts.empty():
                print(f"MAIN: Input thread finished and queue empty for {idle_timeout}s - shutting down")
                break
            elif i_thread.is_alive():
                print(f"MAIN: Input thread still running but queue empty for {idle_timeout}s - continuing...")
                # Reset idle timer to give Input thread more time
                idle_start_time = time.time()
            else:
                print(f"MAIN: Queue might populate soon, continuing...")
        
        # Brief sleep to prevent busy waiting while monitoring queue
        time.sleep(0.1)


    # Wait for Input thread to finish
    print("MAIN: Waiting for Input thread to complete...")
    if i_thread.is_alive():
        i_thread.join(timeout=5.0)
        if i_thread.is_alive():
            print("MAIN: Input thread still running after timeout")
        else:
            print("MAIN: Input thread completed cleanly")
    else:
        print("MAIN: Input thread already completed")
    
    stop_event.set() # Signal the Output thread to stop
    o_thread.join() # Wait for the Output thread to finish

    # Final summary
    print(f"\nMAIN: PROCESSING COMPLETE!")
    print(f"MAIN:   Total batches processed: {batch_uid+1}")
    print("MAIN: Cleanup complete - true continuous batching demonstrated!")