# examples/demo.py

import sys

sys.path.append('..')

from src.api.transformer import VMPTransformer, transform_assembly_to_vmp


# from vmp_transformer.src.api.transformer import VMPTransformer, transform_assembly_to_vmp

def example_simple_arithmetic():
    """Example: Simple arithmetic operations"""
    print("=== Example 1: Simple Arithmetic ===")

    asm_code = {
        "tools": "[{\"name\":\"retrieve_data\",\"description\":\"从汇编源码中指定的区域内读取数据元素，假设适当的数据类型，并且原始地址已经被如D1、D2等标签替换。\",\"parameters\":{\"type\":\"object\",\"properties\":{\"label\":{\"type\":\"string\",\"description\":\"准备好的汇编语言代码中代表数据存放点的标识符（例如，D1, D2）。\"},\"type\":{\"type\":\"string\",\"description\":\"In terms of the labeled data type, one can anticipate it to be a primary type (such as i8) or an array type (like i8[8]). Primary types are limited to: i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, byte, word, dword, qword, and string.\"}},\"required\":[\"label\",\"type\"],\"additionalProperties\":false}}]",
        "messages": [{"role": "user",
                      "content": "Demonstrate the C code that generates the assembly output below:\n\n<mv88fx_snd_readl>:\n  endbr64\n  sub    $0x8,%rsp\n  lea    -0xf000000(%rsi),%edi\n  call   <readl@plt>\n  add    $0x8,%rsp\n  ret"},
                     {"role": "assistant",
                      "content": "unsigned int mv88fx_snd_readl(struct mv88fx_snd_chip *chip, unsigned int offs) {\n  unsigned int val = readl((0xf1000000 + offs));\n  return (val);\n}"}]}
    asm_code_content = asm_code["messages"][0]["content"]
    # print(asm_code_content)
    assembly_code = """
      push   %rbp
      leave
      ret"""
    assembly_code = asm_code_content
    # 使用mov    %rsp,%rbp会报错，最好 mov ecx, 5 ?

    # Transform to VMP
    result = transform_assembly_to_vmp(assembly_code, debug=True)

    if 'error' not in result:
        print("Original Assembly:")
        print(assembly_code)
        print("\nVMP Protected Assembly:")
        print(result['vmp_assembly'])
        print(f"\nBytecode size: {result['metadata']['bytecode_size']} bytes")
        print(f"Variables: {result['metadata']['variables']}")
    else:
        print(f"Error: {result['error']}")


def example_loop():
    """Example: Loop with comparison"""
    print("\n=== Example 2: Loop ===")

    assembly_code = """
    section .text
    loop_start:
        mov ecx, 5        ; Counter
    loop_body:
        dec ecx           ; Decrement counter
        cmp ecx, 0        ; Compare with 0
        jne loop_body     ; Jump if not equal
        ret
    """

    result = transform_assembly_to_vmp(assembly_code)

    if 'error' not in result:
        print("Original Assembly:")
        print(assembly_code)
        print("\nVMP Protected Assembly:")
        print(result['vmp_assembly'])
    else:
        print(f"Error: {result['error']}")


def example_memory_operations():
    """Example: Memory operations"""
    print("\n=== Example 3: Memory Operations ===")

    assembly_code = """
    section .data
        buffer db 10, 20, 30, 40

    section .text
    main:
        mov eax, buffer   ; Load buffer address
        mov ebx, [eax]    ; Load first byte
        add eax, 1        ; Next byte
        mov ecx, [eax]    ; Load second byte
        add edx, ebx, ecx ; Add values
        ret edx
    """

    transformer = VMPTransformer(debug=False)
    result = transformer.transform_assembly(assembly_code)

    if 'error' not in result:
        print("Original Assembly:")
        print(assembly_code)
        print("\nVMP Protected Assembly (first 10 lines):")
        lines = result['vmp_assembly'].split('\n')
        for line in lines:
            print(line)
        # if len(lines) > 10:
        #     print("...")
    else:
        print(f"Error: {result['error']}")


def example_batch_transform():
    """Example: Batch transformation"""
    print("\n=== Example 4: Batch Transformation ===")

    files = [
        ("add.asm", "mov eax, 5\nadd eax, 3\nret eax"),
        ("mul.asm", "mov eax, 4\nmov ebx, 6\nmul ecx, eax, ebx\nret ecx"),
        ("xor.asm", "mov eax, 0xFF\nxor eax, 0xAA\nret eax")
    ]

    transformer = VMPTransformer()
    results = transformer.batch_transform(files)

    for filename, result in results.items():
        print(f"\n{filename}:")
        if 'error' not in result:
            print(f"  Bytecode size: {result['metadata']['bytecode_size']} bytes")
            print(f"  Variables: {result['metadata']['variables']}")
        else:
            print(f"  Error: {result['error']}")


def example_advanced_vmp():
    """Example: Advanced VMP with analysis"""
    print("\n=== Example 5: Advanced VMP with Analysis ===")

    assembly_code = """
    section .text
    factorial:
        mov eax, 5        ; n = 5
        mov ebx, 1        ; result = 1
    fact_loop:
        cmp eax, 1        ; if n <= 1
        jle fact_done     ; goto done
        mul ebx, ebx, eax ; result *= n
        dec eax           ; n--
        jmp fact_loop     ; continue loop
    fact_done:
        ret ebx           ; return result
    """

    transformer = VMPTransformer(debug=True)
    result = transformer.transform_assembly(assembly_code)

    if 'error' not in result:
        print("Analyzing VMP bytecode...")

        # Analyze the generated bytecode
        analysis = transformer.analyze_vmp_code(result['bytecode'])

        if analysis['success']:
            print("\nExecution trace:")
            for trace_line in analysis['trace']:  # First 10 trace lines
                print(f"  {trace_line}")
        else:
            print(f"Analysis failed: {analysis['error']}")
    else:
        print(f"Error: {result['error']}")


import json
import os
from datetime import datetime
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
import queue
import threading
import time


def process_single_entry(entry_data, line_num):
    """Process a single entry and return the result"""
    try:
        # Parse JSON
        asm_code = json.loads(entry_data)

        # Extract assembly code
        if 'messages' in asm_code and len(asm_code['messages']) > 0:
            asm_code_content = asm_code['messages'][0]['content']

            # Extract function name if present
            func_name = "unknown"
            if '<' in asm_code_content and '>:' in asm_code_content:
                func_name = asm_code_content.split('<')[1].split('>:')[0]

            # Transform to VMP
            result = transform_assembly_to_vmp(asm_code_content, debug=False)

            if 'error' not in result:
                return {
                    'line': line_num,
                    'function': func_name,
                    'status': 'success',
                    'bytecode_size': result['metadata']['bytecode_size'],
                    'variables': len(result['metadata']['variables'])
                }
            else:
                return {
                    'line': line_num,
                    'function': func_name,
                    'status': 'failed',
                    'error': result['error']
                }
        else:
            return {
                'line': line_num,
                'status': 'failed',
                'error': 'Invalid JSON structure - missing messages'
            }

    except json.JSONDecodeError as e:
        return {
            'line': line_num,
            'status': 'failed',
            'error': f'JSON parsing error: {e}'
        }
    except Exception as e:
        return {
            'line': line_num,
            'status': 'failed',
            'error': f'Unexpected error: {e}'
        }


def process_chunk(chunk_data):
    """Process a chunk of lines"""
    results = []
    for line_num, line in chunk_data:
        if line.strip():
            result = process_single_entry(line, line_num)
            results.append(result)
    return results


def progress_monitor(progress_queue, total_lines):
    """Monitor and display progress"""
    processed = 0
    start_time = time.time()

    while processed < total_lines:
        try:
            increment = progress_queue.get(timeout=1)
            if increment is None:  # Poison pill
                break
            processed += increment

            # Calculate and display progress
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total_lines - processed) / rate if rate > 0 else 0

            print(f"\rProgress: {processed}/{total_lines} ({processed / total_lines * 100:.1f}%) "
                  f"| Rate: {rate:.1f} entries/sec | ETA: {eta:.1f}s", end='', flush=True)

        except queue.Empty:
            continue

    print()  # New line after progress


def evaluation_mibench():
    """Evaluate all entries in mibench-asm.jsonl using parallel processing"""
    print("=== MiBench ASM Evaluation (Parallel) ===")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Configuration
    num_processes = 16
    chunk_size = 1000  # Lines per chunk

    # File path
    jsonl_file = "/data/jiacheng/dylan/l4av/l4av/vmp_transformer/examples/mibench-asm.jsonl"

    # Check if file exists
    if not os.path.exists(jsonl_file):
        print(f"Error: File {jsonl_file} not found!")
        return

    print(f"Using {num_processes} processes")
    print(f"Processing file: {jsonl_file}")
    print("-" * 80)

    # First, count total lines for progress tracking
    print("Counting total lines...")
    total_lines = 0
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    print(f"Total lines to process: {total_lines}")

    # Prepare chunks
    print("Preparing data chunks...")
    chunks = []
    current_chunk = []

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            current_chunk.append((line_num, line))

            if len(current_chunk) >= chunk_size:
                chunks.append(current_chunk)
                current_chunk = []

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)

    print(f"Created {len(chunks)} chunks")

    # Setup progress monitoring
    manager = Manager()
    progress_queue = manager.Queue()

    # Start progress monitor thread
    progress_thread = threading.Thread(
        target=progress_monitor,
        args=(progress_queue, total_lines)
    )
    progress_thread.start()

    # Process chunks in parallel
    print("Starting parallel processing...")
    all_results = []

    with Pool(processes=num_processes) as pool:
        # Process chunks and collect results
        chunk_results = pool.map(process_chunk, chunks)

        # Flatten results and update progress
        for chunk_result in chunk_results:
            all_results.extend(chunk_result)
            progress_queue.put(len(chunk_result))

    # Stop progress monitor
    progress_queue.put(None)
    progress_thread.join()

    # Calculate statistics
    total_count = len(all_results)
    success_count = sum(1 for r in all_results if r['status'] == 'success')
    failure_count = total_count - success_count

    # Categorize errors
    errors = {}
    for result in all_results:
        if result['status'] == 'failed' and 'error' in result:
            error_key = result['error'].split(':')[0] if ':' in result['error'] else result['error']
            errors[error_key] = errors.get(error_key, 0) + 1

    # Calculate statistics for successful transformations
    if success_count > 0:
        successful_results = [r for r in all_results if r['status'] == 'success']
        avg_bytecode_size = sum(r['bytecode_size'] for r in successful_results) / len(successful_results)
        avg_variables = sum(r['variables'] for r in successful_results) / len(successful_results)
    else:
        avg_bytecode_size = 0
        avg_variables = 0

    # Final summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total entries processed: {total_count}")
    print(f"Successful transformations: {success_count} ({success_count / total_count * 100:.1f}%)")
    print(f"Failed transformations: {failure_count} ({failure_count / total_count * 100:.1f}%)")

    if success_count > 0:
        print(f"\nSuccessful transformation statistics:")
        print(f"  Average bytecode size: {avg_bytecode_size:.1f} bytes")
        print(f"  Average variables: {avg_variables:.1f}")

    if errors:
        print("\nError breakdown (top 10):")
        sorted_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)[:10]
        for error_type, count in sorted_errors:
            print(f"  {error_type}: {count}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more error types")

    # Save detailed results to file
    output_file = f"mibench_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Sort results by line number for better organization
    all_results.sort(key=lambda x: x['line'])

    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total': total_count,
                'success': success_count,
                'failed': failure_count,
                'success_rate': f"{success_count / total_count * 100:.1f}%",
                'avg_bytecode_size': avg_bytecode_size,
                'avg_variables': avg_variables,
                'processing_time': f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            },
            'errors': errors,
            'details': all_results[:1000]  # Save first 1000 detailed results to avoid huge files
        }, f, indent=2)

    print(f"Detailed results saved to: {output_file}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Run all examples"""
    print("VMP Transformer Demo")
    print("===================\n")

    # example_simple_arithmetic()
    # example_loop()
    # example_memory_operations()
    # example_batch_transform()
    # example_advanced_vmp()

    evaluation_mibench()
    print("\n===================")
    print("Demo completed!")


if __name__ == "__main__":
    main()