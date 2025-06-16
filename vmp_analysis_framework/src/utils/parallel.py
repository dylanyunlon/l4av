"""
Utility modules for VMP Analysis Framework
"""

# === parallel.py ===
"""
Parallel processing utilities
"""

import logging
import multiprocessing as mp
from typing import List, Any, Callable, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import traceback

logger = logging.getLogger(__name__)


class ParallelProcessor:
    """Utilities for parallel processing of large datasets"""

    @staticmethod
    def process_in_parallel(data: List[Any],
                            process_func: Callable[[Any], Any],
                            num_workers: Optional[int] = None,
                            batch_size: int = 1000,
                            desc: str = "Processing") -> List[Any]:
        """
        Process data in parallel using multiprocessing

        Args:
            data: List of items to process
            process_func: Function to apply to each item
            num_workers: Number of worker processes (default: CPU count)
            batch_size: Size of batches for processing
            desc: Description for progress bar

        Returns:
            List of processed results
        """
        if num_workers is None:
            num_workers = mp.cpu_count()

        logger.info(f"Starting parallel processing with {num_workers} workers")

        results = []
        failed_items = 0

        # Split data into batches
        batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(_process_batch, batch, process_func): i
                for i, batch in enumerate(batches)
            }

            # Process completed batches
            for future in tqdm(as_completed(future_to_batch),
                               total=len(batches),
                               desc=desc):
                batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {str(e)}")
                    failed_items += len(batches[batch_idx])
                    # Add None for failed items
                    results.extend([None] * len(batches[batch_idx]))

        if failed_items > 0:
            logger.warning(f"Failed to process {failed_items} items")

        return results

    @staticmethod
    def map_reduce(data: List[Any],
                   map_func: Callable[[Any], Any],
                   reduce_func: Callable[[List[Any]], Any],
                   num_workers: Optional[int] = None) -> Any:
        """
        Perform map-reduce operation on data

        Args:
            data: Input data
            map_func: Function to map over data
            reduce_func: Function to reduce mapped results
            num_workers: Number of worker processes

        Returns:
            Reduced result
        """
        # Map phase
        mapped_results = ParallelProcessor.process_in_parallel(
            data, map_func, num_workers, desc="Mapping"
        )

        # Filter out None results
        valid_results = [r for r in mapped_results if r is not None]

        # Reduce phase
        logger.info("Starting reduce phase")
        result = reduce_func(valid_results)

        return result


def _process_batch(batch: List[Any], process_func: Callable[[Any], Any]) -> List[Any]:
    """Process a batch of items"""
    results = []
    for item in batch:
        try:
            result = process_func(item)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing item: {str(e)}\n{traceback.format_exc()}")
            results.append(None)
    return results


# === asm_parser.py ===
"""
Assembly parsing utilities
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Instruction:
    """Represents a parsed assembly instruction"""
    address: Optional[int]
    label: Optional[str]
    mnemonic: str
    operands: List[str]
    comment: Optional[str]
    raw_text: str


class AssemblyParser:
    """Advanced assembly parsing utilities"""

    # x86-64 instruction categories
    INSTRUCTION_CATEGORIES = {
        'arithmetic': ['add', 'sub', 'mul', 'imul', 'div', 'idiv', 'inc', 'dec', 'neg'],
        'logic': ['and', 'or', 'xor', 'not', 'test', 'cmp'],
        'shift': ['shl', 'shr', 'sal', 'sar', 'rol', 'ror', 'rcl', 'rcr'],
        'data_transfer': ['mov', 'movzx', 'movsx', 'movsxd', 'lea', 'xchg', 'push', 'pop'],
        'control_flow': ['jmp', 'je', 'jne', 'jz', 'jnz', 'ja', 'jb', 'jg', 'jl', 'jge',
                         'jle', 'jo', 'jno', 'js', 'jns', 'jc', 'jnc', 'call', 'ret',
                         'loop', 'loope', 'loopne'],
        'string': ['movs', 'cmps', 'scas', 'lods', 'stos', 'rep', 'repe', 'repne'],
        'system': ['int', 'syscall', 'sysenter', 'sysexit', 'cpuid', 'rdtsc', 'rdtscp'],
        'sse': ['movaps', 'movups', 'movdqa', 'movdqu', 'pxor', 'paddb', 'paddw', 'paddd'],
        'avx': ['vmovaps', 'vmovups', 'vxorps', 'vaddps', 'vmulps'],
        'crypto': ['aesenc', 'aesenclast', 'aesdec', 'aesdeclast', 'aesimc', 'aeskeygenassist'],
        'vm_specific': ['vm_dispatch', 'vm_handler', 'vm_fetch', 'vm_decode', 'vm_execute']
    }

    def __init__(self):
        self.label_pattern = re.compile(r'^\.?([a-zA-Z_][a-zA-Z0-9_]*):')
        self.instruction_pattern = re.compile(
            r'^(?:([0-9a-fA-F]+):)?\s*(?:([a-zA-Z_][a-zA-Z0-9_]*):)?\s*([a-zA-Z]\w*)\s*(.*?)(?:;\s*(.*))?$'
        )
        self.operand_pattern = re.compile(r'[,\s]+')
        self.register_pattern = re.compile(
            r'\b(rax|rbx|rcx|rdx|rsi|rdi|rbp|rsp|r8|r9|r10|r11|r12|r13|r14|r15|'
            r'eax|ebx|ecx|edx|esi|edi|ebp|esp|r8d|r9d|r10d|r11d|r12d|r13d|r14d|r15d|'
            r'ax|bx|cx|dx|si|di|bp|sp|r8w|r9w|r10w|r11w|r12w|r13w|r14w|r15w|'
            r'al|bl|cl|dl|sil|dil|bpl|spl|r8b|r9b|r10b|r11b|r12b|r13b|r14b|r15b|'
            r'ah|bh|ch|dh)\b',
            re.IGNORECASE
        )

    def parse_assembly(self, asm_text: str) -> List[Instruction]:
        """Parse assembly text into structured instructions"""
        instructions = []

        for line in asm_text.strip().split('\n'):
            line = line.strip()

            # Skip empty lines and pure comments
            if not line or line.startswith(';'):
                continue

            # Skip directives
            if line.startswith('.') and ':' not in line:
                continue

            # Parse instruction
            match = self.instruction_pattern.match(line)
            if match:
                address_str, label, mnemonic, operands_str, comment = match.groups()

                # Parse address if present
                address = int(address_str, 16) if address_str else None

                # Parse operands
                operands = []
                if operands_str:
                    operands = [op.strip() for op in self.operand_pattern.split(operands_str) if op.strip()]

                instruction = Instruction(
                    address=address,
                    label=label,
                    mnemonic=mnemonic.lower(),
                    operands=operands,
                    comment=comment,
                    raw_text=line
                )

                instructions.append(instruction)

        return instructions

    def analyze_instruction_mix(self, instructions: List[Instruction]) -> Dict[str, int]:
        """Analyze the mix of instruction categories"""
        category_counts = defaultdict(int)
        uncategorized = 0

        for inst in instructions:
            categorized = False
            for category, mnemonics in self.INSTRUCTION_CATEGORIES.items():
                if inst.mnemonic in mnemonics:
                    category_counts[category] += 1
                    categorized = True
                    break

            if not categorized:
                uncategorized += 1

        if uncategorized > 0:
            category_counts['uncategorized'] = uncategorized

        return dict(category_counts)

    def extract_control_flow_graph(self, instructions: List[Instruction]) -> Dict[str, List[str]]:
        """Extract basic control flow graph from instructions"""
        cfg = defaultdict(list)
        labels = {}

        # First pass: collect all labels
        for i, inst in enumerate(instructions):
            if inst.label:
                labels[inst.label] = i

        # Second pass: build CFG
        for i, inst in enumerate(instructions):
            current_block = f"block_{i}"

            if inst.mnemonic in ['jmp', 'je', 'jne', 'jz', 'jnz', 'ja', 'jb',
                                 'jg', 'jl', 'jge', 'jle', 'jo', 'jno',
                                 'js', 'jns', 'jc', 'jnc']:
                # Conditional or unconditional jump
                if inst.operands and inst.operands[0] in labels:
                    target_block = f"block_{labels[inst.operands[0]]}"
                    cfg[current_block].append(target_block)

                # Add fall-through for conditional jumps
                if inst.mnemonic != 'jmp' and i < len(instructions) - 1:
                    cfg[current_block].append(f"block_{i + 1}")

            elif inst.mnemonic == 'call':
                # Function call
                if inst.operands and inst.operands[0] in labels:
                    target_block = f"block_{labels[inst.operands[0]]}"
                    cfg[current_block].append(target_block)

                # Add return path
                if i < len(instructions) - 1:
                    cfg[current_block].append(f"block_{i + 1}")

            elif inst.mnemonic == 'ret':
                # Return instruction - no successors
                pass

            else:
                # Regular instruction - falls through to next
                if i < len(instructions) - 1:
                    cfg[current_block].append(f"block_{i + 1}")

        return dict(cfg)

    def detect_patterns(self, instructions: List[Instruction]) -> Dict[str, List[Dict[str, Any]]]:
        """Detect common assembly patterns"""
        patterns = defaultdict(list)

        # Pattern 1: Push-pop sequences
        for i in range(len(instructions) - 1):
            if (instructions[i].mnemonic == 'push' and
                    instructions[i + 1].mnemonic == 'pop'):
                patterns['push_pop_sequence'].append({
                    'index': i,
                    'registers': (instructions[i].operands[0] if instructions[i].operands else None,
                                  instructions[i + 1].operands[0] if instructions[i + 1].operands else None)
                })

        # Pattern 2: XOR self (zeroing)
        for i, inst in enumerate(instructions):
            if (inst.mnemonic == 'xor' and len(inst.operands) >= 2 and
                    inst.operands[0] == inst.operands[1]):
                patterns['xor_zeroing'].append({
                    'index': i,
                    'register': inst.operands[0]
                })

        # Pattern 3: Function prologue
        for i in range(len(instructions) - 2):
            if (instructions[i].mnemonic == 'push' and
                    instructions[i].operands and instructions[i].operands[0] == 'rbp' and
                    instructions[i + 1].mnemonic == 'mov' and
                    instructions[i + 1].operands and instructions[i + 1].operands[0] == 'rbp' and
                    instructions[i + 1].operands[1] == 'rsp'):
                patterns['function_prologue'].append({'index': i})

        # Pattern 4: Function epilogue
        for i in range(len(instructions) - 1):
            if (instructions[i].mnemonic == 'leave' or
                    (instructions[i].mnemonic == 'mov' and
                     instructions[i].operands and instructions[i].operands[0] == 'rsp' and
                     instructions[i].operands[1] == 'rbp')):
                if i + 1 < len(instructions) and instructions[i + 1].mnemonic == 'ret':
                    patterns['function_epilogue'].append({'index': i})

        # Pattern 5: Loop patterns
        for i, inst in enumerate(instructions):
            if inst.mnemonic in ['loop', 'loope', 'loopne']:
                patterns['explicit_loop'].append({
                    'index': i,
                    'type': inst.mnemonic
                })

        # Pattern 6: Indirect jumps (VM dispatch pattern)
        for i, inst in enumerate(instructions):
            if inst.mnemonic == 'jmp' and inst.operands:
                operand = inst.operands[0]
                if '[' in operand or any(reg in operand for reg in ['rax', 'rbx', 'rcx', 'rdx']):
                    patterns['indirect_jump'].append({
                        'index': i,
                        'target': operand
                    })

        return dict(patterns)

    def calculate_complexity_metrics(self, instructions: List[Instruction]) -> Dict[str, float]:
        """Calculate various complexity metrics"""
        metrics = {}

        # Basic counts
        total_instructions = len(instructions)
        metrics['instruction_count'] = total_instructions

        # Unique mnemonics (instruction diversity)
        unique_mnemonics = len(set(inst.mnemonic for inst in instructions))
        metrics['unique_instructions'] = unique_mnemonics
        metrics['instruction_diversity'] = unique_mnemonics / max(total_instructions, 1)

        # Control flow complexity
        control_flow_instructions = sum(
            1 for inst in instructions
            if inst.mnemonic in self.INSTRUCTION_CATEGORIES['control_flow']
        )
        metrics['control_flow_density'] = control_flow_instructions / max(total_instructions, 1)

        # Register usage
        all_registers = []
        for inst in instructions:
            for operand in inst.operands:
                registers = self.register_pattern.findall(operand)
                all_registers.extend(registers)

        unique_registers = len(set(reg.lower() for reg in all_registers))
        metrics['unique_registers_used'] = unique_registers
        metrics['register_usage_density'] = len(all_registers) / max(total_instructions, 1)

        # Memory access patterns
        memory_access_pattern = re.compile(r'\[.*\]')
        memory_accesses = sum(
            1 for inst in instructions
            for operand in inst.operands
            if memory_access_pattern.search(operand)
        )
        metrics['memory_access_density'] = memory_accesses / max(total_instructions, 1)

        # Arithmetic intensity
        arithmetic_instructions = sum(
            1 for inst in instructions
            if inst.mnemonic in self.INSTRUCTION_CATEGORIES['arithmetic']
        )
        metrics['arithmetic_intensity'] = arithmetic_instructions / max(total_instructions, 1)

        return metrics


# === requirements.txt ===
"""
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.3.0
scikit-learn>=0.24.0
scipy>=1.7.0
networkx>=2.6.0
tqdm>=4.62.0
pyyaml>=5.4.0
jinja2>=3.0.0
capstone>=4.0.2
"""