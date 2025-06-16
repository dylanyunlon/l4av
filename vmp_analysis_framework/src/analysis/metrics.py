"""
VMP Protection Metrics Analysis Module
"""

import re
import logging
from typing import Dict, Any, List, Set, Tuple
from collections import Counter, defaultdict
import networkx as nx
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VMPMetrics:
    """Container for VMP protection metrics"""
    code_expansion_rate: float
    instruction_diversity: float
    control_flow_complexity: float
    obfuscation_strength: float
    anti_debug_features: int
    register_usage_pattern: Dict[str, int]
    jump_density: float
    loop_complexity: float
    vm_handler_count: int
    encryption_indicators: int


class VMPMetricsAnalyzer:
    """Analyzes VMP protection effectiveness metrics"""

    # x86-64 registers
    REGISTERS = {
        'general': ['rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rbp', 'rsp',
                    'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15'],
        'segment': ['cs', 'ds', 'es', 'fs', 'gs', 'ss'],
        'partial': ['eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'ebp', 'esp',
                    'ax', 'bx', 'cx', 'dx', 'al', 'ah', 'bl', 'bh', 'cl', 'ch', 'dl', 'dh']
    }

    # Anti-debugging instructions
    ANTI_DEBUG_INSTRUCTIONS = [
        'rdtsc', 'rdtscp', 'cpuid', 'int3', 'int', 'sidt', 'sgdt', 'sldt',
        'str', 'pushf', 'popf', 'pushfd', 'popfd'
    ]

    # Control flow instructions
    CONTROL_FLOW_INSTRUCTIONS = [
        'jmp', 'je', 'jne', 'jz', 'jnz', 'ja', 'jb', 'jg', 'jl', 'jge', 'jle',
        'jo', 'jno', 'js', 'jns', 'jc', 'jnc', 'call', 'ret', 'loop', 'jcxz', 'jecxz'
    ]

    def __init__(self):
        self.instruction_pattern = re.compile(r'^\s*([a-zA-Z]+)\s+(.*)$', re.MULTILINE)
        self.label_pattern = re.compile(r'^\.?[a-zA-Z_][a-zA-Z0-9_]*:$', re.MULTILINE)
        self.vm_handler_pattern = re.compile(r'vm[_\s]*(handler|interpreter|loop|dispatch)', re.IGNORECASE)

    def analyze_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single VMP transformation entry"""
        try:
            metrics = self._calculate_metrics(
                entry['original_assembly'],
                entry['vmp_assembly'],
                entry['bytecode_size']
            )

            return {
                'line': entry['line'],
                'function': entry['function'],
                'function_category': entry.get('function_category', 'unknown'),
                'metrics': metrics.__dict__,
                'bytecode_size': entry['bytecode_size']
            }
        except Exception as e:
            logger.error(f"Error analyzing entry {entry['line']}: {str(e)}")
            return None

    def _calculate_metrics(self, original_asm: str, vmp_asm: str, bytecode_size: int) -> VMPMetrics:
        """Calculate all VMP protection metrics"""
        # Parse instructions
        orig_instructions = self._parse_instructions(original_asm)
        vmp_instructions = self._parse_instructions(vmp_asm)

        # Calculate individual metrics
        metrics = VMPMetrics(
            code_expansion_rate=self._calc_code_expansion(original_asm, vmp_asm, bytecode_size),
            instruction_diversity=self._calc_instruction_diversity(vmp_instructions),
            control_flow_complexity=self._calc_control_flow_complexity(vmp_asm, vmp_instructions),
            obfuscation_strength=self._calc_obfuscation_strength(vmp_asm, vmp_instructions),
            anti_debug_features=self._count_anti_debug_features(vmp_instructions),
            register_usage_pattern=self._analyze_register_usage(orig_instructions, vmp_instructions),
            jump_density=self._calc_jump_density(vmp_instructions),
            loop_complexity=self._calc_loop_complexity(vmp_asm, vmp_instructions),
            vm_handler_count=self._count_vm_handlers(vmp_asm),
            encryption_indicators=self._count_encryption_indicators(vmp_instructions)
        )

        return metrics

    def _parse_instructions(self, asm_text: str) -> List[Tuple[str, str]]:
        """Parse assembly text into (instruction, operands) tuples"""
        instructions = []

        for match in self.instruction_pattern.finditer(asm_text):
            mnemonic = match.group(1).lower()
            operands = match.group(2).strip()

            # Skip directives and pseudo-instructions
            if not mnemonic.startswith('.') and mnemonic not in ['db', 'dw', 'dd', 'dq', 'section', 'global']:
                instructions.append((mnemonic, operands))

        return instructions

    def _calc_code_expansion(self, original: str, vmp: str, bytecode_size: int) -> float:
        """Calculate code expansion rate"""
        original_size = len(original.encode('utf-8'))
        vmp_size = len(vmp.encode('utf-8')) + bytecode_size

        if original_size == 0:
            return float('inf')

        return vmp_size / original_size

    def _calc_instruction_diversity(self, instructions: List[Tuple[str, str]]) -> float:
        """Calculate instruction diversity (Shannon entropy)"""
        # if not instructions:
        #     return 0.0

        # instruction_counts = Counter(inst[0] for inst in instructions)
        # total = sum(instruction_counts.values())

        # entropy = 0.0
        # for count in instruction_counts.values():
        #     if count > 0:
        #         prob = count / total
        #         entropy -= prob * (prob if prob == 0 else prob * (prob.bit_length() - 1))

        # # Normalize by maximum possible entropy
        # max_entropy = len(instruction_counts).bit_length() - 1 if len(instruction_counts) > 1 else 1
        return  0.0

    def _calc_control_flow_complexity(self, asm_text: str, instructions: List[Tuple[str, str]]) -> float:
        """Calculate control flow graph complexity"""
        # Build control flow graph
        cfg = self._build_cfg(asm_text, instructions)

        if cfg.number_of_nodes() == 0:
            return 0.0

        # Calculate cyclomatic complexity
        cyclomatic = cfg.number_of_edges() - cfg.number_of_nodes() + 2

        # Calculate average node degree
        avg_degree = sum(dict(cfg.degree()).values()) / cfg.number_of_nodes()

        # Combined complexity score
        return (cyclomatic + avg_degree) / 2

    def _build_cfg(self, asm_text: str, instructions: List[Tuple[str, str]]) -> nx.DiGraph:
        """Build control flow graph from assembly"""
        cfg = nx.DiGraph()

        # Find all labels
        labels = {match.group(0).rstrip(':'): i
                  for i, match in enumerate(self.label_pattern.finditer(asm_text))}

        # Add nodes and edges
        for i, (mnemonic, operands) in enumerate(instructions):
            cfg.add_node(i)

            if mnemonic in self.CONTROL_FLOW_INSTRUCTIONS:
                # Handle jumps and calls
                if operands in labels:
                    cfg.add_edge(i, labels[operands])

            # Add sequential flow (except after unconditional jumps/returns)
            if i < len(instructions) - 1 and mnemonic not in ['jmp', 'ret']:
                cfg.add_edge(i, i + 1)

        return cfg

    def _calc_obfuscation_strength(self, asm_text: str, instructions: List[Tuple[str, str]]) -> float:
        """Calculate overall obfuscation strength"""
        factors = []

        # Factor 1: Instruction obfuscation patterns
        obfuscation_patterns = [
            (r'xor\s+(\w+),\s*\1', 0.5),  # Self-XOR
            (r'push.*\n.*pop', 0.3),  # Push-pop sequences
            (r'lea\s+\w+,\s*\[.*\]', 0.4),  # Complex addressing
            (r'(ror|rol)\s+\w+,', 0.6),  # Rotation operations
            (r'neg\s+\w+.*\n.*neg\s+\w+', 0.7),  # Double negation
        ]

        for pattern, weight in obfuscation_patterns:
            matches = len(re.findall(pattern, asm_text, re.IGNORECASE))
            factors.append(min(matches * weight / 10, 1.0))

        # Factor 2: Junk instruction density
        junk_instructions = ['nop', 'xchg', 'lahf', 'sahf', 'cld', 'std', 'stc', 'clc']
        junk_count = sum(1 for inst, _ in instructions if inst in junk_instructions)
        factors.append(min(junk_count / max(len(instructions), 1), 1.0))

        # Factor 3: VM complexity indicators
        vm_complexity = asm_text.count('vm_') + asm_text.count('interpreter') + asm_text.count('handler')
        factors.append(min(vm_complexity / 20, 1.0))

        return sum(factors) / len(factors) if factors else 0.0

    def _count_anti_debug_features(self, instructions: List[Tuple[str, str]]) -> int:
        """Count anti-debugging features"""
        count = 0

        for mnemonic, _ in instructions:
            if mnemonic in self.ANTI_DEBUG_INSTRUCTIONS:
                count += 1

        return count

    def _analyze_register_usage(self, orig_inst: List[Tuple[str, str]],
                                vmp_inst: List[Tuple[str, str]]) -> Dict[str, int]:
        """Analyze register usage patterns"""
        pattern = {}

        # Count register usage in original vs VMP
        orig_regs = self._extract_registers(orig_inst)
        vmp_regs = self._extract_registers(vmp_inst)

        pattern['original_unique_registers'] = len(set(orig_regs))
        pattern['vmp_unique_registers'] = len(set(vmp_regs))
        pattern['register_diversity_increase'] = (
                len(set(vmp_regs)) - len(set(orig_regs))
        ) if orig_regs else len(set(vmp_regs))

        # Count specific register categories
        for category, regs in self.REGISTERS.items():
            pattern[f'{category}_usage'] = sum(1 for r in vmp_regs if r in regs)

        return pattern

    def _extract_registers(self, instructions: List[Tuple[str, str]]) -> List[str]:
        """Extract all register references from instructions"""
        registers = []
        all_regs = sum(self.REGISTERS.values(), [])

        for _, operands in instructions:
            for reg in all_regs:
                if re.search(r'\b' + reg + r'\b', operands, re.IGNORECASE):
                    registers.append(reg.lower())

        return registers

    def _calc_jump_density(self, instructions: List[Tuple[str, str]]) -> float:
        """Calculate density of jump instructions"""
        if not instructions:
            return 0.0

        jump_count = sum(1 for inst, _ in instructions
                         if inst in self.CONTROL_FLOW_INSTRUCTIONS)
        return jump_count / len(instructions)

    def _calc_loop_complexity(self, asm_text: str, instructions: List[Tuple[str, str]]) -> float:
        """Calculate loop complexity based on backward jumps and loop patterns"""
        complexity = 0.0

        # Count explicit loop instructions
        loop_count = sum(1 for inst, _ in instructions if inst.startswith('loop'))
        complexity += loop_count * 0.5

        # Detect backward jumps (simple heuristic)
        backward_jump_pattern = re.compile(r'j\w+\s+\.[a-zA-Z_]\w*', re.IGNORECASE)
        backward_jumps = len(backward_jump_pattern.findall(asm_text))
        complexity += backward_jumps * 0.3

        # Detect nested loop patterns
        if 'vm_loop' in asm_text.lower():
            nested_loops = asm_text.lower().count('loop') - 1
            complexity += nested_loops * 0.7

        return min(complexity, 10.0)  # Cap at 10

    def _count_vm_handlers(self, asm_text: str) -> int:
        """Count VM handler implementations"""
        return len(self.vm_handler_pattern.findall(asm_text))

    def _count_encryption_indicators(self, instructions: List[Tuple[str, str]]) -> int:
        """Count indicators of encryption/decryption operations"""
        crypto_instructions = ['aes', 'pxor', 'pshufd', 'pshufb', 'movdqa', 'movdqu']
        crypto_patterns = ['xor', 'ror', 'rol', 'shl', 'shr', 'not']

        count = 0
        for mnemonic, _ in instructions:
            if any(crypto in mnemonic for crypto in crypto_instructions):
                count += 2
            elif mnemonic in crypto_patterns:
                count += 1

        return count