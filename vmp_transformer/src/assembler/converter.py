# src/assembler/converter.py

import random
import re
from ..core.opcodes import *
from ..core.crypto import XorShift32


class AssemblyToVMPConverter:
    """Convert assembly code to VMP protected bytecode"""

    def __init__(self, arch='x86_64'):
        self.arch = arch
        self.bytecode = []
        self.data_segment = []
        self.labels = {}
        self.variables = {}
        self.current_offset = 0
        self.data_offset = 0
        self.function_name = None

        # Architecture-specific register mappings
        self.register_mappings = {
            'x86_64': {
                # 64-bit registers
                'rax': 0, 'rbx': 1, 'rcx': 2, 'rdx': 3,
                'rsi': 4, 'rdi': 5, 'rbp': 6, 'rsp': 7,
                'r8': 8, 'r9': 9, 'r10': 10, 'r11': 11,
                'r12': 12, 'r13': 13, 'r14': 14, 'r15': 15,
                # 32-bit registers
                'eax': 16, 'ebx': 17, 'ecx': 18, 'edx': 19,
                'esi': 20, 'edi': 21, 'ebp': 22, 'esp': 23,
                # 16-bit registers
                'ax': 24, 'bx': 25, 'cx': 26, 'dx': 27,
                'si': 28, 'di': 29, 'bp': 30, 'sp': 31,
                # 8-bit registers
                'al': 32, 'bl': 33, 'cl': 34, 'dl': 35,
                'ah': 36, 'bh': 37, 'ch': 38, 'dh': 39
            },
            'arm64': {
                # General purpose registers
                **{f'x{i}': i for i in range(31)},
                'sp': 31, 'xzr': 31,  # Stack pointer and zero register
                # 32-bit views
                **{f'w{i}': 32 + i for i in range(31)},
                'wzr': 63,
                # Special registers
                'pc': 64, 'lr': 65, 'fp': 66
            },
            'riscv64': {
                # ABI names
                'zero': 0, 'ra': 1, 'sp': 2, 'gp': 3,
                'tp': 4, 'fp': 8, 's0': 8,
                # Temporaries
                **{f't{i}': 5 + (i if i < 2 else 23 + i) for i in range(7)},
                # Saved registers
                **{f's{i}': 8 + i for i in range(12)},
                # Arguments
                **{f'a{i}': 10 + i for i in range(8)},
                # Direct register names
                **{f'x{i}': i for i in range(32)}
            }
        }

        # Architecture-specific instruction patterns
        self.arch_patterns = {
            'x86_64': {
                'nop': 'nop',
                'stack_pointer': 'rsp',
                'base_pointer': 'rbp',
                'instruction_pointer': 'rip',
                'return_reg': 'rax',
                'param_regs': ['rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9'],
                'caller_saved': ['rax', 'rcx', 'rdx', 'rsi', 'rdi', 'r8', 'r9', 'r10', 'r11'],
                'callee_saved': ['rbx', 'rbp', 'r12', 'r13', 'r14', 'r15']
            },
            'arm64': {
                'nop': 'nop',
                'stack_pointer': 'sp',
                'base_pointer': 'x29',
                'instruction_pointer': 'pc',
                'return_reg': 'x0',
                'param_regs': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'],
                'caller_saved': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7',
                                 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15'],
                'callee_saved': ['x19', 'x20', 'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28']
            },
            'riscv64': {
                'nop': 'nop',
                'stack_pointer': 'sp',
                'base_pointer': 'fp',
                'instruction_pointer': 'pc',
                'return_reg': 'a0',
                'param_regs': ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7'],
                'caller_saved': ['ra', 't0', 't1', 't2', 't3', 't4', 't5', 't6',
                                 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7'],
                'callee_saved': ['sp', 'gp', 'tp', 's0', 's1', 's2', 's3', 's4',
                                 's5', 's6', 's7', 's8', 's9', 's10', 's11']
            }
        }

        # Set current architecture
        self.registers = self.register_mappings.get(arch, self.register_mappings['x86_64'])
        self.patterns = self.arch_patterns.get(arch, self.arch_patterns['x86_64'])

        # Register size mapping (architecture independent)
        self.register_sizes = {
            'r': 8, 'e': 4, 'w': 4, 'x': 8, '': 2, 'l': 1, 'h': 1
        }

    def generate_junk_instructions(self, count=3):
        """Generate random junk instructions for obfuscation based on architecture"""
        if self.arch == 'x86_64':
            return self._generate_x86_64_junk(count)
        elif self.arch == 'arm64':
            return self._generate_arm64_junk(count)
        elif self.arch == 'riscv64':
            return self._generate_riscv64_junk(count)
        else:
            return []  # Unknown architecture

    def _generate_x86_64_junk(self, count):
        """Generate x86-64 specific junk instructions"""
        # Use architecture-specific registers
        regs = self.patterns['caller_saved'][:6]  # Use first 6 caller-saved registers

        junk_patterns = [
            # Stack operations that cancel out
            [f"push {regs[0]}", f"pop {regs[0]}"],
            [f"push {regs[1]}", f"pop {regs[1]}"],
            [f"push {regs[2]}", f"pop {regs[2]}"],

            # Register operations that do nothing
            [f"xor {regs[3]}, {regs[3]}", f"or {regs[3]}, {regs[3]}"],
            [f"mov {regs[4]}, {regs[4]}"],
            [f"and {regs[5]}, -1"],
            [f"or {regs[0]}, 0"],

            # Math operations that cancel
            [f"inc {regs[1]}", f"dec {regs[1]}"],
            [f"add {regs[2]}, 1", f"sub {regs[2]}, 1"],
            [f"neg {regs[3]}", f"neg {regs[3]}"],

            # Bit operations
            [f"shl {regs[4]}, 1", f"shr {regs[4]}, 1"],
            [f"rol {regs[5]}, 8", f"ror {regs[5]}, 8"],

            # Memory operations (safe)
            [f"lea {regs[0]}, [{self.patterns['stack_pointer']}]"],
            [f"lea {regs[1]}, [{self.patterns['instruction_pointer']}]"],

            # Flag operations
            ["clc", "stc", "clc"],
            ["cld"],
            ["std", "cld"],
        ]

        instructions = []
        for _ in range(count):
            pattern = random.choice(junk_patterns)
            if isinstance(pattern, list):
                instructions.extend(pattern)
            else:
                instructions.append(pattern)

        return instructions

    def _generate_arm64_junk(self, count):
        """Generate ARM64 specific junk instructions"""
        regs = ['x8', 'x9', 'x10', 'x11', 'x12', 'x13']  # Temporary registers

        junk_patterns = [
            # Register operations
            [f"mov {regs[0]}, {regs[0]}"],
            [f"eor {regs[1]}, {regs[1]}, {regs[1]}", f"orr {regs[1]}, {regs[1]}, {regs[1]}"],
            [f"and {regs[2]}, {regs[2]}, #-1"],

            # Math operations that cancel
            [f"add {regs[3]}, {regs[3]}, #1", f"sub {regs[3]}, {regs[3]}, #1"],
            [f"neg {regs[4]}, {regs[4]}", f"neg {regs[4]}, {regs[4]}"],

            # Bit operations
            [f"lsl {regs[5]}, {regs[5]}, #1", f"lsr {regs[5]}, {regs[5]}, #1"],
            [f"ror {regs[0]}, {regs[0]}, #8", f"ror {regs[0]}, {regs[0]}, #24"],

            # Memory operations (safe)
            ["nop"],
            ["dmb sy"],  # Data memory barrier
        ]

        instructions = []
        for _ in range(count):
            pattern = random.choice(junk_patterns)
            if isinstance(pattern, list):
                instructions.extend(pattern)
            else:
                instructions.append(pattern)

        return instructions

    def _generate_riscv64_junk(self, count):
        """Generate RISC-V specific junk instructions"""
        regs = ['t0', 't1', 't2', 't3', 't4', 't5']  # Temporary registers

        junk_patterns = [
            # Register operations
            [f"mv {regs[0]}, {regs[0]}"],  # pseudo-instruction for add rd, rs, zero
            [f"xor {regs[1]}, {regs[1]}, {regs[1]}", f"or {regs[1]}, {regs[1]}, {regs[1]}"],
            [f"andi {regs[2]}, {regs[2]}, -1"],

            # Math operations that cancel
            [f"addi {regs[3]}, {regs[3]}, 1", f"addi {regs[3]}, {regs[3]}, -1"],
            [f"neg {regs[4]}, {regs[4]}", f"neg {regs[4]}, {regs[4]}"],

            # Bit operations
            [f"slli {regs[5]}, {regs[5]}, 1", f"srli {regs[5]}, {regs[5]}, 1"],

            # Memory operations (safe)
            ["nop"],
            ["fence"],  # Memory fence
        ]

        instructions = []
        for _ in range(count):
            pattern = random.choice(junk_patterns)
            if isinstance(pattern, list):
                instructions.extend(pattern)
            else:
                instructions.append(pattern)

        return instructions

    def preprocess_assembly(self, asm_code):
        """Preprocess assembly code from the given format"""
        if isinstance(asm_code, dict):
            # Extract assembly code from the message format
            messages = asm_code.get("messages", [])
            if messages:
                content = messages[0].get("content", "")
                # Remove everything before the double newline
                if "\n\n" in content:
                    content = content.split("\n\n", 1)[1]
                return content
        return asm_code

    def generate_seed(self):
        """Generate random seed for encryption"""
        return random.randint(1, 0xFFFFFFFF)

    def emit_seed(self, seed):
        """Emit 4-byte seed (unencrypted)"""
        for i in range(4):
            self.bytecode.append((seed >> (8 * i)) & 0xFF)
        self.current_offset += 4

    def emit_byte(self, byte):
        """Emit single byte"""
        self.bytecode.append(byte & 0xFF)
        self.current_offset += 1

    def emit_value(self, value, size):
        """Emit multi-byte value"""
        # Handle negative values
        if value < 0:
            value = (1 << (size * 8)) + value

        for i in range(size):
            self.bytecode.append((value >> (8 * i)) & 0xFF)
        self.current_offset += size

    def emit_register(self, reg_name):
        """Emit register reference"""
        if reg_name in self.registers:
            reg_id = self.registers[reg_name]
            self.emit_byte(1)  # value_size (register ID)
            self.emit_byte(2)  # value_type (2 = register)
            self.emit_byte(reg_id)
        else:
            # Unknown register, treat as variable
            self.emit_variable_ref(reg_name, 8)

    def emit_variable_ref(self, var_name, size):
        """Emit variable reference"""
        if var_name not in self.variables:
            # Allocate new variable
            self.variables[var_name] = self.data_offset
            self.data_offset += size

        self.emit_byte(size)  # value_size
        self.emit_byte(0)  # value_type (0 = variable)
        self.emit_value(self.variables[var_name], POINTER_SIZE)

    def emit_constant(self, value, size):
        """Emit constant value"""
        self.emit_byte(size)  # value_size
        self.emit_byte(1)  # value_type (1 = constant)
        self.emit_value(value, size)

    def parse_operand(self, operand):
        """Parse an assembly operand and return its components"""
        operand = operand.strip()

        # Architecture-specific operand parsing
        if self.arch == 'x86_64':
            return self._parse_x86_64_operand(operand)
        elif self.arch == 'arm64':
            return self._parse_arm64_operand(operand)
        elif self.arch == 'riscv64':
            return self._parse_riscv64_operand(operand)
        else:
            # Default parsing
            return {'type': 'unknown', 'value': operand}

    def _parse_x86_64_operand(self, operand):
        """Parse x86-64 specific operand"""
        # Check for register
        if operand.startswith('%'):
            return {'type': 'register', 'value': operand[1:]}

        # Check for immediate value
        if operand.startswith('$'):
            # Remove the $ and parse the value
            value_str = operand[1:]
            if value_str.startswith('-0x'):
                value = -int(value_str[3:], 16)
            elif value_str.startswith('0x'):
                value = int(value_str[2:], 16)
            else:
                value = int(value_str)
            return {'type': 'immediate', 'value': value}

        # Check for memory reference with offset
        match = re.match(r'(-?0x[0-9a-fA-F]+|-?\d+)?\((%\w+)\)', operand)
        if match:
            offset = 0
            if match.group(1):
                offset_str = match.group(1)
                # Handle negative hex numbers
                if offset_str.startswith('-0x'):
                    offset = -int(offset_str[3:], 16)
                elif offset_str.startswith('0x'):
                    offset = int(offset_str[2:], 16)
                elif offset_str.startswith('-'):
                    offset = int(offset_str)
                else:
                    offset = int(offset_str)

            base_reg = match.group(2)[1:]  # Remove %
            return {'type': 'memory', 'base': base_reg, 'offset': offset}

        # Check for function call reference
        if operand.startswith('<') and operand.endswith('>'):
            func_name = operand[1:-1]
            if '@' in func_name:
                func_name = func_name.split('@')[0]
            return {'type': 'function', 'value': func_name}

        # Default: treat as label
        return {'type': 'label', 'value': operand}

    def _parse_arm64_operand(self, operand):
        """Parse ARM64 specific operand"""
        # Register (x0-x30, w0-w30, sp, etc.)
        if re.match(r'^[xw]\d+$|^sp$|^lr$|^fp$|^xzr$|^wzr$', operand):
            return {'type': 'register', 'value': operand}

        # Immediate value
        if operand.startswith('#'):
            value_str = operand[1:]
            if value_str.startswith('-0x'):
                value = -int(value_str[3:], 16)
            elif value_str.startswith('0x'):
                value = int(value_str[2:], 16)
            else:
                value = int(value_str)
            return {'type': 'immediate', 'value': value}

        # Memory reference [reg, #offset]
        match = re.match(r'\[(\w+)(?:,\s*#(-?\d+))?\]', operand)
        if match:
            base_reg = match.group(1)
            offset = int(match.group(2)) if match.group(2) else 0
            return {'type': 'memory', 'base': base_reg, 'offset': offset}

        # Label
        return {'type': 'label', 'value': operand}

    def _parse_riscv64_operand(self, operand):
        """Parse RISC-V specific operand"""
        # Register (x0-x31 or ABI names)
        if operand in self.registers or re.match(r'^x\d+$', operand):
            return {'type': 'register', 'value': operand}

        # Immediate value
        try:
            if operand.startswith('-0x'):
                value = -int(operand[3:], 16)
            elif operand.startswith('0x'):
                value = int(operand[2:], 16)
            else:
                value = int(operand)
            return {'type': 'immediate', 'value': value}
        except ValueError:
            pass

        # Memory reference offset(reg)
        match = re.match(r'(-?\d+)\((\w+)\)', operand)
        if match:
            offset = int(match.group(1))
            base_reg = match.group(2)
            return {'type': 'memory', 'base': base_reg, 'offset': offset}

        # Label
        return {'type': 'label', 'value': operand}

    def convert_assembly_line(self, line):
        """Convert single assembly instruction to VMP bytecode"""
        line = line.strip()
        if not line:
            return

        # Handle function labels
        if line.startswith('<') and line.endswith('>:'):
            self.function_name = line[1:-2]
            self.labels[self.function_name] = self.current_offset
            return

        # Handle regular labels
        if line.endswith(':') and not line.startswith(' '):
            label = line[:-1]
            self.labels[label] = self.current_offset
            return

        # Split instruction and operands
        parts = line.split(None, 1)
        if not parts:
            return

        instruction = parts[0].lower()
        operands = parts[1].split(',') if len(parts) > 1 else []
        operands = [op.strip() for op in operands]

        # Start new basic block
        opcode_seed = self.generate_seed()
        code_seed = self.generate_seed()
        self.emit_seed(opcode_seed)
        self.emit_seed(code_seed)

        # Encrypt opcode
        xorshift = XorShift32(opcode_seed)
        opcode_mapping = {}
        seen = set()

        for op in range(1, OP_TOTAL + 1):
            while True:
                encrypted = xorshift.next() & 0xFF
                if encrypted not in seen:
                    seen.add(encrypted)
                    opcode_mapping[op] = encrypted
                    break

        # Architecture-specific instruction conversion
        if self.arch == 'x86_64':
            self._convert_x86_64_instruction(instruction, operands, opcode_mapping)
        elif self.arch == 'arm64':
            self._convert_arm64_instruction(instruction, operands, opcode_mapping)
        elif self.arch == 'riscv64':
            self._convert_riscv64_instruction(instruction, operands, opcode_mapping)

    def _convert_x86_64_instruction(self, instruction, operands, opcode_mapping):
        """Convert x86-64 instructions"""
        if instruction == 'mov':
            src_op = self.parse_operand(operands[0])
            dst_op = self.parse_operand(operands[1])

            if dst_op['type'] == 'memory':
                # Store to memory
                self.emit_byte(opcode_mapping[STORE_OP])

                # Source
                if src_op['type'] == 'register':
                    self.emit_register(src_op['value'])
                elif src_op['type'] == 'immediate':
                    self.emit_constant(src_op['value'], 8)

                # Destination (memory)
                self.emit_register(dst_op['base'])
                if dst_op['offset'] != 0:
                    # Add offset handling
                    self.emit_constant(dst_op['offset'], 8)

            elif src_op['type'] == 'memory':
                # Load from memory
                self.emit_byte(opcode_mapping[LOAD_OP])

                # Destination
                if dst_op['type'] == 'register':
                    self.emit_register(dst_op['value'])

                # Source (memory)
                self.emit_register(src_op['base'])
                if src_op['offset'] != 0:
                    self.emit_constant(src_op['offset'], 8)

            else:
                # Register to register or immediate to register
                self.emit_byte(opcode_mapping[STORE_OP])

                if src_op['type'] == 'immediate':
                    self.emit_constant(src_op['value'], 8)
                else:
                    self.emit_register(src_op['value'])

                self.emit_register(dst_op['value'])

        elif instruction == 'push':
            # Push operation
            self.emit_byte(opcode_mapping[PUSH_OP])
            op = self.parse_operand(operands[0])

            if op['type'] == 'register':
                self.emit_register(op['value'])
            elif op['type'] == 'immediate':
                self.emit_constant(op['value'], 8)

        elif instruction == 'pop':
            # Pop operation
            self.emit_byte(opcode_mapping[POP_OP])
            op = self.parse_operand(operands[0])

            if op['type'] == 'register':
                self.emit_register(op['value'])

        elif instruction == 'sub':
            # SUB instruction
            self.emit_byte(opcode_mapping[BinaryOperator_OP])
            self.emit_byte(BINOP_SUB)

            # For x86, first operand is both source and destination
            dst_op = self.parse_operand(operands[1])
            src_op = self.parse_operand(operands[0])

            # Result (destination)
            if dst_op['type'] == 'register':
                self.emit_register(dst_op['value'])

            # Operands
            if dst_op['type'] == 'register':
                self.emit_register(dst_op['value'])

            if src_op['type'] == 'immediate':
                self.emit_constant(src_op['value'], 8)
            elif src_op['type'] == 'register':
                self.emit_register(src_op['value'])

        elif instruction == 'add':
            # ADD instruction
            self.emit_byte(opcode_mapping[BinaryOperator_OP])
            self.emit_byte(BINOP_ADD)

            dst_op = self.parse_operand(operands[1])
            src_op = self.parse_operand(operands[0])

            # Result (destination)
            if dst_op['type'] == 'register':
                self.emit_register(dst_op['value'])

            # Operands
            if dst_op['type'] == 'register':
                self.emit_register(dst_op['value'])

            if src_op['type'] == 'immediate':
                self.emit_constant(src_op['value'], 8)
            elif src_op['type'] == 'register':
                self.emit_register(src_op['value'])

        elif instruction == 'call':
            # Function call
            self.emit_byte(opcode_mapping[Call_OP])
            op = self.parse_operand(operands[0])

            if op['type'] == 'function':
                # Emit function name as string constant
                func_name = op['value']
                self.emit_byte(len(func_name))
                self.emit_byte(3)  # value_type (3 = string)
                for char in func_name:
                    self.emit_byte(ord(char))

        elif instruction == 'ret':
            # Return instruction
            self.emit_byte(opcode_mapping[Ret_OP])

            # Check if there's a return value in rax
            self.emit_register('rax')

        elif instruction == 'leave':
            # Leave instruction (restore stack frame)
            # Equivalent to: mov %rbp, %rsp; pop %rbp
            self.emit_byte(opcode_mapping[STORE_OP])
            self.emit_register('rbp')
            self.emit_register('rsp')

        elif instruction == 'endbr64':
            # Intel CET instruction - treat as NOP
            self.emit_byte(opcode_mapping[NOP_OP])

    def _convert_arm64_instruction(self, instruction, operands, opcode_mapping):
        """Convert ARM64 instructions"""
        if instruction in ['mov', 'movz', 'movk']:
            src_op = self.parse_operand(operands[1])
            dst_op = self.parse_operand(operands[0])

            self.emit_byte(opcode_mapping[STORE_OP])

            if src_op['type'] == 'immediate':
                self.emit_constant(src_op['value'], 8)
            else:
                self.emit_register(src_op['value'])

            self.emit_register(dst_op['value'])

        elif instruction == 'ldr':
            # Load instruction
            self.emit_byte(opcode_mapping[LOAD_OP])
            dst_op = self.parse_operand(operands[0])
            src_op = self.parse_operand(operands[1])

            self.emit_register(dst_op['value'])
            self.emit_register(src_op['base'])
            if src_op.get('offset', 0) != 0:
                self.emit_constant(src_op['offset'], 8)

        elif instruction == 'str':
            # Store instruction
            self.emit_byte(opcode_mapping[STORE_OP])
            src_op = self.parse_operand(operands[0])
            dst_op = self.parse_operand(operands[1])

            self.emit_register(src_op['value'])
            self.emit_register(dst_op['base'])
            if dst_op.get('offset', 0) != 0:
                self.emit_constant(dst_op['offset'], 8)

        elif instruction in ['add', 'sub', 'and', 'orr', 'eor']:
            # Binary operations
            opcode = {
                'add': BINOP_ADD,
                'sub': BINOP_SUB,
                'and': BINOP_AND,
                'orr': BINOP_OR,
                'eor': BINOP_XOR
            }.get(instruction, BINOP_ADD)

            self.emit_byte(opcode_mapping[BinaryOperator_OP])
            self.emit_byte(opcode)

            # ARM64: dst, src1, src2
            dst_op = self.parse_operand(operands[0])
            src1_op = self.parse_operand(operands[1])
            src2_op = self.parse_operand(operands[2]) if len(operands) > 2 else src1_op

            self.emit_register(dst_op['value'])
            self.emit_register(src1_op['value'])

            if src2_op['type'] == 'immediate':
                self.emit_constant(src2_op['value'], 8)
            else:
                self.emit_register(src2_op['value'])

        elif instruction == 'bl':
            # Branch and link (call)
            self.emit_byte(opcode_mapping[Call_OP])
            op = self.parse_operand(operands[0])

            if op['type'] == 'label':
                func_name = op['value']
                self.emit_byte(len(func_name))
                self.emit_byte(3)  # value_type (3 = string)
                for char in func_name:
                    self.emit_byte(ord(char))

        elif instruction == 'ret':
            # Return instruction
            self.emit_byte(opcode_mapping[Ret_OP])
            self.emit_register('x0')  # Return value in x0

    def _convert_riscv64_instruction(self, instruction, operands, opcode_mapping):
        """Convert RISC-V instructions"""
        if instruction in ['mv', 'li']:
            # Move or load immediate
            if instruction == 'mv':
                dst_op = self.parse_operand(operands[0])
                src_op = self.parse_operand(operands[1])
            else:  # li
                dst_op = self.parse_operand(operands[0])
                src_op = self.parse_operand(operands[1])

            self.emit_byte(opcode_mapping[STORE_OP])

            if src_op['type'] == 'immediate':
                self.emit_constant(src_op['value'], 8)
            else:
                self.emit_register(src_op['value'])

            self.emit_register(dst_op['value'])

        elif instruction in ['add', 'addi', 'sub', 'subi', 'and', 'andi', 'or', 'ori', 'xor', 'xori']:
            # Binary operations
            base_op = instruction.rstrip('i')
            opcode = {
                'add': BINOP_ADD,
                'sub': BINOP_SUB,
                'and': BINOP_AND,
                'or': BINOP_OR,
                'xor': BINOP_XOR
            }.get(base_op, BINOP_ADD)

            self.emit_byte(opcode_mapping[BinaryOperator_OP])
            self.emit_byte(opcode)

            dst_op = self.parse_operand(operands[0])
            src1_op = self.parse_operand(operands[1])
            src2_op = self.parse_operand(operands[2])

            self.emit_register(dst_op['value'])
            self.emit_register(src1_op['value'])

            if src2_op['type'] == 'immediate':
                self.emit_constant(src2_op['value'], 8)
            else:
                self.emit_register(src2_op['value'])

        elif instruction in ['lw', 'ld']:
            # Load word/doubleword
            self.emit_byte(opcode_mapping[LOAD_OP])
            dst_op = self.parse_operand(operands[0])
            src_op = self.parse_operand(operands[1])

            self.emit_register(dst_op['value'])
            self.emit_register(src_op['base'])
            if src_op.get('offset', 0) != 0:
                self.emit_constant(src_op['offset'], 8)

        elif instruction in ['sw', 'sd']:
            # Store word/doubleword
            self.emit_byte(opcode_mapping[STORE_OP])
            src_op = self.parse_operand(operands[0])
            dst_op = self.parse_operand(operands[1])

            self.emit_register(src_op['value'])
            self.emit_register(dst_op['base'])
            if dst_op.get('offset', 0) != 0:
                self.emit_constant(dst_op['offset'], 8)

        elif instruction == 'jal':
            # Jump and link (call)
            self.emit_byte(opcode_mapping[Call_OP])

            if len(operands) == 1:
                # jal label
                op = self.parse_operand(operands[0])
            else:
                # jal rd, label
                op = self.parse_operand(operands[1])

            if op['type'] == 'label':
                func_name = op['value']
                self.emit_byte(len(func_name))
                self.emit_byte(3)  # value_type (3 = string)
                for char in func_name:
                    self.emit_byte(ord(char))

        elif instruction == 'ret':
            # Return instruction (pseudo-instruction for jalr zero, 0(ra))
            self.emit_byte(opcode_mapping[Ret_OP])
            self.emit_register('a0')  # Return value in a0

    def convert_assembly(self, assembly_code):
        """Convert complete assembly code to VMP bytecode"""
        self.bytecode = []
        self.data_segment = []
        self.labels = {}
        self.variables = {}
        self.current_offset = 0
        self.data_offset = 0

        # Preprocess the assembly code
        assembly_code = self.preprocess_assembly(assembly_code)

        # First pass: convert instructions and collect labels
        lines = assembly_code.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith(';'):  # Skip comments
                self.convert_assembly_line(line)

        # Second pass: patch jump targets
        # (In a real implementation, we would need to track and patch label references)

        return bytes(self.bytecode), self.variables

    def generate_dynamic_interpreter_table(self):
        """Generate randomized interpreter table with decoy entries"""
        table_code = []

        # Random table name
        table_name = f"vmp_interpreter_table_{random.randint(100, 999)}"

        # Generate random number of fake entries
        num_fake_entries = random.randint(3, 8)
        real_entry_pos = random.randint(0, num_fake_entries)

        table_code.append(f"    {table_name}:")

        # Generate fake patterns
        fake_patterns = [
            lambda: f"dq 0x{random.randint(0x1000000000000000, 0xFFFFFFFFFFFFFFFF):016X}",
            lambda: f"dq 0x{''.join(random.choice('0123456789ABCDEF') for _ in range(16))}",
            lambda: "dq 0xDEADBEEFDEADBEEF",
            lambda: "dq 0xCAFEBABECAFEBABE",
            lambda: "dq 0x4141414141414141",
            lambda: "dq 0x9090909090909090",
            lambda: f"dq vmp_fake_{random.randint(100, 999)}",
            lambda: f"dq .fake_label_{random.randint(100, 999)}"
        ]

        # Build table with real entry at random position
        for i in range(num_fake_entries + 1):
            if i == real_entry_pos:
                # Real interpreter entry
                table_code.append(f"    dq vmp_interpreter_impl_{random.randint(1000, 9999)}")
            else:
                # Fake entry
                pattern = random.choice(fake_patterns)
                table_code.append(f"    {pattern()}")

        # Add additional obfuscation data
        if random.choice([True, False]):
            table_code.extend([
                "    ; Decoy data",
                f"    times {random.randint(4, 16)} dq 0",
                f"    db 'VMPX', {random.randint(1, 255)}, {random.randint(1, 255)}, {random.randint(1, 255)}, {random.randint(1, 255)}"
            ])

        # Store table info for entry code
        self.interpreter_table_name = table_name
        self.real_entry_offset = real_entry_pos * 8

        return '\n'.join(table_code) + '\n'

    def generate_obfuscated_mov(self, dst, src, use_alt=True):
        """Generate different ways to move data based on architecture"""
        patterns = []

        if self.arch == 'x86_64':
            if use_alt and random.choice([True, False]):
                # Pattern 1: XOR swap
                patterns = [
                    f"xor {dst}, {src}",
                    f"xor {src}, {dst}",
                    f"xor {dst}, {src}"
                ]
            elif use_alt and random.choice([True, False]):
                # Pattern 2: Push/Pop
                patterns = [
                    f"push {src}",
                    f"pop {dst}"
                ]
            elif use_alt and random.choice([True, False]):
                # Pattern 3: LEA for register
                patterns = [f"lea {dst}, [{src}]"]
            else:
                # Standard mov
                patterns = [f"mov {dst}, {src}"]

        elif self.arch == 'arm64':
            if use_alt and random.choice([True, False]):
                # Pattern 1: ORR (logical OR with zero)
                patterns = [f"orr {dst}, xzr, {src}"]
            elif use_alt and random.choice([True, False]):
                # Pattern 2: ADD with zero
                patterns = [f"add {dst}, {src}, #0"]
            else:
                # Standard mov
                patterns = [f"mov {dst}, {src}"]

        elif self.arch == 'riscv64':
            if use_alt and random.choice([True, False]):
                # Pattern 1: ADDI with zero
                patterns = [f"addi {dst}, {src}, 0"]
            elif use_alt and random.choice([True, False]):
                # Pattern 2: OR with zero
                patterns = [f"or {dst}, {src}, zero"]
            else:
                # Standard mv (pseudo-instruction)
                patterns = [f"mv {dst}, {src}"]

        return patterns

    def shuffle_with_dependencies(self, instructions):
        """Shuffle instructions while respecting dependencies"""
        # Simple shuffle for independent instructions
        independent = []
        dependent = []

        for inst in instructions:
            if any(keyword in inst for keyword in ['loop', 'jmp', 'jnz', 'jz', 'je', 'jne', 'jg', 'jl', 'jge', 'jle',
                                                   'call', 'ret', ':', 'b.', 'bl', 'br', 'beq', 'bne', 'blt', 'bge',
                                                   'jal', 'jalr', 'beqz', 'bnez']):
                dependent.append(inst)
            else:
                independent.append(inst)

        random.shuffle(independent)

        # Interleave shuffled independent with dependent
        result = []
        i_idx = 0
        for inst in instructions:
            if inst in dependent:
                result.append(inst)
            else:
                if i_idx < len(independent):
                    result.append(independent[i_idx])
                    i_idx += 1

        return result

    def generate_dynamic_entry_code(self):
        """Generate obfuscated dynamic entry point code based on architecture"""
        if self.arch == 'x86_64':
            return self._generate_x86_64_entry_code()
        elif self.arch == 'arm64':
            return self._generate_arm64_entry_code()
        elif self.arch == 'riscv64':
            return self._generate_riscv64_entry_code()
        else:
            return ["    ; Unsupported architecture"]

    def _generate_x86_64_entry_code(self):
        """Generate x86-64 specific entry code"""
        # Generate random values for obfuscation
        xor_keys = [random.randint(0x1000, 0xFFFF) for _ in range(4)]
        fake_offsets = [random.randint(0x100, 0x1000) for _ in range(3)]
        threshold = random.randint(0x400, 0x600)
        rol_count = random.randint(5, 9)

        entry_code = []

        # Randomly choose register preservation order
        regs_to_save = ['rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'r8', 'r9']
        random.shuffle(regs_to_save)
        regs_to_use = regs_to_save[:4]

        # Anti-debug section with randomization
        anti_debug_methods = []

        # Method 1: RDTSC timing
        if random.choice([True, False]):
            anti_debug_methods.append([
                "    ; Timing analysis",
                "    rdtsc",
                f"    mov {random.choice(['ebx', 'r10d'])}, eax",
                f"    mov {random.choice(['ecx', 'r11d'])}, edx",
                *self.generate_junk_instructions(2),
                "    xor rax, rax",
                "    cpuid",
                *self.generate_junk_instructions(1),
                "    rdtsc",
                f"    sub eax, {random.choice(['ebx', 'r10d'])}",
                f"    sbb edx, {random.choice(['ecx', 'r11d'])}",
                f"    cmp eax, 0x{threshold:x}",
                "    ja .debugger_detected"
            ])

        # Method 2: Debug registers
        if random.choice([True, False]):
            dr_checks = []
            for dr in ['dr0', 'dr1', 'dr2', 'dr3']:
                if random.choice([True, False]):
                    dr_checks.extend([
                        f"    mov rax, {dr}",
                        "    test rax, rax",
                        "    jnz .debugger_detected"
                    ])
            if dr_checks:
                anti_debug_methods.append(["    ; Debug register inspection"] + dr_checks)

        # Method 3: INT3 scanning
        if random.choice([True, False]):
            anti_debug_methods.append([
                "    ; Scan for breakpoints",
                "    lea rsi, [rip]",
                f"    mov ecx, {random.randint(0x100, 0x200)}",
                ".scan_int3:",
                "    lodsb",
                "    cmp al, 0xCC",
                "    je .debugger_detected",
                "    loop .scan_int3"
            ])

        # Build entry code with randomization
        entry_code.append("    ; Dynamic entry point")

        # Save registers in random order
        for reg in regs_to_use:
            entry_code.append(f"    push {reg}")
            if random.choice([True, False]):
                entry_code.extend([f"    {inst}" for inst in self.generate_junk_instructions(1)])

        # Add anti-debug methods in random order
        random.shuffle(anti_debug_methods)
        for method in anti_debug_methods:
            entry_code.extend(method)
            entry_code.extend([f"    {inst}" for inst in self.generate_junk_instructions(random.randint(1, 3))])

        # Integrity check with variable implementation
        integrity_ops = [
            ("add", "add rax, rdx"),
            ("xor", "xor rax, rdx"),
            ("imul", "imul rax, rdx, 0x13"),
            ("ror", f"ror rax, {random.randint(1, 7)}"),
            ("rol", f"rol rax, {rol_count}")
        ]

        chosen_ops = random.sample(integrity_ops, 3)

        entry_code.extend([
            "    ",
            "    ; Integrity verification",
            "    lea rsi, [vmp_code_seg]",
            f"    mov rcx, {len(self.bytecode)}",
            "    xor rax, rax",
            ".checksum_loop:",
            "    movzx rdx, byte [rsi]"
        ])

        for _, op in chosen_ops:
            entry_code.append(f"    {op}")

        entry_code.extend([
            "    inc rsi",
            "    loop .checksum_loop",
            "    mov [vmp_checksum], rax"
        ])

        # Dynamic address calculation with multiple patterns
        addr_calc_patterns = []

        # Pattern 1: XOR chain
        pattern1 = [
            f"    lea rax, [{getattr(self, 'interpreter_table_name', 'vmp_interpreter_table')}]",
            f"    add rax, {getattr(self, 'real_entry_offset', 0)}  ; Real entry offset",
            "    mov rax, [rax]  ; Load actual address from table",
            f"    mov {random.choice(['rbx', 'r10'])}, 0x{xor_keys[0]:x}",
            f"    xor rax, {random.choice(['rbx', 'r10'])}",
            f"    add rax, 0x{fake_offsets[0]:x}",
            f"    sub rax, 0x{fake_offsets[0]:x}",
            f"    xor rax, {random.choice(['rbx', 'r10'])}"
        ]

        # Pattern 2: Rotate and mask
        pattern2 = [
            f"    lea rbx, [{getattr(self, 'interpreter_table_name', 'vmp_interpreter_table')}]",
            f"    add rbx, {getattr(self, 'real_entry_offset', 0)}",
            "    mov rax, [rbx]  ; Load from table",
            f"    ror rax, {random.randint(1, 16)}",
            f"    mov rcx, 0x{xor_keys[1]:x}",
            "    xor rax, rcx",
            f"    rol rax, {random.randint(1, 16)}",
            "    xor rax, rcx"
        ]

        # Pattern 3: Mathematical operations
        pattern3 = [
            f"    lea rcx, [{getattr(self, 'interpreter_table_name', 'vmp_interpreter_table')}]",
            f"    mov rax, [rcx + {getattr(self, 'real_entry_offset', 0)}]  ; Direct offset load",
            f"    mov rdx, 0x{random.randint(3, 17):x}",
            "    mov rbx, rax",
            "    mul rdx",
            "    mov rax, rbx",
            "    div rdx"
        ]

        chosen_pattern = random.choice([pattern1, pattern2, pattern3])
        entry_code.append("    ")
        entry_code.append("    ; Address calculation")
        entry_code.extend(chosen_pattern)

        # Add more obfuscation layers
        for i in range(random.randint(1, 3)):
            entry_code.extend([
                f"    mov {random.choice(['rcx', 'rdx', 'r8'])}, 0x{xor_keys[i]:x}",
                "    not rcx",
                "    and rax, rcx",
                "    not rcx",
                "    or rax, rcx",
                "    xor rax, rcx"
            ])
            entry_code.extend([f"    {inst}" for inst in self.generate_junk_instructions(2)])

        # Setup VM context with variations
        context_setup = [
            "    ; VM context initialization",
            *self.generate_obfuscated_mov('rsi', '[vmp_code_seg]', False),
            *self.generate_obfuscated_mov('rdi', '[vmp_registers]', False),
            *self.generate_obfuscated_mov('rdx', '[vmp_data_seg]', False),
            "    cld"
        ]

        # Save context with random additional pushes
        context_save = [
            "    push rbp",
            "    mov rbp, rsp"
        ]

        if random.choice([True, False]):
            context_save.extend([
                "    push r12",
                "    push r13",
                "    push r14",
                "    push r15"
            ])

        context_save.append(f"    sub rsp, 0x{random.randint(0x80, 0x200):x}")

        entry_code.extend(context_setup)
        entry_code.extend(context_save)

        # Generate different jump methods
        jump_methods = [
            # Method 1: RET
            [
                "    push rax",
                "    ret"
            ],
            # Method 2: JMP through register
            [
                "    jmp rax"
            ],
            # Method 3: CALL and adjust
            [
                "    call rax",
                "    jmp .after_interpreter",
                ".after_interpreter:"
            ]
        ]

        chosen_jump = random.choice(jump_methods)
        entry_code.append("    ")
        entry_code.append("    ; Transfer control")
        entry_code.extend(chosen_jump)

        # Return sequence
        return_label = f".vm_return_{random.randint(1000, 9999)}:"
        entry_code.extend([
            "    ",
            return_label,
            "    mov rsp, rbp",
            "    pop rbp"
        ])

        # Restore registers in reverse order
        for reg in reversed(regs_to_use):
            entry_code.append(f"    pop {reg}")

        entry_code.append("    ret")

        # Debugger detected handler with variations
        crash_methods = [
            ["ud2"],
            ["int3", "nop", "jmp $-2"],
            ["xor rsp, rsp", "ret"],
            ["mov rax, 0", "div rax"]
        ]

        entry_code.extend([
            "    ",
            ".debugger_detected:",
            "    ; Anti-tampering response"
        ])

        # Clear sensitive registers
        for reg in ['rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi']:
            entry_code.append(f"    xor {reg}, {reg}")

        entry_code.extend(random.choice(crash_methods))

        # Shuffle independent sections
        return self.shuffle_with_dependencies(entry_code)

    def _generate_arm64_entry_code(self):
        """Generate ARM64 specific entry code"""
        entry_code = []

        # ARM64 specific values
        xor_keys = [random.randint(0x1000, 0xFFFF) for _ in range(4)]

        entry_code.append("    ; ARM64 Dynamic entry point")

        # Save registers
        regs_to_save = ['x19', 'x20', 'x21', 'x22', 'x23', 'x24', 'x25', 'x26']
        random.shuffle(regs_to_save)
        regs_to_use = regs_to_save[:4]

        # Prologue
        entry_code.extend([
            "    stp x29, x30, [sp, #-16]!",  # Save frame pointer and link register
            "    mov x29, sp"
        ])

        # Save callee-saved registers
        for i in range(0, len(regs_to_use), 2):
            if i + 1 < len(regs_to_use):
                entry_code.append(f"    stp {regs_to_use[i]}, {regs_to_use[i + 1]}, [sp, #-16]!")
            else:
                entry_code.append(f"    str {regs_to_use[i]}, [sp, #-8]!")

        # Anti-debug checks (ARM64 specific)
        if random.choice([True, False]):
            entry_code.extend([
                "    ; Check for debugger",
                "    mrs x0, mdscr_el1",  # Read debug state register
                "    tbnz x0, #0, .debugger_detected"  # Test bit 0
            ])

        # Integrity check
        entry_code.extend([
            "    ; Integrity verification",
            "    adrp x0, vmp_code_seg",
            "    add x0, x0, :lo12:vmp_code_seg",
            f"    mov x1, #{len(self.bytecode)}",
            "    mov x2, #0",
            ".checksum_loop:",
            "    ldrb w3, [x0], #1",
            "    add x2, x2, x3",
            "    subs x1, x1, #1",
            "    b.ne .checksum_loop",
            "    adrp x0, vmp_checksum",
            "    str x2, [x0, :lo12:vmp_checksum]"
        ])

        # Load interpreter address
        entry_code.extend([
            "    ; Load interpreter",
            f"    adrp x0, {getattr(self, 'interpreter_table_name', 'vmp_interpreter_table')}",
            f"    add x0, x0, :lo12:{getattr(self, 'interpreter_table_name', 'vmp_interpreter_table')}",
            f"    ldr x0, [x0, #{getattr(self, 'real_entry_offset', 0)}]"
        ])

        # Setup VM context
        entry_code.extend([
            "    ; VM context setup",
            "    adrp x1, vmp_code_seg",
            "    add x1, x1, :lo12:vmp_code_seg",
            "    adrp x2, vmp_registers",
            "    add x2, x2, :lo12:vmp_registers",
            "    adrp x3, vmp_data_seg",
            "    add x3, x3, :lo12:vmp_data_seg"
        ])

        # Jump to interpreter
        entry_code.extend([
            "    ; Transfer control",
            "    blr x0"
        ])

        # Epilogue
        for i in range(len(regs_to_use) - 1, -1, -2):
            if i > 0:
                entry_code.append(f"    ldp {regs_to_use[i - 1]}, {regs_to_use[i]}, [sp], #16")
            else:
                entry_code.append(f"    ldr {regs_to_use[i]}, [sp], #8")

        entry_code.extend([
            "    ldp x29, x30, [sp], #16",
            "    ret"
        ])

        # Debugger detected handler
        entry_code.extend([
            "    ",
            ".debugger_detected:",
            "    ; Anti-tampering response",
            "    mov x0, #0",
            "    mov x1, #0",
            "    mov x2, #0",
            "    mov x3, #0",
            "    brk #0"  # ARM64 breakpoint instruction
        ])

        return entry_code

    def _generate_riscv64_entry_code(self):
        """Generate RISC-V64 specific entry code"""
        entry_code = []

        entry_code.append("    ; RISC-V64 Dynamic entry point")

        # Save registers
        regs_to_save = ['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7']
        random.shuffle(regs_to_save)
        regs_to_use = regs_to_save[:4]

        # Prologue
        stack_size = 8 * (len(regs_to_use) + 2)  # +2 for ra and fp
        entry_code.extend([
            f"    addi sp, sp, -{stack_size}",
            f"    sd ra, {stack_size - 8}(sp)",
            f"    sd fp, {stack_size - 16}(sp)",
            "    addi fp, sp, {stack_size}"
        ])

        # Save callee-saved registers
        for i, reg in enumerate(regs_to_use):
            entry_code.append(f"    sd {reg}, {i * 8}(sp)")

        # Integrity check
        entry_code.extend([
            "    ; Integrity verification",
            "    la t0, vmp_code_seg",
            f"    li t1, {len(self.bytecode)}",
            "    li t2, 0",
            ".checksum_loop:",
            "    lbu t3, 0(t0)",
            "    add t2, t2, t3",
            "    addi t0, t0, 1",
            "    addi t1, t1, -1",
            "    bnez t1, .checksum_loop",
            "    la t0, vmp_checksum",
            "    sd t2, 0(t0)"
        ])

        # Load interpreter address
        entry_code.extend([
            "    ; Load interpreter",
            f"    la t0, {getattr(self, 'interpreter_table_name', 'vmp_interpreter_table')}",
            f"    ld t0, {getattr(self, 'real_entry_offset', 0)}(t0)"
        ])

        # Setup VM context
        entry_code.extend([
            "    ; VM context setup",
            "    la a0, vmp_code_seg",
            "    la a1, vmp_registers",
            "    la a2, vmp_data_seg"
        ])

        # Jump to interpreter
        entry_code.extend([
            "    ; Transfer control",
            "    jalr t0"
        ])

        # Epilogue
        for i, reg in enumerate(regs_to_use):
            entry_code.append(f"    ld {reg}, {i * 8}(sp)")

        entry_code.extend([
            f"    ld ra, {stack_size - 8}(sp)",
            f"    ld fp, {stack_size - 16}(sp)",
            f"    addi sp, sp, {stack_size}",
            "    ret"
        ])

        return entry_code

    def generate_vmp_text(self, assembly_code):
        """Generate VMP protected assembly text representation"""
        bytecode, variables = self.convert_assembly(assembly_code)

        # Generate text representation
        vmp_text = f"; VMP Protected Assembly ({self.arch})\n"
        vmp_text += "; Generated by Advanced VMP Transformer\n"
        vmp_text += "; Warning: This code is protected against debugging and tampering\n"

        if self.function_name:
            vmp_text += f"; Original function: {self.function_name}\n"

        vmp_text += "\n"

        # Data section
        vmp_text += "section .data\n"
        vmp_text += f"    vmp_code_seg db {', '.join(str(b) for b in bytecode)}"
        if len(bytecode) > 50:
            vmp_text += f" ; {len(bytecode)} bytes total\n"
        else:
            vmp_text += "\n"

        vmp_text += f"    vmp_data_seg times {self.data_offset} db 0\n"

        # Register storage
        vmp_text += "    ; Register storage area\n"
        if self.arch == 'x86_64':
            vmp_text += "    vmp_registers times 40 dq 0  ; 40 registers * 8 bytes\n"
        elif self.arch == 'arm64':
            vmp_text += "    vmp_registers times 67 dq 0  ; ARM64 registers\n"
        elif self.arch == 'riscv64':
            vmp_text += "    vmp_registers times 32 dq 0  ; RISC-V registers\n"

        # Additional data for protection
        vmp_text += "    ; Protection data\n"
        vmp_text += "    vmp_checksum dq 0\n"

        # Generate dynamic interpreter table
        vmp_text += self.generate_dynamic_interpreter_table()

        # Variable mapping
        if variables:
            vmp_text += "\n; Variable offsets:\n"
            for var, offset in variables.items():
                vmp_text += f";   {var}: offset {offset}\n"

        # Text section with dynamic entry
        vmp_text += "\nsection .text\n"
        vmp_text += "global vmp_protected_entry\n"
        vmp_text += "\nvmp_protected_entry:\n"

        # Add the dynamic entry code
        entry_code = self.generate_dynamic_entry_code()
        for line in entry_code:
            vmp_text += line + "\n"

        # Generate dynamic interpreter stub
        vmp_text += self.generate_interpreter_stub()

        return vmp_text

    def generate_interpreter_stub(self):
        """Generate randomized interpreter implementation stub based on architecture"""
        if self.arch == 'x86_64':
            return self._generate_x86_64_interpreter_stub()
        elif self.arch == 'arm64':
            return self._generate_arm64_interpreter_stub()
        elif self.arch == 'riscv64':
            return self._generate_riscv64_interpreter_stub()
        else:
            return "\n; Unsupported architecture for interpreter stub\n"

    def _generate_x86_64_interpreter_stub(self):
        """Generate x86-64 specific interpreter stub"""
        stub_code = []

        # Random label names
        label_suffix = random.randint(1000, 9999)
        entry_label = f"vmp_interpreter_impl_{label_suffix}:"
        loop_label = f".vm_loop_{random.randint(100, 999)}:"
        decode_label = f".decode_op_{random.randint(100, 999)}:"
        execute_label = f".execute_{random.randint(100, 999)}:"
        error_label = f".vm_error_{random.randint(100, 999)}:"

        stub_code.append("\n; Dynamic interpreter implementation")
        stub_code.append(entry_label)

        # Random register allocation for VM state
        vm_regs = {
            'ip': random.choice(['r8', 'r9', 'r10']),
            'stack': random.choice(['r11', 'r12', 'r13']),
            'scratch1': random.choice(['r14', 'r15', 'rbx']),
            'scratch2': random.choice(['rax', 'rcx', 'rdx'])
        }

        # Initialize VM state with obfuscation
        init_code = [
            "    ; Initialize VM state",
            f"    mov {vm_regs['ip']}, rsi  ; Instruction pointer",
            f"    mov {vm_regs['stack']}, rdi  ; Register storage"
        ]

        # Add random initialization patterns
        if random.choice([True, False]):
            init_code.extend([
                f"    xor {vm_regs['scratch1']}, {vm_regs['scratch1']}",
                f"    xor {vm_regs['scratch2']}, {vm_regs['scratch2']}"
            ])

        stub_code.extend(init_code)
        stub_code.extend(self.generate_junk_instructions(random.randint(2, 4)))

        # Main interpreter loop structure
        loop_patterns = []

        # Pattern 1: Standard loop
        pattern1 = [
            loop_label,
            "    ; Fetch opcode seeds",
            f"    mov eax, [{vm_regs['ip']}]  ; Opcode seed",
            f"    mov ebx, [{vm_regs['ip']} + 4]  ; Code seed",
            f"    add {vm_regs['ip']}, 8",
            "    ",
            "    ; Decode opcode",
            f"    push {vm_regs['ip']}",
            f"    push {vm_regs['stack']}",
            f"    pop {vm_regs['stack']}",
            f"    pop {vm_regs['ip']}",
            "    ",
            f"    jmp {loop_label}"
        ]

        # Pattern 2: Unrolled loop
        pattern2 = [
            loop_label,
            "    ; Fetch and decode inline"
        ]
        for i in range(random.randint(2, 4)):
            pattern2.extend([
                f"    ; Iteration {i}",
                f"    movzx eax, byte [{vm_regs['ip']} + {i * 8}]",
                f"    movzx ebx, byte [{vm_regs['ip']} + {i * 8 + 4}]",
            ])
        pattern2.extend([
            f"    add {vm_regs['ip']}, {(i + 1) * 8}",
            f"    jmp {loop_label}"
        ])

        # Pattern 3: Computed goto
        pattern3 = [
            loop_label,
            "    ; Computed dispatch",
            f"    movzx eax, byte [{vm_regs['ip']}]",
            f"    lea {vm_regs['scratch1']}, [.dispatch_table]",
            f"    jmp [{vm_regs['scratch1']} + rax*8]",
            "    ",
            ".dispatch_table:",
            "    ; Jump table entries would go here"
        ]

        chosen_pattern = random.choice([pattern1, pattern2])
        stub_code.extend(chosen_pattern)

        # Add error handling
        error_patterns = [
            [
                error_label,
                "    ; VM error handler",
                "    xor rax, rax",
                "    dec rax  ; Return -1",
                "    jmp .vm_exit"
            ],
            [
                error_label,
                "    ; Exception handler",
                "    mov rax, 0xDEADBEEF",
                "    jmp .vm_exit"
            ]
        ]

        stub_code.extend(random.choice(error_patterns))

        # Add exit handler with cleanup
        exit_code = [
            "    ",
            ".vm_exit:",
            "    ; Cleanup VM state"
        ]

        # Random cleanup operations
        if random.choice([True, False]):
            exit_code.extend([
                f"    xor {vm_regs['ip']}, {vm_regs['ip']}",
                f"    xor {vm_regs['stack']}, {vm_regs['stack']}"
            ])

        # Different return methods
        return_methods = [
            ["    ret"],
            ["    jmp [rsp]"],
            ["    pop rcx", "    jmp rcx"]
        ]

        exit_code.extend(random.choice(return_methods))
        stub_code.extend(exit_code)

        # Add some fake/dead code sections
        if random.choice([True, False]):
            stub_code.extend([
                "    ",
                f".dead_code_{random.randint(100, 999)}:",
                "    ; Unreachable code for obfuscation"
            ])
            stub_code.extend(self.generate_junk_instructions(random.randint(5, 10)))

        return '\n'.join(stub_code)

    def _generate_arm64_interpreter_stub(self):
        """Generate ARM64 specific interpreter stub"""
        stub_code = []

        # Random label names
        label_suffix = random.randint(1000, 9999)
        entry_label = f"vmp_interpreter_impl_{label_suffix}:"
        loop_label = f".vm_loop_{random.randint(100, 999)}:"

        stub_code.append("\n; ARM64 interpreter implementation")
        stub_code.append(entry_label)

        # VM state registers
        vm_regs = {
            'ip': 'x19',  # Instruction pointer
            'stack': 'x20',  # Register storage
            'scratch1': 'x21',
            'scratch2': 'x22'
        }

        # Initialize
        stub_code.extend([
            "    ; Initialize VM state",
            f"    mov {vm_regs['ip']}, x1",
            f"    mov {vm_regs['stack']}, x2",
            f"    mov {vm_regs['scratch1']}, #0",
            f"    mov {vm_regs['scratch2']}, #0"
        ])

        # Main loop
        stub_code.extend([
            loop_label,
            "    ; Fetch opcode seeds",
            f"    ldr w8, [{vm_regs['ip']}]",
            f"    ldr w9, [{vm_regs['ip']}, #4]",
            f"    add {vm_regs['ip']}, {vm_regs['ip']}, #8",
            "    ",
            "    ; Decode and dispatch",
            "    ; (Implementation would go here)",
            "    ",
            f"    b {loop_label}"
        ])

        # Exit
        stub_code.extend([
            "    ",
            ".vm_exit:",
            "    ret"
        ])

        return '\n'.join(stub_code)

    def _generate_riscv64_interpreter_stub(self):
        """Generate RISC-V64 specific interpreter stub"""
        stub_code = []

        # Random label names
        label_suffix = random.randint(1000, 9999)
        entry_label = f"vmp_interpreter_impl_{label_suffix}:"
        loop_label = f".vm_loop_{random.randint(100, 999)}:"

        stub_code.append("\n; RISC-V64 interpreter implementation")
        stub_code.append(entry_label)

        # VM state registers
        vm_regs = {
            'ip': 's0',  # Instruction pointer
            'stack': 's1',  # Register storage
            'scratch1': 's2',
            'scratch2': 's3'
        }

        # Initialize
        stub_code.extend([
            "    ; Initialize VM state",
            f"    mv {vm_regs['ip']}, a0",
            f"    mv {vm_regs['stack']}, a1",
            f"    li {vm_regs['scratch1']}, 0",
            f"    li {vm_regs['scratch2']}, 0"
        ])

        # Main loop
        stub_code.extend([
            loop_label,
            "    ; Fetch opcode seeds",
            f"    lw t0, 0({vm_regs['ip']})",
            f"    lw t1, 4({vm_regs['ip']})",
            f"    addi {vm_regs['ip']}, {vm_regs['ip']}, 8",
            "    ",
            "    ; Decode and dispatch",
            "    ; (Implementation would go here)",
            "    ",
            f"    j {loop_label}"
        ])

        # Exit
        stub_code.extend([
            "    ",
            ".vm_exit:",
            "    ret"
        ])

        return '\n'.join(stub_code)

    def detect_architecture(self, assembly_code):
        """Auto-detect architecture from assembly code"""
        # Simple heuristic based on register names and instruction patterns
        if re.search(r'%[re]?[abcd]x|%r[0-9]+|mov[lqwb]|push|pop', assembly_code):
            return 'x86_64'
        elif re.search(r'\b[xw][0-9]+\b|ldr|str|bl|adrp', assembly_code):
            return 'arm64'
        elif re.search(r'\b[xt][0-9]+\b|addi|ld|sd|jal|li', assembly_code):
            return 'riscv64'
        else:
            return 'x86_64'  # Default


# Example usage with architecture detection
def convert_with_auto_detect(assembly_code):
    """Convert assembly with automatic architecture detection"""
    converter = AssemblyToVMPConverter()

    # Detect architecture
    detected_arch = converter.detect_architecture(assembly_code)

    # Reinitialize with detected architecture
    converter = AssemblyToVMPConverter(arch=detected_arch)

    # Convert
    return converter.generate_vmp_text(assembly_code)


