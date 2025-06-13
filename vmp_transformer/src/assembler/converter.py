# src/assembler/converter.py

import random
import re
from ..core.opcodes import *
from ..core.crypto import XorShift32


class AssemblyToVMPConverter:
    """Convert assembly code to VMP protected bytecode"""

    def __init__(self):
        self.bytecode = []
        self.data_segment = []
        self.labels = {}
        self.variables = {}
        self.current_offset = 0
        self.data_offset = 0
        self.function_name = None

        # x86-64 register mapping
        self.registers = {
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
        }

        # Register size mapping
        self.register_sizes = {
            'r': 8, 'e': 4, '': 2, 'l': 1, 'h': 1
        }

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
        """Parse an x86 operand and return its components"""
        operand = operand.strip()

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

        # Convert x86-64 instructions
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

    def generate_vmp_text(self, assembly_code):
        """Generate VMP protected assembly text representation"""
        bytecode, variables = self.convert_assembly(assembly_code)

        # Generate text representation
        vmp_text = "; VMP Protected Assembly\n"
        vmp_text += "; Generated by VMP Transformer\n"

        if self.function_name:
            vmp_text += f"; Original function: {self.function_name}\n"

        vmp_text += "\n"

        # Data section
        vmp_text += "section .data\n"
        vmp_text += f"    vmp_code_seg db {', '.join(str(b) for b in bytecode[:20])}"
        if len(bytecode) > 20:
            vmp_text += f", ... ; {len(bytecode)} bytes total\n"
        else:
            vmp_text += "\n"

        vmp_text += f"    vmp_data_seg times {self.data_offset} db 0\n"

        # Register storage
        vmp_text += "    ; Register storage area\n"
        vmp_text += "    vmp_registers times 40 dq 0  ; 40 registers * 8 bytes\n\n"

        # Variable mapping
        if variables:
            vmp_text += "; Variable offsets:\n"
            for var, offset in variables.items():
                vmp_text += f";   {var}: offset {offset}\n"

        vmp_text += "\nsection .text\n"
        vmp_text += "    ; VMP interpreter entry point\n"
        vmp_text += "    call vmp_interpreter\n"
        vmp_text += "    ; Result in rax register\n"

        return vmp_text


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

    # Convert x86-64 instructions
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


def generate_vmp_text(self, assembly_code):
    """Generate VMP protected assembly text representation"""
    bytecode, variables = self.convert_assembly(assembly_code)

    # Generate text representation
    vmp_text = "; VMP Protected Assembly\n"
    vmp_text += "; Generated by VMP Transformer\n"

    if self.function_name:
        vmp_text += f"; Original function: {self.function_name}\n"

    vmp_text += "\n"

    # Data section
    vmp_text += "section .data\n"
    vmp_text += f"    vmp_code_seg db {', '.join(str(b) for b in bytecode[:20])}"
    if len(bytecode) > 20:
        vmp_text += f", ... ; {len(bytecode)} bytes total\n"
    else:
        vmp_text += "\n"

    vmp_text += f"    vmp_data_seg times {self.data_offset} db 0\n"

    # Register storage
    vmp_text += "    ; Register storage area\n"
    vmp_text += "    vmp_registers times 40 dq 0  ; 40 registers * 8 bytes\n\n"

    # Variable mapping
    if variables:
        vmp_text += "; Variable offsets:\n"
        for var, offset in variables.items():
            vmp_text += f";   {var}: offset {offset}\n"

    vmp_text += "\nsection .text\n"
    vmp_text += "    ; VMP interpreter entry point\n"
    vmp_text += "    call vmp_interpreter\n"
    vmp_text += "    ; Result in rax register\n"

    return vmp_text


