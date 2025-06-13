# src/assembler/parser.py

import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Union


@dataclass
class Instruction:
    """Represents a parsed assembly instruction"""
    mnemonic: str
    operands: List[str]
    line_number: int
    label: Optional[str] = None
    comment: Optional[str] = None


@dataclass
class ParsedAssembly:
    """Represents parsed assembly program"""
    instructions: List[Instruction]
    labels: Dict[str, int]
    data_section: Dict[str, Union[int, str, bytes]]


class AssemblyParser:
    """Parse assembly code into structured format"""

    def __init__(self):
        # Common x86/x64 instructions
        self.valid_mnemonics = {
            # Data transfer
            'mov', 'movzx', 'movsx', 'lea', 'push', 'pop', 'movabs',
            # Arithmetic
            'add', 'sub', 'mul', 'imul', 'div', 'idiv', 'inc', 'dec', 'neg',
            # Logic
            'and', 'or', 'xor', 'not', 'shl', 'shr', 'sal', 'sar', 'rol', 'ror',
            # Control flow
            'jmp', 'je', 'jne', 'jz', 'jnz', 'jg', 'jge', 'jl', 'jle', 'ja', 'jae', 'jb', 'jbe',
            'call', 'ret', 'retq', 'loop', 'leave',
            # Comparison
            'cmp', 'test',
            # Other
            'nop', 'int', 'syscall', 'endbr64', 'endbr32'
        }

        # Register names (Intel syntax)
        self.intel_registers = {
            # 64-bit
            'rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rbp', 'rsp',
            'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15',
            # 32-bit
            'eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'ebp', 'esp',
            'r8d', 'r9d', 'r10d', 'r11d', 'r12d', 'r13d', 'r14d', 'r15d',
            # 16-bit
            'ax', 'bx', 'cx', 'dx', 'si', 'di', 'bp', 'sp',
            'r8w', 'r9w', 'r10w', 'r11w', 'r12w', 'r13w', 'r14w', 'r15w',
            # 8-bit
            'al', 'ah', 'bl', 'bh', 'cl', 'ch', 'dl', 'dh',
            'r8b', 'r9b', 'r10b', 'r11b', 'r12b', 'r13b', 'r14b', 'r15b'
        }

        # AT&T syntax registers (with % prefix)
        self.att_registers = {'%' + reg for reg in self.intel_registers}

        # Combined register set
        self.registers = self.intel_registers | self.att_registers

    # def preprocess_assembly(self, asm_code: str) -> str:
    #     """Preprocess assembly code to remove header text before actual code"""
    #     # 修复：不要强制分割，让代码保持原样
    #     # 只有在明确有双换行符且第一部分看起来像是头部信息时才分割
    #     if '\n\n' in asm_code:
    #         parts = asm_code.split('\n\n', 1)
    #         # 检查第一部分是否包含汇编指令
    #         first_part_lines = parts[0].strip().split('\n')
    #         has_asm_instructions = False
    #
    #         for line in first_part_lines:
    #             line = line.strip()
    #             if line and not line.startswith('#') and not line.startswith('//'):
    #                 # 检查是否包含汇编指令或标签
    #                 if ':' in line or any(line.lower().startswith(mnem) for mnem in self.valid_mnemonics):
    #                     has_asm_instructions = True
    #                     break
    #
    #         # 如果第一部分没有汇编指令，才使用第二部分
    #         if not has_asm_instructions and len(parts) > 1:
    #             asm_code = parts[1]
    #
    #     return asm_code.strip()

    def preprocess_assembly(self, asm_code: str) -> str:
        """Preprocess assembly code to remove header text before actual code"""
        # Remove content before double newline
        if '\n\n' in asm_code:
            parts = asm_code.split('\n\n', 1)
            if len(parts) > 1:
                asm_code = parts[1]
                print(asm_code)

        return asm_code.strip()

    def parse_line(self, line: str, line_number: int) -> Optional[Instruction]:
        """Parse single line of assembly"""
        # Remove comments
        comment = None
        if ';' in line:
            line, comment = line.split(';', 1)
            comment = comment.strip()

        line = line.strip()
        if not line:
            return None

        # Check for function label format <function_name>:
        func_label_match = re.match(r'<(.+?)>:', line)
        if func_label_match:
            label = func_label_match.group(1)
            return Instruction(
                mnemonic='label',
                operands=[],
                line_number=line_number,
                label=label,
                comment=comment
            )

        # Check for regular label
        label = None
        if ':' in line and not line.startswith(':'):
            # 修复：确保标签解析正确
            colon_pos = line.find(':')
            potential_label = line[:colon_pos].strip()
            # 检查是否是有效的标签（不包含空格或特殊字符）
            if potential_label and ' ' not in potential_label and '\t' not in potential_label:
                label = potential_label
                line = line[colon_pos + 1:].strip()

                if not line:
                    # Label only line
                    return Instruction(
                        mnemonic='label',
                        operands=[],
                        line_number=line_number,
                        label=label,
                        comment=comment
                    )

        # Parse instruction
        tokens = self.tokenize_instruction(line)

        if not tokens:
            return None

        # 修复：添加错误处理
        if len(tokens) == 0:
            return None
        mnemonic = tokens[0].lower()
        operands = tokens[1:] if len(tokens) > 1 else []
        return Instruction(
            mnemonic=mnemonic,
            operands=operands,
            line_number=line_number,
            label=label,
            comment=comment
        )

    def tokenize_instruction(self, line: str) -> List[str]:
        """Tokenize instruction line handling complex operands"""
        # Handle parentheses and brackets as part of operands
        tokens = []
        current_token = ''
        in_parens = 0
        in_brackets = 0

        i = 0
        while i < len(line):
            char = line[i]

            if char == '(':
                in_parens += 1
                current_token += char
            elif char == ')':
                in_parens -= 1
                current_token += char
            elif char == '[':
                in_brackets += 1
                current_token += char
            elif char == ']':
                in_brackets -= 1
                current_token += char
            elif char in ' \t' and in_parens == 0 and in_brackets == 0:
                # 修复：空格和制表符只作为分隔符，不包括逗号
                if current_token:
                    tokens.append(current_token)
                    current_token = ''
            elif char == ',' and in_parens == 0 and in_brackets == 0:
                # 修复：逗号作为操作数之间的分隔符
                if current_token:
                    tokens.append(current_token)
                    current_token = ''
                # 跳过逗号本身
            else:
                current_token += char

            i += 1

        if current_token:
            tokens.append(current_token)

        return tokens

    def parse_operand(self, operand: str) -> dict:
        """Parse operand to determine its type"""
        operand = operand.strip()

        # Function call reference <function@plt>
        if operand.startswith('<') and operand.endswith('>'):
            return {'type': 'function_ref', 'name': operand[1:-1]}

        # Immediate value with $ prefix (AT&T syntax)
        if operand.startswith('$'):
            value_str = operand[1:]
            if value_str.startswith('0x'):
                return {'type': 'immediate', 'value': int(value_str, 16)}
            elif value_str.isdigit() or (value_str.startswith('-') and value_str[1:].isdigit()):
                return {'type': 'immediate', 'value': int(value_str)}

        # Immediate value (Intel syntax)
        if operand.isdigit() or (operand.startswith('-') and operand[1:].isdigit()):
            return {'type': 'immediate', 'value': int(operand)}

        # Hex immediate
        if operand.startswith('0x') or operand.startswith('0X'):
            return {'type': 'immediate', 'value': int(operand, 16)}

        # Register
        if operand in self.registers:
            reg_name = operand[1:] if operand.startswith('%') else operand
            return {'type': 'register', 'name': reg_name, 'att_syntax': operand.startswith('%')}

        # Memory reference with displacement and register
        # AT&T syntax: -0x4(%rbp) or 0x10(%rax,%rbx,4)
        # Intel syntax: [rbp-0x4] or [rax+rbx*4+0x10]
        if '(' in operand and ')' in operand:
            return self.parse_att_memory_operand(operand)
        elif operand.startswith('[') and operand.endswith(']'):
            inner = operand[1:-1].strip()
            return {'type': 'memory', 'address': self.parse_address(inner)}

        # Label/symbol
        return {'type': 'label', 'name': operand}

    def parse_att_memory_operand(self, operand: str) -> dict:
        """Parse AT&T style memory operand"""
        # Extract displacement and register parts
        match = re.match(r'(-?[0-9a-fA-Fx]*)\((.+)\)', operand)
        if not match:
            return {'type': 'memory', 'expression': operand}

        displacement = match.group(1)
        register_part = match.group(2)

        result = {'type': 'memory'}

        # Parse displacement
        if displacement:
            if displacement.startswith('0x'):
                result['displacement'] = int(displacement, 16)
            elif displacement.lstrip('-').isdigit():
                result['displacement'] = int(displacement)

        # Parse register part (base,index,scale)
        reg_parts = register_part.split(',')
        if len(reg_parts) >= 1 and reg_parts[0]:
            result['base'] = reg_parts[0].lstrip('%')
        if len(reg_parts) >= 2 and reg_parts[1]:
            result['index'] = reg_parts[1].lstrip('%')
        if len(reg_parts) >= 3 and reg_parts[2]:
            result['scale'] = int(reg_parts[2])

        return result

    def parse_address(self, address_expr: str) -> dict:
        """Parse memory address expression (Intel syntax)"""
        # Simple cases
        if address_expr in self.intel_registers:
            return {'base': address_expr}

        # More complex addressing modes would be parsed here
        # For now, return as string
        return {'expression': address_expr}

    def parse(self, assembly_code: str) -> ParsedAssembly:
        """Parse complete assembly program"""
        # Preprocess to remove header text
        assembly_code = self.preprocess_assembly(assembly_code)

        lines = assembly_code.strip().split('\n')
        instructions = []
        labels = {}
        data_section = {}

        current_section = 'text'
        instruction_index = 0

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # Check for section directive
            if line.startswith('section') or line.startswith('.section'):
                if 'data' in line:
                    current_section = 'data'
                elif 'text' in line or 'code' in line:
                    current_section = 'text'
                continue

            if current_section == 'text':
                try:
                    instruction = self.parse_line(line, line_num)
                    if instruction:
                        if instruction.label:
                            labels[instruction.label] = instruction_index
                        if instruction.mnemonic != 'label':
                            instructions.append(instruction)
                            instruction_index += 1
                except Exception as e:
                    print(f"Error parsing line {line_num}: {line}")
                    print(f"Error details: {e}")
                    continue

            elif current_section == 'data':
                # Parse data declarations
                if line and not line.startswith(';'):
                    self.parse_data_declaration(line, data_section)

        return ParsedAssembly(
            instructions=instructions,
            labels=labels,
            data_section=data_section
        )

    def parse_data_declaration(self, line: str, data_section: dict):
        """Parse data section declarations"""
        # Simple parsing for common directives
        tokens = line.split()
        if len(tokens) >= 3:
            name = tokens[0]
            directive = tokens[1].lower()

            if directive in ['db', 'dw', 'dd', 'dq']:
                # Byte/word/dword/qword data
                values = ' '.join(tokens[2:]).split(',')
                data_section[name] = [self.parse_data_value(v.strip()) for v in values]
            elif directive == 'equ':
                # Constant
                data_section[name] = self.parse_data_value(tokens[2])

    def parse_data_value(self, value: str) -> Union[int, str]:
        """Parse data value"""
        value = value.strip()

        # String
        if value.startswith('"') and value.endswith('"'):
            return value[1:-1]

        # Hex
        if value.startswith('0x') or value.startswith('0X'):
            return int(value, 16)

        # Decimal
        if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            return int(value)

        # Character
        if value.startswith("'") and value.endswith("'"):
            return ord(value[1])

        return value


