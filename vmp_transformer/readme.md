# VMP Transformer

A Python implementation of Virtual Machine Protection (VMP) for transforming assembly code into VMP-protected bytecode.

## Overview

This project provides a Python-based VMP interpreter and transformer that can:
- Convert normal assembly code (text) into VMP-protected assembly code (text)
- Implement core VMP features including opcode encryption, instruction virtualization, and control flow obfuscation
- Support integration with LLM-based compiler systems

## Features

- **Opcode Encryption**: XorShift32-based opcode obfuscation
- **Instruction Support**: ALLOCA, LOAD, STORE, Binary Operations, GEP, CMP, CAST, BR, CALL, RET
- **Memory Management**: Separate code and data segments
- **Text-to-Text Transformation**: Convert assembly text to VMP-protected assembly text
- **Extensible Architecture**: Easy to add new instructions and protection mechanisms

## Installation

```bash
git clone https://github.com/dylanyunlong/vmp_transformer.git
cd vmp_transformer
pip install -r requirements.txt
```

## Quick Start

```python
from src.api.transformer import transform_assembly_to_vmp

# Your assembly code
assembly_code = """
mov eax, 10
add eax, 20
ret eax
"""

# Transform to VMP
result = transform_assembly_to_vmp(assembly_code)

if 'error' not in result:
    print("VMP Protected Assembly:")
    print(result['vmp_assembly'])
else:
    print(f"Error: {result['error']}")
```

## Architecture

```
vmp_transformer/
├── src/
│   ├── core/           # VMP interpreter core
│   ├── assembler/      # Assembly parsing and conversion
│   ├── resources/      # Instruction sets and templates
│   └── api/            # Public API
└── examples/           # Usage examples
```

## Usage Examples

### 1. Simple Transformation

```python
from src.api.transformer import VMPTransformer

transformer = VMPTransformer()
result = transformer.transform_assembly("""
    mov eax, 5
    add eax, 3
    ret eax
""")

print(result['vmp_assembly'])
```

### 2. Batch Processing

```python
files = [
    ("file1.asm", "mov eax, 1\nret"),
    ("file2.asm", "add eax, ebx\nret")
]

results = transformer.batch_transform(files)
```

### 3. With Analysis

```python
transformer = VMPTransformer(debug=True)
result = transformer.transform_assembly(assembly_code)

# Analyze execution
analysis = transformer.analyze_vmp_code(result['bytecode'])
print(analysis['trace'])
```

## API Reference

### VMPTransformer

Main class for assembly transformation.

#### Methods

- `transform_assembly(assembly_code: str) -> dict`
  - Transform assembly code to VMP protected format
  - Returns: `{'vmp_assembly': str, 'bytecode': bytes, 'metadata': dict}`

- `analyze_vmp_code(bytecode: bytes, initial_data: bytes = None) -> dict`
  - Analyze VMP bytecode execution
  - Returns: `{'success': bool, 'trace': list, 'final_data': bytes}`

- `batch_transform(assembly_files: list) -> dict`
  - Transform multiple files
  - Returns: Dictionary with results for each file

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
black src/
flake8 src/
mypy src/
```

## Integration with LLM

This transformer is designed to work with LLM-generated assembly code:

1. LLM generates assembly code (text)
2. VMP Transformer converts to protected assembly (text)
3. Protected assembly can be further processed or displayed

Example integration:

```python
# LLM generates assembly
llm_output = llm_model.generate_assembly(c_code)

# Apply VMP protection
protected = transform_assembly_to_vmp(llm_output)

# Use protected assembly
print(protected['vmp_assembly'])
```

## Limitations

- Currently supports a subset of x86/x64 assembly
- Text-based transformation (not binary)
- Basic protection level (can be extended)

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Acknowledgments

Based on the xVMPInterpreter C implementation.