# examples/demo.py

import sys
# sys.path.append('..')

from vmp_transformer.src.api.transformer import VMPTransformer, transform_assembly_to_vmp

def example_simple_arithmetic():
    """Example: Simple arithmetic operations"""
    print("=== Example 1: Simple Arithmetic ===")

    asm_code = {"tools":"[{\"name\":\"retrieve_data\",\"description\":\"从汇编源码中指定的区域内读取数据元素，假设适当的数据类型，并且原始地址已经被如D1、D2等标签替换。\",\"parameters\":{\"type\":\"object\",\"properties\":{\"label\":{\"type\":\"string\",\"description\":\"准备好的汇编语言代码中代表数据存放点的标识符（例如，D1, D2）。\"},\"type\":{\"type\":\"string\",\"description\":\"In terms of the labeled data type, one can anticipate it to be a primary type (such as i8) or an array type (like i8[8]). Primary types are limited to: i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, byte, word, dword, qword, and string.\"}},\"required\":[\"label\",\"type\"],\"additionalProperties\":false}}]","messages":[{"role":"user","content":"Demonstrate the C code that generates the assembly output below:\n\n<mv88fx_snd_readl>:\n  endbr64\n  sub    $0x8,%rsp\n  lea    -0xf000000(%rsi),%edi\n  call   <readl@plt>\n  add    $0x8,%rsp\n  ret"},{"role":"assistant","content":"unsigned int mv88fx_snd_readl(struct mv88fx_snd_chip *chip, unsigned int offs) {\n  unsigned int val = readl((0xf1000000 + offs));\n  return (val);\n}"}]}
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

def main():
    """Run all examples"""
    print("VMP Transformer Demo")
    print("===================\n")
    
    example_simple_arithmetic()
    # example_loop()
    # example_memory_operations()
    # example_batch_transform()
    # example_advanced_vmp()
    
    print("\n===================")
    print("Demo completed!")

if __name__ == "__main__":
    main()