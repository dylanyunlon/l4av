# requirements.txt
asyncio
dataclasses
typing
struct
logging

# setup.py
from setuptools import setup, find_packages

setup(
    name="vmp-mcp-server",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # 基础依赖
    ],
    entry_points={
        "console_scripts": [
            "vmp-mcp-server=mcp_server.main:main",
        ],
    },
    python_requires=">=3.8",
)

# example_usage.py
"""
示例：如何使用VMP MCP服务
"""
import json

def create_mcp_request(method, params=None):
    """创建MCP请求"""
    return {
        "jsonrpc": "2.0",
        "id": "1",
        "method": method,
        "params": params or {}
    }

# 示例1：转换简单的汇编代码
def example_transform_asm():
    # 原始汇编代码
    asm_code = """
main:
    push %rbp
    mov %rsp, %rbp
    
    ; 简单的加法运算
    mov $10, %rax
    mov $20, %rbx
    add %rbx, %rax
    
    ; 条件跳转
    cmp $30, %rax
    je success
    jmp fail
    
success:
    mov $0, %rax
    jmp end
    
fail:
    mov $1, %rax
    
end:
    pop %rbp
    ret
"""
    
    # 创建转换请求
    request = create_mcp_request("tools/call", {
        "name": "transform_asm_to_vmp",
        "arguments": {
            "asm_code": asm_code,
            "protection_level": "advanced",
            "encryption_seed": 0xDEADBEEF
        }
    })
    
    print("请求:")
    print(json.dumps(request, indent=2))
    
    # 模拟响应
    response = {
        "jsonrpc": "2.0",
        "id": "1",
        "result": {
            "vmp_code": "; VMP Protected Assembly\n; Protection Level: advanced\n...",
            "metadata": {
                "original_size": len(asm_code),
                "vmp_size": 2048,
                "protection_level": "advanced",
                "encryption_seed": "0xdeadbeef"
            }
        }
    }
    
    print("\n响应:")
    print(json.dumps(response, indent=2))

# 示例2：分析VMP代码
def example_analyze_vmp():
    vmp_code = """
section .vmp_data
    vmp_code_seg db 5000 dup(0)
    vmp_data_seg db 5000 dup(0)
    
section .text
global _vmp_start
_vmp_start:
    ; VMP protected code here
"""
    
    request = create_mcp_request("tools/call", {
        "name": "analyze_vmp_code",
        "arguments": {
            "vmp_code": vmp_code,
            "show_details": True
        }
    })
    
    print("分析VMP代码请求:")
    print(json.dumps(request, indent=2))

# 示例3：转换单条指令
def example_convert_instruction():
    request = create_mcp_request("tools/call", {
        "name": "convert_instruction",
        "arguments": {
            "instruction": "add",
            "operands": ["%rax", "%rbx"]
        }
    })
    
    print("转换单条指令请求:")
    print(json.dumps(request, indent=2))

if __name__ == "__main__":
    print("=== VMP MCP 使用示例 ===\n")
    
    print("1. 转换汇编代码为VMP保护代码:")
    example_transform_asm()
    
    print("\n" + "="*50 + "\n")
    
    print("2. 分析VMP代码:")
    example_analyze_vmp()
    
    print("\n" + "="*50 + "\n")
    
    print("3. 转换单条指令:")
    example_convert_instruction()

# test_vmp_interpreter.py
"""
测试VMP解释器
"""
from src.vmp_interpreter.interpreter import VMPInterpreter
from src.vmp_interpreter.opcodes import Opcode

def test_basic_operations():
    """测试基本操作"""
    # 创建解释器
    interpreter = VMPInterpreter()
    
    # 测试字节码：简单的加法
    # ALLOCA 8字节变量在偏移0
    # STORE 立即数10到变量
    # STORE 立即数20到变量  
    # BINARY_OP ADD
    # RET
    
    bytecode = bytearray()
    
    # 添加xorshift种子
    bytecode.extend([0x78, 0x56, 0x34, 0x12])  # opcode seed
    bytecode.extend([0x21, 0x43, 0x65, 0x87])  # code seed
    
    # ALLOCA指令
    bytecode.append(Opcode.ALLOCA_OP)
    bytecode.append(8)  # size
    bytecode.append(0)  # type
    bytecode.extend([0, 0, 0, 0, 0, 0, 0, 0])  # offset
    bytecode.extend([0, 0, 0, 0, 0, 0, 0, 0])  # area offset
    
    # 更多指令...
    
    # 加载字节码
    interpreter.load_code(bytecode)
    
    # 执行
    result = interpreter.execute()
    
    print(f"执行结果: {result[:8]}")

def test_asm_transform():
    """测试汇编转换"""
    from src.asm_transformer.transformer import ASMToVMPTransformer
    
    asm_code = """
    mov $10, %rax
    add $5, %rax
    ret
    """
    
    transformer = ASMToVMPTransformer(protection_level="basic")
    vmp_code = transformer.transform(asm_code)
    
    print("原始汇编:")
    print(asm_code)
    print("\nVMP保护后:")
    print(vmp_code[:500] + "...")  # 只显示前500字符

if __name__ == "__main__":
    print("测试VMP解释器基本操作:")
    test_basic_operations()
    
    print("\n" + "="*50 + "\n")
    
    print("测试汇编转换:")
    test_asm_transform()

# README.md
# VMP MCP Server

将汇编代码转换为VMP（虚拟机保护）代码的MCP服务器。

## 功能特性

- **汇编到VMP转换**: 将普通x86/x64汇编代码转换为VMP保护的代码
- **多级保护**: 支持basic、advanced、maximum三种保护级别
- **MCP接口**: 通过标准MCP协议供大语言模型调用
- **代码分析**: 分析VMP保护后的代码结构
- **单指令转换**: 支持单条指令的转换测试

## 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/vmp-mcp-project.git
cd vmp-mcp-project

# 安装依赖
pip install -r requirements.txt

# 安装包
pip install -e .
```

## 使用方法

### 1. 启动MCP服务器

```bash
python -m mcp_server.main
```

### 2. 发送MCP请求

服务器通过标准输入接收JSON-RPC请求，通过标准输出返回响应。

#### 转换汇编代码

```json
{
  "jsonrpc": "2.0",
  "id": "1",
  "method": "tools/call",
  "params": {
    "name": "transform_asm_to_vmp",
    "arguments": {
      "asm_code": "mov $10, %rax\nadd $20, %rax\nret",
      "protection_level": "advanced"
    }
  }
}
```

### 3. Python API使用

```python
from src.asm_transformer.transformer import ASMToVMPTransformer

# 创建转换器
transformer = ASMToVMPTransformer(protection_level="advanced")

# 转换汇编代码
asm_code = """
main:
    mov $10, %rax
    add $20, %rax
    ret
"""

vmp_code = transformer.transform(asm_code)
print(vmp_code)
```

## 架构说明

- **VMP解释器**: Python实现的VMP虚拟机，支持所有核心指令
- **汇编解析器**: 解析x86/x64汇编指令
- **MCP服务器**: 实现MCP协议，提供工具接口
- **转换器**: 将汇编指令转换为VMP字节码

## 支持的指令

- 数据移动: mov, lea, push, pop
- 算术运算: add, sub, mul, div
- 逻辑运算: and, or, xor, shl, shr
- 控制流: jmp, je, jne, call, ret
- 比较: cmp, test

## 许可证

MIT License
