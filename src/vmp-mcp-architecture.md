# VMP MCP 项目架构设计

## 项目目录结构

```
vmp-mcp-project/
├── src/
│   ├── vmp_interpreter/
│   │   ├── __init__.py
│   │   ├── interpreter.py      # 核心VMP解释器
│   │   ├── opcodes.py         # 操作码定义
│   │   ├── handlers.py        # 指令处理器
│   │   └── utils.py           # 工具函数
│   ├── mcp_server/
│   │   ├── __init__.py
│   │   ├── server.py          # MCP服务器实现
│   │   ├── tools.py           # MCP工具定义
│   │   └── resources.py       # MCP资源定义
│   └── asm_transformer/
│       ├── __init__.py
│       ├── parser.py          # 汇编代码解析器
│       └── transformer.py     # VMP转换器
├── tests/
│   ├── test_interpreter.py
│   ├── test_mcp_server.py
│   └── test_transformer.py
├── examples/
│   ├── sample_asm.txt         # 示例汇编代码
│   └── sample_vmp.txt         # VMP保护后的代码
├── requirements.txt
├── setup.py
└── README.md
```

## 核心组件设计

### 1. VMP解释器（Python版）

**主要类和功能：**
- `VMPInterpreter`: 核心解释器类
- `OpcodeManager`: 操作码管理和加密
- `MemoryManager`: 代码段和数据段管理
- `InstructionHandler`: 各种指令的处理器

### 2. MCP服务

**MCP工具定义：**
```python
tools = [
    {
        "name": "transform_asm_to_vmp",
        "description": "将普通汇编代码转换为VMP保护的汇编代码",
        "input_schema": {
            "type": "object",
            "properties": {
                "asm_code": {
                    "type": "string",
                    "description": "输入的汇编代码文本"
                },
                "protection_level": {
                    "type": "string",
                    "enum": ["basic", "advanced", "maximum"],
                    "description": "VMP保护级别"
                }
            },
            "required": ["asm_code"]
        }
    },
    {
        "name": "analyze_vmp_code",
        "description": "分析VMP保护后的代码结构",
        "input_schema": {
            "type": "object",
            "properties": {
                "vmp_code": {
                    "type": "string",
                    "description": "VMP保护的汇编代码"
                }
            },
            "required": ["vmp_code"]
        }
    }
]
```

### 3. 转换流程

```
普通汇编代码（文本）
    ↓
汇编解析器（提取指令和操作数）
    ↓
VMP编码器（转换为VMP字节码）
    ↓
加密处理（xorshift32）
    ↓
VMP汇编生成器
    ↓
VMP保护的汇编代码（文本）
```

## 开发步骤

### 第一阶段：Python VMP解释器实现
1. 实现基础数据结构和常量定义
2. 实现xorshift32加密算法
3. 实现内存管理（代码段、数据段）
4. 实现各操作码的处理函数
5. 实现主解释循环

### 第二阶段：汇编转换器实现
1. 实现简单的汇编解析器
2. 实现汇编指令到VMP操作码的映射
3. 实现VMP字节码生成器
4. 实现VMP汇编代码生成器

### 第三阶段：MCP服务实现
1. 设置MCP服务器框架
2. 实现工具接口
3. 实现资源管理
4. 添加错误处理和日志

### 第四阶段：测试和优化
1. 单元测试
2. 集成测试
3. 性能优化
4. 文档编写
