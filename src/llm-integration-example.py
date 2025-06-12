# llm_integration_example.py
"""
演示LLM如何通过MCP调用VMP转换服务
"""
import json
import subprocess
import asyncio
from typing import Dict, Any

class LLM4ASMWithVMP:
    """LLM4ASM与VMP集成的示例类"""
    
    def __init__(self):
        self.mcp_process = None
        
    async def start_mcp_server(self):
        """启动MCP服务器"""
        self.mcp_process = await asyncio.create_subprocess_exec(
            'python', '-m', 'mcp_server.main',
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # 初始化MCP
        init_request = {
            "jsonrpc": "2.0",
            "id": "init",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024.11.05",
                "capabilities": {}
            }
        }
        
        response = await self._send_request(init_request)
        print(f"MCP服务器初始化: {response}")
        
    async def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """发送请求到MCP服务器"""
        if not self.mcp_process:
            raise RuntimeError("MCP服务器未启动")
            
        # 发送请求
        request_json = json.dumps(request) + '\n'
        self.mcp_process.stdin.write(request_json.encode())
        await self.mcp_process.stdin.drain()
        
        # 读取响应
        response_line = await self.mcp_process.stdout.readline()
        response = json.loads(response_line.decode())
        
        return response
    
    async def llm_generate_asm(self, c_code: str) -> str:
        """模拟LLM生成汇编代码"""
        # 这里是模拟LLM生成的汇编代码
        # 实际应用中，这会调用您训练的编译器大模型
        
        print(f"输入的C代码:\n{c_code}\n")
        
        # 模拟生成的汇编代码
        asm_code = """
; Function: calculate
calculate:
    push %rbp
    mov %rsp, %rbp
    
    ; int a = 10
    mov $10, %eax
    mov %eax, -4(%rbp)
    
    ; int b = 20  
    mov $20, %eax
    mov %eax, -8(%rbp)
    
    ; int sum = a + b
    mov -4(%rbp), %eax
    add -8(%rbp), %eax
    mov %eax, -12(%rbp)
    
    ; return sum
    mov -12(%rbp), %eax
    
    pop %rbp
    ret
"""
        
        print(f"LLM生成的汇编代码:\n{asm_code}")
        return asm_code
    
    async def apply_vmp_protection(self, asm_code: str, protection_level: str = "advanced") -> str:
        """应用VMP保护"""
        # 调用MCP服务转换为VMP保护的代码
        request = {
            "jsonrpc": "2.0",
            "id": "vmp_transform",
            "method": "tools/call",
            "params": {
                "name": "transform_asm_to_vmp",
                "arguments": {
                    "asm_code": asm_code,
                    "protection_level": protection_level,
                    "encryption_seed": 0xC0FFEE
                }
            }
        }
        
        response = await self._send_request(request)
        
        if "error" in response:
            raise RuntimeError(f"VMP转换失败: {response['error']}")
            
        return response["result"]["vmp_code"]
    
    async def full_pipeline(self, c_code: str):
        """完整的处理流程：C代码 -> 汇编 -> VMP保护"""
        print("=== LLM4ASM + VMP 完整流程演示 ===\n")
        
        # 1. LLM生成汇编代码
        asm_code = await self.llm_generate_asm(c_code)
        
        # 2. 应用VMP保护
        print("\n正在应用VMP保护...")
        vmp_code = await self.apply_vmp_protection(asm_code, "advanced")
        
        print(f"\nVMP保护后的代码（前500字符）:\n{vmp_code[:500]}...")
        
        # 3. 分析VMP代码
        print("\n分析VMP代码结构...")
        analyze_request = {
            "jsonrpc": "2.0",
            "id": "analyze",
            "method": "tools/call",
            "params": {
                "name": "analyze_vmp_code",
                "arguments": {
                    "vmp_code": vmp_code,
                    "show_details": True
                }
            }
        }
        
        analyze_response = await self._send_request(analyze_request)
        print(f"分析结果: {json.dumps(analyze_response['result'], indent=2)}")
        
    async def cleanup(self):
        """清理资源"""
        if self.mcp_process:
            self.mcp_process.terminate()
            await self.mcp_process.wait()

# 使用示例
async def main():
    # C代码示例
    c_code = """
int calculate() {
    int a = 10;
    int b = 20;
    int sum = a + b;
    return sum;
}
"""
    
    # 创建集成实例
    llm_vmp = LLM4ASMWithVMP()
    
    try:
        # 启动MCP服务器
        await llm_vmp.start_mcp_server()
        
        # 执行完整流程
        await llm_vmp.full_pipeline(c_code)
        
    finally:
        # 清理
        await llm_vmp.cleanup()

# 文本层面的转换示例
def text_level_transformation_example():
    """
    展示纯文本层面的转换
    这是您实际使用场景的核心
    """
    print("\n=== 文本层面的VMP转换示例 ===\n")
    
    # 输入：LLM生成的汇编文本
    original_asm_text = """
    ; Simple function
    func:
        mov eax, 10
        add eax, 20
        ret
    """
    
    # 输出：VMP保护的汇编文本（示例）
    vmp_protected_text = """
    ; VMP Protected Assembly
    ; Protection Level: advanced
    ; Encryption Seed: 0xc0ffee
    
    section .vmp_data
        vmp_code_seg db 5000 dup(0)
        vmp_data_seg db 5000 dup(0)
    
    section .text
    global _vmp_start
    _vmp_start:
        ; Label: func at offset 0
        ; mov eax, 10
        ; VMP bytecode: 78563412214365870201080000000000000000000a000000
        ; add eax, 20  
        ; VMP bytecode: 0401080000000000000000001400000000000000
        ; ret
        ; VMP bytecode: 0a0000
        
        ; Initialize VMP interpreter
        mov rdi, vmp_code_seg
        mov rsi, vmp_data_seg
        mov rdx, 48
        call _vmp_interpreter_init
        
        ; Execute VMP code
        call _vmp_interpreter_execute
        
        ; Cleanup and return
        xor eax, eax
        ret
    
    ; VMP Interpreter Stub
    _vmp_interpreter_init:
        ; ... interpreter initialization code ...
        ret
    
    _vmp_interpreter_execute:
        ; ... interpreter execution loop ...
        ret
    
    ; Embedded VMP bytecode
    section .vmp_bytecode
        vmp_bytecode_data db 0x78, 0x56, 0x34, 0x12, 0x21, 0x43, 0x65, 0x87, \
                            0x02, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, \
                            0x00, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, \
                            0x04, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, \
                            0x00, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, \
                            0x00, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00
    """
    
    print("原始汇编文本:")
    print(original_asm_text)
    print("\nVMP保护后的文本:")
    print(vmp_protected_text)
    
    print("\n关键转换点:")
    print("1. 每条指令转换为VMP字节码")
    print("2. 添加VMP解释器框架")
    print("3. 嵌入加密的字节码数据")
    print("4. 保持文本可读性，同时实现保护")

if __name__ == "__main__":
    # 运行异步主函数
    # asyncio.run(main())
    
    # 展示文本层面的转换
    text_level_transformation_example()
