# mcp_server/server.py
import json
import asyncio
from typing import Dict, Any, List
from dataclasses import dataclass
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MCPRequest:
    id: str
    method: str
    params: Dict[str, Any]

@dataclass
class MCPResponse:
    id: str
    result: Any = None
    error: Dict[str, Any] = None

class VMPMCPServer:
    def __init__(self):
        self.tools = self._define_tools()
        self.handlers = {
            "initialize": self.handle_initialize,
            "tools/list": self.handle_tools_list,
            "tools/call": self.handle_tools_call,
        }
        
    def _define_tools(self) -> List[Dict[str, Any]]:
        """定义MCP工具"""
        return [
            {
                "name": "transform_asm_to_vmp",
                "description": "将普通汇编代码转换为VMP保护的汇编代码",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "asm_code": {
                            "type": "string",
                            "description": "输入的汇编代码文本"
                        },
                        "protection_level": {
                            "type": "string",
                            "enum": ["basic", "advanced", "maximum"],
                            "default": "basic",
                            "description": "VMP保护级别"
                        },
                        "encryption_seed": {
                            "type": "integer",
                            "default": 0x12345678,
                            "description": "加密种子（可选）"
                        }
                    },
                    "required": ["asm_code"]
                }
            },
            {
                "name": "analyze_vmp_code",
                "description": "分析VMP保护后的代码结构",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "vmp_code": {
                            "type": "string",
                            "description": "VMP保护的汇编代码"
                        },
                        "show_details": {
                            "type": "boolean",
                            "default": False,
                            "description": "是否显示详细分析"
                        }
                    },
                    "required": ["vmp_code"]
                }
            },
            {
                "name": "convert_instruction",
                "description": "将单条汇编指令转换为VMP字节码",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "description": "汇编指令"
                        },
                        "operands": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "操作数列表"
                        }
                    },
                    "required": ["instruction"]
                }
            }
        ]
    
    async def handle_initialize(self, request: MCPRequest) -> MCPResponse:
        """处理初始化请求"""
        return MCPResponse(
            id=request.id,
            result={
                "protocolVersion": "2024.11.05",
                "capabilities": {
                    "tools": {},
                    "resources": {}
                },
                "serverInfo": {
                    "name": "vmp-mcp-server",
                    "version": "1.0.0"
                }
            }
        )
    
    async def handle_tools_list(self, request: MCPRequest) -> MCPResponse:
        """列出可用工具"""
        return MCPResponse(
            id=request.id,
            result={"tools": self.tools}
        )
    
    async def handle_tools_call(self, request: MCPRequest) -> MCPResponse:
        """调用工具"""
        tool_name = request.params.get("name")
        arguments = request.params.get("arguments", {})
        
        try:
            if tool_name == "transform_asm_to_vmp":
                result = await self.transform_asm_to_vmp(arguments)
            elif tool_name == "analyze_vmp_code":
                result = await self.analyze_vmp_code(arguments)
            elif tool_name == "convert_instruction":
                result = await self.convert_instruction(arguments)
            else:
                return MCPResponse(
                    id=request.id,
                    error={"code": -32601, "message": f"Unknown tool: {tool_name}"}
                )
            
            return MCPResponse(id=request.id, result=result)
            
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {str(e)}")
            return MCPResponse(
                id=request.id,
                error={"code": -32603, "message": str(e)}
            )
    
    async def transform_asm_to_vmp(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """转换汇编代码为VMP保护代码"""
        from ..asm_transformer.transformer import ASMToVMPTransformer
        
        asm_code = args["asm_code"]
        protection_level = args.get("protection_level", "basic")
        encryption_seed = args.get("encryption_seed", 0x12345678)
        
        transformer = ASMToVMPTransformer(
            protection_level=protection_level,
            encryption_seed=encryption_seed
        )
        
        vmp_code = transformer.transform(asm_code)
        
        return {
            "vmp_code": vmp_code,
            "metadata": {
                "original_size": len(asm_code),
                "vmp_size": len(vmp_code),
                "protection_level": protection_level,
                "encryption_seed": hex(encryption_seed)
            }
        }
    
    async def analyze_vmp_code(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """分析VMP代码"""
        vmp_code = args["vmp_code"]
        show_details = args.get("show_details", False)
        
        # 这里实现VMP代码分析逻辑
        analysis = {
            "code_segments": self._analyze_segments(vmp_code),
            "encryption_info": self._analyze_encryption(vmp_code),
            "instruction_count": self._count_instructions(vmp_code)
        }
        
        if show_details:
            analysis["detailed_bytecode"] = self._get_detailed_bytecode(vmp_code)
        
        return analysis
    
    async def convert_instruction(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """转换单条指令"""
        from ..asm_transformer.parser import InstructionParser
        
        instruction = args["instruction"]
        operands = args.get("operands", [])
        
        parser = InstructionParser()
        vmp_bytecode = parser.convert_to_vmp(instruction, operands)
        
        return {
            "instruction": instruction,
            "operands": operands,
            "vmp_bytecode": vmp_bytecode.hex(),
            "bytecode_size": len(vmp_bytecode)
        }
    
    def _analyze_segments(self, vmp_code: str) -> Dict[str, Any]:
        """分析代码段"""
        # 实现代码段分析
        return {
            "total_segments": 1,
            "segment_sizes": [len(vmp_code)]
        }
    
    def _analyze_encryption(self, vmp_code: str) -> Dict[str, Any]:
        """分析加密信息"""
        # 实现加密分析
        return {
            "encryption_type": "xorshift32",
            "encrypted": True
        }
    
    def _count_instructions(self, vmp_code: str) -> int:
        """统计指令数量"""
        # 简单统计
        return vmp_code.count("VMP_")
    
    def _get_detailed_bytecode(self, vmp_code: str) -> List[Dict[str, Any]]:
        """获取详细字节码"""
        # 实现详细分析
        return []
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理MCP请求"""
        request = MCPRequest(
            id=request_data.get("id", ""),
            method=request_data.get("method", ""),
            params=request_data.get("params", {})
        )
        
        handler = self.handlers.get(request.method)
        if not handler:
            response = MCPResponse(
                id=request.id,
                error={"code": -32601, "message": f"Method not found: {request.method}"}
            )
        else:
            response = await handler(request)
        
        # 转换响应为字典
        result = {"jsonrpc": "2.0", "id": response.id}
        if response.error:
            result["error"] = response.error
        else:
            result["result"] = response.result
        
        return result

# mcp_server/main.py
import asyncio
import sys
import json

async def main():
    """MCP服务器主入口"""
    server = VMPMCPServer()
    
    # 从标准输入读取请求
    async def read_input():
        loop = asyncio.get_event_loop()
        while True:
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                    
                # 解析JSON-RPC请求
                request_data = json.loads(line.strip())
                
                # 处理请求
                response = await server.process_request(request_data)
                
                # 输出响应
                print(json.dumps(response), flush=True)
                
            except json.JSONDecodeError as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"}
                }
                print(json.dumps(error_response), flush=True)
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
    
    await read_input()

if __name__ == "__main__":
    asyncio.run(main())
