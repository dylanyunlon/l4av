from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载合并后的模型
model = AutoModelForCausalLM.from_pretrained("llm4vmp.compiler", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("llm4vmp.compiler")

# 测试 C 代码到汇编的转换
test_code = "<s>system\nPlease compile this source code using clang-12 with optimization level O0 into assembly code. No strip the assembly code.</s>\nuser\nint ChewingLargeTable2::add_index_internal(int phrase_length,\n                                             const ChewingKey index[],\n                                             const ChewingKey keys[],\n                                             phrase_token_t token) {\n#define CASE(len) case len: { return add_index_internal<len>(index, keys, token); }\n    switch(phrase_length) {\n        CASE(1);\n        CASE(2);\n        CASE(3);\n        CASE(4);\n        CASE(5);\n        CASE(6);\n        CASE(7);\n        CASE(8);\n        CASE(9);\n        CASE(10);\n        CASE(11);\n        CASE(12);\n        CASE(13);\n        CASE(14);\n        CASE(15);\n        CASE(16);\n    default:\n        assert(false);\n    }\n#undef CASE\n    return ERROR_FILE_CORRUPTION;\n}</s>\nassistant\n"

# 使用训练时的格式
prompt = f"<s>[INST] Given the following C code, generate the corresponding assembly code:\n\n{test_code}\n\n[/INST]"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=8192, temperature=0.1)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)