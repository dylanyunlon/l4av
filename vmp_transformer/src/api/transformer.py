# src/api/transformer.py

from ..assembler.converter import AssemblyToVMPConverter
from ..assembler.parser import AssemblyParser
from ..core.interpreter import VMPInterpreter

class VMPTransformer:
    """Main API for transforming assembly to VMP protected code"""
    
    def __init__(self, debug=False):
        self.converter = AssemblyToVMPConverter()
        self.parser = AssemblyParser()
        self.debug = debug
    
    def transform_assembly(self, assembly_code):
        """
        Transform normal assembly code to VMP protected assembly
        
        Args:
            assembly_code (str): Normal assembly code text
            
        Returns:
            dict: {
                'vmp_assembly': str,  # VMP protected assembly text
                'bytecode': bytes,    # Raw VMP bytecode
                'metadata': dict      # Additional metadata
            }
        """
        try:
            # Parse assembly code
            parsed = self.parser.parse(assembly_code)
            
            # Convert to VMP bytecode
            bytecode, variables = self.converter.convert_assembly(assembly_code)
            
            # Generate VMP protected assembly text
            vmp_text = self.converter.generate_vmp_text(assembly_code)
            
            # Metadata
            metadata = {
                'original_lines': len(assembly_code.strip().split('\n')),
                'bytecode_size': len(bytecode),
                'variables': variables,
                'protection_level': 'standard'
            }
            
            return {
                'vmp_assembly': vmp_text,
                'bytecode': bytecode,
                'metadata': metadata
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'vmp_assembly': None,
                'bytecode': None,
                'metadata': None
            }
    
    def analyze_vmp_code(self, vmp_bytecode, initial_data=None):
        """
        Analyze VMP bytecode execution (for debugging/verification)
        
        Args:
            vmp_bytecode (bytes): VMP bytecode to analyze
            initial_data (bytes): Initial data segment
            
        Returns:
            dict: Execution trace and analysis results
        """
        interpreter = VMPInterpreter(debug=self.debug)
        interpreter.load_program(vmp_bytecode, initial_data)
        
        try:
            trace = interpreter.execute()
            return {
                'success': True,
                'trace': trace,
                'final_data': interpreter.memory.data_seg[:100]  # First 100 bytes
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'trace': interpreter.execution_trace
            }
    
    def batch_transform(self, assembly_files):
        """
        Transform multiple assembly files
        
        Args:
            assembly_files (list): List of (filename, content) tuples
            
        Returns:
            dict: Results for each file
        """
        results = {}
        
        for filename, content in assembly_files:
            results[filename] = self.transform_assembly(content)
        
        return results


# Convenience functions
def transform_assembly_to_vmp(assembly_code, debug=False):
    """Quick function to transform assembly to VMP"""
    transformer = VMPTransformer(debug=debug)
    return transformer.transform_assembly(assembly_code)


def create_vmp_transformer(config=None):
    """Factory function to create configured transformer"""
    config = config or {}
    return VMPTransformer(
        debug=config.get('debug', False)
    )