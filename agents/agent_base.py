import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

class BaseAgent:
    def __init__(self, name, role, system_prompt, tools_declarations=None):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        
        api_key = os.getenv("LLM_HW_API_KEY")
        if not api_key:
            raise ValueError(f"API Key not set for agent {name}")
        
        self.client = genai.Client(api_key=api_key)
        
        config_kwargs = {"system_instruction": system_prompt}
        if tools_declarations:
            # Wrappin declarations in a Tool object
            tools = types.Tool(function_declarations=tools_declarations)
            config_kwargs["tools"] = [tools]
            
        self.config = types.GenerateContentConfig(**config_kwargs)
        self.chat = self.client.chats.create(model="gemini-2.5-pro", config=self.config)

    def run(self, user_input):
        print(f"\n{'='*80}")
        print(f"🤖 {self.name} is analyzing the task...")
        print(f"{'='*80}")
        response = self.chat.send_message(user_input)
        
        while True:
            if not response.candidates or not response.candidates[0].content:
                msg = f"[{self.name}] Error: No content in response. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'None'}"
                print(msg)
                return msg
                
            parts = response.candidates[0].content.parts
            if parts is None or len(parts) == 0:
                print(f"[{self.name}] Info: Empty parts list. Finish reason: {response.candidates[0].finish_reason}")
                return ""

            has_func_call = False
            tool_results = []
            
            for part in parts:
                if part.function_call:
                    has_func_call = True
                    func_name = part.function_call.name
                    args = part.function_call.args
                    
                    if func_name == "execute_python_code" and "code_string" in args:
                        code = args["code_string"]
                        import re
                        
                        params = {}
                        for pattern in [
                            r'max_depth\s*=\s*(\d+)',
                            r'learning_rate\s*=\s*([\d.]+)',
                            r'n_estimators\s*=\s*(\d+)',
                            r'min_samples_leaf\s*=\s*(\d+)',
                            r'min_samples_split\s*=\s*(\d+)'
                        ]:
                            match = re.search(pattern, code)
                            if match:
                                param_name = pattern.split('\\')[0]
                                params[param_name] = match.group(1)
                        
                        print(f"\n💡 {self.name}: Training model with hyperparameters")
                        if params:
                            for k, v in params.items():
                                print(f"   {k} = {v}")
                        else:
                            print(f"   Using default parameters")
                    else:
                        args_str = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in args.items()])
                        print(f"\n💡 {self.name}: I will use '{func_name}' tool")
                        if args:
                            print(f"   Parameters: {args_str}")
                    
                    try:
                        result = self.execute_tool(func_name, args)
                        
                        if func_name == "execute_python_code":
                            import re
                            acc_match = re.search(r'Accuracy:\s*([\d.]+)', str(result))
                            f1_match = re.search(r'F1 Score:\s*([\d.]+)', str(result))
                            
                            if acc_match and f1_match:
                                print(f"   ✓ Results: Accuracy={acc_match.group(1)}, F1={f1_match.group(1)}")
                            elif "Error" in str(result) or "Traceback" in str(result):
                                error_preview = str(result).split('\n')[-3] if '\n' in str(result) else str(result)[:100]
                                print(f"   ✗ Error: {error_preview}")
                        
                        tool_results.append(types.Part.from_function_response(
                            name=func_name,
                            response={"result": result}
                        ))
                    except Exception as e:
                        print(f"   ✗ Error: {str(e)}")
                        tool_results.append(types.Part.from_function_response(
                            name=func_name,
                            response={"result": f"Error: {str(e)}"}
                        ))
            
            if not has_func_call:
                final_text = response.text
                if not final_text:
                    final_text = "".join([p.text for p in response.candidates[0].content.parts if p.text])
                
                print(f"\n📋 {self.name}'s Summary:")
                print(f"{'─'*80}")
                print(final_text)
                print(f"{'─'*80}")
                return final_text
                
            response = self.chat.send_message(tool_results)
