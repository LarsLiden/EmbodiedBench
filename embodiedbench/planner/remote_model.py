import json
import sys
import os
import base64
import anthropic
import google.generativeai as genai
from openai import OpenAI, AzureOpenAI
from azure.identity import (
    DefaultAzureCredential,
    ChainedTokenCredential,
    AzureCliCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
import typing_extensions as typing
import lmdeploy
from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig
from embodiedbench.planner.planner_config.generation_guide import llm_generation_guide, vlm_generation_guide
from embodiedbench.planner.planner_config.generation_guide_manip import llm_generation_guide_manip, vlm_generation_guide_manip
from embodiedbench.planner.planner_utils import convert_format_2claude, convert_format_2gemini, ActionPlan_1, ActionPlan, ActionPlan_lang, \
                                             ActionPlan_1_manip, ActionPlan_manip, ActionPlan_lang_manip, fix_json

max_completion_tokens = 2048
remote_url = os.environ.get('remote_url')

class RemoteModel:
    def __init__(
        self,
        model_name,
        model_type='remote',
        language_only=False,
        tp=1,
        task_type=None, # used to distinguish between manipulation and other environments
        temperature=0.0
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.language_only = language_only
        self.task_type = task_type
        self.temperature = temperature

        if self.model_type == 'local':
            backend_config = PytorchEngineConfig(session_len=12000, dtype='float16', tp=tp)
            self.model = pipeline(self.model_name, backend_config=backend_config)
        elif self.model_type == 'azure_openai':
            # Azure OpenAI setup with proper authentication
            scope = "api://trapi/.default"
            
            azure_credential = ChainedTokenCredential(
                AzureCliCredential(),
                DefaultAzureCredential(
                    exclude_cli_credential=True,
                    exclude_environment_credential=True,
                    exclude_shared_token_cache_credential=True,
                    exclude_developer_cli_credential=True,
                    exclude_powershell_credential=True,
                    exclude_interactive_browser_credential=True,
                    exclude_visual_studio_code_credentials=True,
                    managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
                ),
            )

            token_provider = get_bearer_token_provider(
                ChainedTokenCredential(
                    AzureCliCredential(),
                    ManagedIdentityCredential(),
                ),
                scope,
            )
            
            api_version = "2024-12-01-preview"
            instance = "gcr/shared"
            endpoint = f"https://trapi.research.microsoft.com/{instance}"
            
            self.model = AzureOpenAI(
                azure_endpoint=endpoint,
                azure_ad_token_provider=token_provider,
                api_version=api_version,
            )
        else:
            if "claude" in self.model_name:
                self.model = anthropic.Anthropic(
                    api_key=os.environ.get("ANTHROPIC_API_KEY"),
                )
            elif "gemini" in self.model_name:
                self.model = OpenAI(
                    api_key=os.environ.get("GEMINI_API_KEY"),
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )
            elif "gpt" in self.model_name:
                self.model = OpenAI()
            elif 'qwen' in self.model_name:
                self.model = OpenAI(
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
            elif "Qwen2-VL" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "Qwen2.5-VL" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "Llama-3.2-11B-Vision-Instruct" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "OpenGVLab/InternVL" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "meta-llama/Llama-3.2-90B-Vision-Instruct" in self.model_name:
                self.model = OpenAI(base_url = remote_url)
            elif "90b-vision-instruct" in self.model_name: # you can use fireworks to inference
                self.model = OpenAI(base_url='https://api.fireworks.ai/inference/v1',
                                    api_key=os.environ.get("firework_API_KEY"))
            else:
                try:
                    self.model = OpenAI(base_url = remote_url)
                except:
                    raise ValueError(f"Unsupported model name: {model_name}")


    def respond(self, message_history: list):
        if self.model_type == 'local':
            return self._call_local(message_history)
        elif self.model_type == 'azure_openai':
            return self._call_azure_openai(message_history)
        else:
            if "claude" in self.model_name:
                return self._call_claude(message_history)
            elif "gemini" in self.model_name:
                return self._call_gemini(message_history)
            elif "gpt" in self.model_name:
                return self._call_gpt(message_history)
            elif 'qwen' in self.model_name:
                return self._call_gpt(message_history)
            elif "Qwen2-VL-7B-Instruct" in self.model_name:
                return self._call_qwen7b(message_history)
            elif "Qwen2.5-VL-7B-Instruct" in self.model_name:
                return self._call_qwen7b(message_history)
            elif "Qwen2-VL-72B-Instruct" in self.model_name:
                return self._call_qwen72b(message_history)
            elif "Qwen2.5-VL-72B-Instruct" in self.model_name:
                return self._call_qwen72b(message_history)
            elif "Llama-3.2-11B-Vision-Instruct" in self.model_name:
                return self._call_llama11b(message_history)
            elif "meta-llama/Llama-3.2-90B-Vision-Instruct" in self.model_name:
                return self._call_qwen72b(message_history)
            elif "90b-vision-instruct" in self.model_name:
                return self._call_llama90(message_history)
            elif "OpenGVLab/InternVL" in self.model_name:
                return self._call_intern38b(message_history)
            # elif "OpenGVLab/InternVL2_5-38B" in self.model_name:
            #     return self._call_intern38b(message_history)
            # elif "OpenGVLab/InternVL2_5-78B" in self.model_name:
            #     return self._call_intern38b(message_history)
            else:
                raise ValueError(f"Unsupported model name: {self.model_name}")

    def _call_local(self, message_history: list):
        if self.task_type == 'manip':
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "embodied_planning",
                    "schema": llm_generation_guide_manip if self.language_only else vlm_generation_guide_manip
                }
            }
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "embodied_planning",
                    "schema": llm_generation_guide if self.language_only else vlm_generation_guide
                }
            }
        response = self.model(
            message_history,
            gen_config=GenerationConfig(
                temperature=self.temperature,
                response_format=response_format,
                max_new_tokens=max_completion_tokens,
            )
        )
        out = response.text
        out = fix_json(out)
        return out

    def _call_claude(self, message_history: list):

        if not self.language_only:
            message_history = convert_format_2claude(message_history)

        response = self.model.messages.create(
            model=self.model_name,
            max_tokens=max_completion_tokens,
            temperature=self.temperature,
            messages=message_history
        )

        return response.content[0].text 

    def _call_gemini(self, message_history: list):

        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        if self.task_type == 'manip':
            response = self.model.beta.chat.completions.parse(
                model=self.model_name, 
                messages=message_history,
                response_format= ActionPlan_lang_manip if self.language_only else ActionPlan_manip,
                temperature=self.temperature,
                max_tokens=max_completion_tokens
            )
        else:
            response = self.model.beta.chat.completions.parse(
                model=self.model_name, 
                messages=message_history,
                response_format= ActionPlan_lang if self.language_only else ActionPlan,
                temperature=self.temperature,
                max_tokens=max_completion_tokens
            )
        tokens = response.usage.prompt_tokens

        return str(response.choices[0].message.parsed.model_dump_json())

    def _call_gpt(self, message_history: list):

        if not self.language_only:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            response_format=response_format,
            temperature=self.temperature,
            max_tokens=max_completion_tokens
        )
        out = response.choices[0].message.content

        return out
    
    def _call_azure_openai(self, message_history: list):
        """
        Handle Azure OpenAI model calls with proper deployment name handling
        and temperature adjustment for o3 models
        """
        deployment_name = self.model_name
        
        # Handle special case for o3 models which only support temperature 1.0
        if "o3" in deployment_name:
            temperature = 1.0
        else:
            temperature = self.temperature

        # Set up response format based on task type and language mode
        if not self.language_only:
            if self.task_type == 'manip':
                response_format = dict(type='json_schema', json_schema=dict(name='embodied_planning', schema=vlm_generation_guide_manip))
            else:
                response_format = dict(type='json_schema', json_schema=dict(name='embodied_planning', schema=vlm_generation_guide))
        else:
            if self.task_type == 'manip':
                response_format = dict(type='json_schema', json_schema=dict(name='embodied_planning', schema=llm_generation_guide_manip))
            else:
                response_format = dict(type='json_schema', json_schema=dict(name='embodied_planning', schema=llm_generation_guide))

        try:
            response = self.model.chat.completions.create(
                model=deployment_name,
                messages=message_history,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_completion_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"--= !!! Invalid Azure OpenAI Request: {e} !!! =--", file=sys.stderr)
            raise
    
    def _call_qwen7b(self, message_history: list):

        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        if not self.language_only:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            response_format=response_format,
            temperature=self.temperature,
            max_tokens=max_completion_tokens
        )

        out = response.choices[0].message.content
        return out
    
    def _call_llama90(self, message_history: list):
        if self.task_type == "manip":
            response = self.model.chat.completions.create(
                model="accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
                messages=message_history,
                response_format={"type": "json_object", "schema": ActionPlan_1_manip.model_json_schema()},
                temperature = self.temperature
            )
            out = response.choices[0].message.content
            
        else:
            response = self.model.chat.completions.create(
                model="accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
                messages=message_history,
                response_format={"type": "json_object", "schema": ActionPlan_1.model_json_schema()},
                temperature = self.temperature
            )
            out = response.choices[0].message.content
        return out
    
    def _call_llama11b(self, message_history):

        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        if not self.language_only:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            response_format=response_format,
            temperature=self.temperature,
            max_tokens=max_completion_tokens
        )
        out = response.choices[0].message.content
        return out
    

    def _call_qwen72b(self, message_history):
        if not self.language_only:
            message_history = convert_format_2gemini(message_history)

        if not self.language_only:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))
        
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            response_format=response_format,
            temperature=self.temperature,
            max_tokens=max_completion_tokens
        )

        # easy to meet json errors
        out = response.choices[0].message.content
        out = fix_json(out)
        return out
    
    def _call_intern38b(self, message_history):

        # if not self.language_only:
        #     message_history = convert_format_2gemini(message_history)

        # no use, lmdeploy use support json schema only if it is pytorch-backended
        if not self.language_only:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=vlm_generation_guide))
        else:
            if self.task_type == 'manip':
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide_manip))
            else:
                response_format=dict(type='json_schema',  json_schema=dict(name='embodied_planning',schema=llm_generation_guide))

        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=message_history,
            # response_format=response_format,
            temperature=self.temperature,
            max_tokens=max_completion_tokens,
        )

        # easy to meet json errors
        out = response.choices[0].message.content
        out = fix_json(out)
        return out



if __name__ == "__main__":

    model = RemoteModel(
        'Qwen/Qwen2-VL-72B-Instruct', #'meta-llama/Llama-3.2-11B-Vision-Instruct',
        True #False
    )#'claude-3-5-sonnet-20241022, Qwen/Qwen2-VL-72B-Instruct, meta-llama/Llama-3.2-11B-Vision-Instruct


    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        

    base64_image = encode_image("../../evaluator/midlevel/output.png")
        
    messages=[
        {
            "role": "user",
            "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                }
            },
            {
                "type": "text",
                "text":f"What do you think for this picture?? {template}?"
            },
            ],
        }
    ]

    response = model.respond(messages)
    print(response)

