import os
import json
import base64
from PIL import Image
import io
import folder_paths
import numpy as np
import torch

# 在文件开头添加try-except来处理导入错误
try:
    from groq import Groq
except ImportError:
    print("Error: groq package not found. Please install it using 'pip install groq'")
    Groq = None

class GroqPromptReverse:
    def __init__(self):
        # 确保目录存在
        self.base_path = os.path.join(folder_paths.get_input_directory(), "../custom_nodes/groq_prompt_reverse")
        os.makedirs(self.base_path, exist_ok=True)
        self.config_file = os.path.join(self.base_path, "config.json")
        self.api_key = self.load_api_key()
        self.type = "prompt_reverse"
        self.output_node = True
        self.description = "使用Groq API反推图片提示词"
    
    def load_api_key(self):
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    return config.get('api_key', '')
        except Exception:
            pass
        return os.getenv('GROQ_API_KEY', '')
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (["llama-3.2-90b-vision-preview"],),  # 更新为支持视觉的模型
                "api_key": ("STRING", {"default": "", "multiline": False}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "reverse_prompt"
    CATEGORY = "prompt"

    def save_api_key(self, api_key):
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump({'api_key': api_key}, f)
        except Exception as e:
            print(f"Error saving API key: {e}")

    def encode_image_to_base64(self, image):
        """将图片转换为base64格式，并进行压缩"""
        try:
            # 将PyTorch tensor转换为numpy数组
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            
            # 确保图片数据在0-255范围内
            image_array = (image[0] * 255).clip(0, 255).astype(np.uint8)
            pil_image = Image.fromarray(image_array)
            
            # 转换为RGB模式
            if len(image_array.shape) == 2:  # 果是灰度图
                pil_image = pil_image.convert('RGB')
            elif image_array.shape[2] == 4:  # 如果是RGBA
                pil_image = pil_image.convert('RGB')
            elif image_array.shape[2] == 1:  # 如果是单通道
                pil_image = pil_image.convert('RGB')
            
            # 设置最大尺寸为 1200 像素
            max_size = 1200
            ratio = min(max_size/float(pil_image.size[0]), max_size/float(pil_image.size[1]))
            if ratio < 1:
                new_size = tuple(int(dim * ratio) for dim in pil_image.size)
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # 保持高质量设置
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG", quality=95, optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return img_str
            
        except Exception as e:
            print(f"图片编码错误: {str(e)}")
            print(f"图片类型: {type(image)}")
            print(f"图片形状: {image.shape if hasattr(image, 'shape') else 'unknown'}")
            return None

    def reverse_prompt(self, image, model, api_key):
        if Groq is None:
            return ("Error: Groq package is not installed. Please install it using 'pip install groq'",)
            
        # 检查API密钥
        if api_key.strip():
            self.api_key = api_key
            self.save_api_key(api_key)
        
        if not self.api_key:
            return ("Error: Please provide a Groq API key in the node settings or environment variable GROQ_API_KEY",)

        try:
            # 初始化Groq客户端
            client = Groq(api_key=self.api_key)
            
            # 编码图片
            base64_image = self.encode_image_to_base64(image)
            if not base64_image:
                return ("Error: Failed to process image",)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": """Please describe this image in natural, flowing detail. Focus on:

                            1. Character/Subject Details:
                            - You need to recognize people's skin color and race
                            - You need to describe the light composition of the picture and the focal length of the camera
                            - You need to describe the camera's point of view
                            - Describe the main character/subject's complete appearance
                            - Note their style, expression, and pose
                            - Detail their clothing, including colors, patterns, and design
                            - Describe any accessories or distinctive features
                            - Include details about hair style, facial features, and overall look

                            2. Color and Style:
                            - Describe the color palette and color combinations
                            - Note the artistic style and technique
                            - Mention any unique artistic elements
                            - Describe textures and materials
                            - Detail any special effects or artistic treatments

                            3. Atmosphere and Setting:
                            - Describe the background and environment
                            - Note the overall mood and atmosphere
                            - Detail the lighting and its effects
                            - Describe any environmental elements
                            - Mention the composition and layout

                            Please provide a detailed, natural description that captures both the obvious and subtle details of the image. Focus on creating a flowing narrative that describes exactly what you see, similar to how you would explain the image to an artist.

                            """
                        }
                    ]
                }
            ]

            # 调用API
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.5,  # 稍微提高以获得更自然的描述
                max_tokens=900,
                top_p=0.95,
                stream=False,
                stop=None
            )
            
            response = completion.choices[0].message.content.strip()
            
            # 最基本的质量词
            essential_terms = [
                "masterpiece",
                "highly detailed",
                "best quality"
            ]
            
            # 只在开头添加基本质量词
            if not any(term.lower() in response.lower() for term in essential_terms):
                response = "masterpiece, highly detailed, best quality, " + response
            
            return (response,)

        except Exception as e:
            print(f"API调用错误: {str(e)}")
            return (f"Error: {str(e)}",)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "GroqPromptReverse": GroqPromptReverse
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GroqPromptReverse": "Groq Prompt Reverse"
} 