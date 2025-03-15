from config.prompt import prompt
from config.api_config import base_url, api_key
from openai import OpenAI
import base64
import os

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

# Function to encode the image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_to_response(image_path,model = 'gpt-4o'):
    print(f'processing {image_path}')
    encoded_image = encode_image(image_path)
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ]
            }
        ],
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    print(image_to_response(r'data/image/1.png'))
    