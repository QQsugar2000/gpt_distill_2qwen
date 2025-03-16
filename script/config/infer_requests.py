from swift.llm import InferRequest
from script.config.prompt import prompt
infer_requests = [
    InferRequest(messages=[{'role': 'user', 'content': 'who are you?'}]),
    InferRequest(messages=[{'role': 'user', 'content': '<image>'+prompt}],
                 images=['data/image/3.png'] ),
    InferRequest(messages=[{'role': 'user', 'content': '<image>'+prompt}],
                 images=['data/image/4.png'] ),
]