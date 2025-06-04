from openai import OpenAI, OpenAIError
import base64
from PIL import Image
import io

# action of the class
action_class = 'a person is falling down'

# Define the base strategy class
class VideoModelStrategy:
    def initialise_model(self, api_key=None):
        raise NotImplementedError("Must override initialise_model")

    def inference(self, clip, additional_info):
        raise NotImplementedError("Must override inference")

# ChatGPT model inheriting from the base strategy
class ChatGPT(VideoModelStrategy):
    def initialise_model(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def inference(self, clip, additional_info):
        base64_frames = []
        for frame in clip:
            pil_image = Image.fromarray(frame)
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_frames.append(img_str)
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Describe the action concisely in one sentence following this format: 'A man [verb]...'. If the action is unclear, default to {action_class}"
                            },
                            *[
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                } for base64_image in base64_frames
                            ]
                        ]
                    }
                ],
                temperature=1,
                max_tokens=250,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response.choices[0].message.content
        except OpenAIError as e:
            raise print(f"Error: {e}")
