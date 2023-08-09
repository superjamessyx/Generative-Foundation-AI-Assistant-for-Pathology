#                                     PathAsst Caption Tool

The PathAsst caption model is based on the llama2-7b architecture and has been fine-tuned using the GPT-4's generated caption split & refine instruction-tuning samples. You can use the tool to automatically split and refine your own or our provided data.

## Usage

> #### Step1: Download model checkpoint in [pathasst_caption_tool](https://huggingface.co/jamessyx/pathasst_caption_tool)



> #### Step2: Load the model

You can refer to https://github.com/lm-sys/FastChat to load our fine-tuned model.

We suggest using the model via the OpenAI API format to load and use the model.  Specifically, download [FastChat](https://github.com/lm-sys/FastChat)and then use the provided command to load the model.

#### RESTful API Server

First, launch the controller

```
python3 -m fastchat.serve.controller
```



Then, launch the model worker(s)

```
python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.3
```



Finally, launch the RESTful API server

```
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```



Now, let us test the API server.

#### OpenAI Official SDK

The goal of `openai_api_server.py` is to implement a fully OpenAI-compatible API server, so the models can be used directly with [openai-python](https://github.com/openai/openai-python) library.

First, install openai-python:

```
pip install --upgrade openai
```



Then, interact with model vicuna:

```
import openai
# to get proper authentication, make sure to use a valid key that's listed in
# the --api-keys flag. if no flag value is provided, the `api_key` will be ignored.
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"

model = "vicuna-7b-v1.3"

input_prompt = ''

# create a chat completion
completion = openai.ChatCompletion.create(
  model=model,
  messages=[{"role": "user", "content": input_prompt}]
)
# print the completion
print(completion.choices[0].message.content)
```





> #### Step3: Use the prompt to implement caption split & refine

Note that you can also adjust the prompt for better performance.

- For caption refine, please use the following prompt:

```python
input_prompt = f'''Given a caption of an image containing sub-images, please decompose the caption in accordance with each sub-image. Be sure to adhere to the following guidelines:
    
1. Preserve the original wording of the caption. Refrain from adding new information, summaries, or introductions.
2. Omit references to the index or number of the sub-images, such as xx), (xx), left, right, etc.
3. There might a common caption is shared among all sub-images, please incorporate it into each sub-image's caption.
The final output should be in JSON format.

Input: {input_prompt}
Output: '''
```



- For caption split, please use the following prompt:

```python
input_prompt = f'''Reconstruct the following caption of a pathology image by following these instructions:

1. Remove any information not directly observable in the image, such as magnification levels (e.g., 40x), metric (e.g., 5 μM, 20 μM), patient age (e.g., 40-year-old man, 25-year-old woman), and anything might irrelevant to the image.
2. Rewrite the caption to clearly and solely describe the visible content within the image.
3. Do not add any extra information, use the original text as much as possible.
4. Do not mention that the information source is the caption. Always answer as if you are directly looking at the image.

Input:  {input_prompt}
Output: '''
```

