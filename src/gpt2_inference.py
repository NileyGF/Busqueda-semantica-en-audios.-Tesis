"""
https://huggingface.co/docs/api-inference/quicktour
https://huggingface.co/gpt2
https://huggingface.co/models
https://huggingface.co/docs/api-inference/detailed_parameters#detailed-parameters
"""
import requests

API_TOKEN = "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"
API_URL = "https://api-inference.huggingface.co/models/gpt2"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

data = query("Can you please let us know more details about your ")

data = query({"inputs": "The answer to the universe is"})
""" When sending your request, you should send a JSON encoded payload. Here are all the options """
# All parameters	
# inputs (required):	a string to be generated from
###      parameters	  dict containing the following keys:
# top_k	                (Default: None). Integer to define the top tokens considered within the sample operation to create new text.
# top_p	                (Default: None). Float to define the tokens that are within the sample operation of text generation. Add tokens in the sample for more probable to least probable until the sum of the probabilities is greater than top_p.
# temperature	        (Default: 1.0). Float (0.0-100.0). The temperature of the sampling operation. 1 means regular sampling, 0 means always take the highest score, 100.0 is getting closer to uniform probability.
# repetition_penalty	(Default: None). Float (0.0-100.0). The more a token is used within generation the more it is penalized to not be picked in successive generation passes.
# max_new_tokens	    (Default: None). Int (0-250). The amount of new tokens to be generated, this does not include the input length it is a estimate of the size of generated text you want. Each new tokens slows down the request, so look for balance between response times and length of text generated.
# max_time	            (Default: None). Float (0-120.0). The amount of time in seconds that the query should take maximum. Network can cause some overhead so it will be a soft limit. Use that in combination with max_new_tokens for best results.
# return_full_text	    (Default: True). Bool. If set to False, the return results will not contain the original query making it easier for prompting.
# num_return_sequences	(Default: 1). Integer. The number of proposition you want to be returned.
# do_sample	            (Optional: True). Bool. Whether or not to use sampling, use greedy decoding otherwise.
###     options	    a dict containing the following keys:
# use_cache	            (Default: true). Boolean. There is a cache layer on the inference API to speedup requests we have already seen. Most models can use those results as is as models are deterministic (meaning the results will be the same anyway). However if you use a non deterministic model, you can set this parameter to prevent the caching mechanism from being used resulting in a real new query.
# wait_for_model	    (Default: false) Boolean. If the model is not ready, wait for it instead of receiving 503. It limits the number of requests required to get your inference done. It is advised to only set this flag to true after receiving a 503 error as it will limit hanging in your application to known places.


# Return value is either a dict or a list of dicts if you sent a list of inputs
# Returned values	
# generated_text	The continuated string
data == [
    {
        "generated_text": 'The answer to the universe is that we are the creation of the entire universe," says Fitch.\n\nAs of the 1960s, six times as many Americans still make fewer than six bucks ($17) per year on their way to retirement.'
    }
]






""" Summarization task """
# This task is well known to summarize longer text into shorter 
# text. Be careful, some models have a maximum length of input.

# Recommended model: facebook/bart-large-cnn.
import requests
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
data = query(
    {
        "inputs": "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.",
        "parameters": {"do_sample": False},
    }
)
# Response
self.assertEqual(
    data,
    [
        {
            "summary_text": "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world.",
        },
    ],
)
