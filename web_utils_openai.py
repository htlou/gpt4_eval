
import argparse
import hashlib
import itertools
import json
import logging

import os
import random
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Union

import openai
import ray
import tqdm
# from tqdm import tqdm

import urllib3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def request_openai_noexcept(
    messages: list[dict[str, str]],
    openai_api_keys: str,
    openai_model: str,
    base_url: str | None = None,
) -> list[dict[str, object]]:
    output = None
    hit_rate_limit = 0
    while True:
        client = openai.OpenAI(api_key=openai_api_keys, base_url=base_url)
        try:
            output = client.chat.completions.create(
                messages=messages,
                model=openai_model,
                max_tokens=8192,
                temperature=0.05,
            )
            break
        except openai.OpenAIError as e:
            logging.error(e)
            if 'maximum context length' in str(e).lower():
                return {
                    'messages': messages,
                    'output': 'ERROR: reached maximum context length',
                    'model': openai_model,
                }
            if 'repetitive patterns' in str(e).lower():
                return {
                    'messages': messages,
                    'output': 'ERROR: Sorry! We have encountered an issue with repetitive patterns in your prompt. Please try again with a different prompt.',
                    'model': openai_model,
                }
            time.sleep(random.randint(5, 30) * 0.1)  # noqa: S311
    return {
        'messages': messages,
        'output': output.choices[0].message.content,
        'model': openai_model,
    }