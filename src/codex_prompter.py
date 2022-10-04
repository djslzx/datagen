"""
Programmatically prompt Codex to generate data from a small starting dataset.
Examine the generated data's statistics to understand how far we can go with this idea.
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

if __name__ == '__main__':
    with open("codex/prompt.txt", "r") as f:
        prompt = "".join(f.readlines())
        print(prompt)

    if not prompt:
        exit(1)

    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=10,
        temperature=0.9,
    )
    # continuation = response["text"]


    print(response)
