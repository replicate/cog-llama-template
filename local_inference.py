import json
import click
import requests

@click.command()
@click.option('--prompt', default='Write a program that sums 2 integers together and returns the results.')
@click.option('--system_prompt', default='You are the worlds greatest programmer, #1 on leetcode, and you always follow instructions exactly.')
@click.option('--max_new_tokens', default=512)
@click.option('--min_new_tokens', default=-1)
@click.option('--temperature', default=0.6)
@click.option('--top_p', default=0.95)
@click.option('--top_k', default=50)
def make_request(prompt, system_prompt, max_new_tokens, min_new_tokens, temperature, top_p, top_k):
    url = "http://0.0.0.0:5000/predictions"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "input": {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        output = json.loads(response.text)['output']
        output_str = "".join(output)
        click.echo(output_str)
    else:
        click.echo(f"Something went wrong: {response.status_code}")

if __name__ == "__main__":
    make_request()
