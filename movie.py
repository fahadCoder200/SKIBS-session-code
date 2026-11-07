import os
from openai import OpenAI
import json

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    max_retries=3
)

data_base = [
    [
    {
        "name": "One Rhino King",
        "no_of_episodes": 55,
        "description": "A Rhino wants to rule a jungle that is ruled by a leopard, leopard is a violent one so Rhino neds to gather strong friends to fight and defeat it."
    },
    {
        "name": "Koruto",
        "no_of_episodes": 205,
        "description": "A Ninja is lost in a universe where time is not present, he must fill his hollow timeless life with weird or useless activity to avoid becoming a rock(which he will become if he does nothing)."
    },
    {
        "name": "Sneach",
        "no_of_episodes": 12,
        "description": "A short action thriller where a small heighted guy fights tall men to take revenge for his killed brother."
    }
    ]
]

def get_names(genre: str):
    names = []
    if genre == "action":
        adv_list = data_base[0]
        for i in adv_list:
            names.append(i["name"])

    return names


TOOL_CALLS = [
    {
        "type": "function",
        "function": {
            "name": "get_names",
            "description": "This returns a list of all the anime names from the database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "genre": {
                        "type": "string",
                        "enum": ["action", "adventure"]
                    }
                }
            }
        }
    }
]

TOOL_IMPL = {
    "get_names": get_names
}

messages = [
    {"role": "system", "content": "You are a movie recommendor chat bot, use the functions given to give answers and guess the anime the user is talking about, do not go out of the names provided!"},
    {"role": "user", "content": "I want an action anime, what do you have?"}
]

response = client.chat.completions.create(
    messages=messages,
    model="openai/gpt-oss-120b",
    tool_choice="auto",
    tools=TOOL_CALLS,
    reasoning_effort="low",
    stream=False
)

msg = response.choices[0].message

if msg.tool_calls:
    # Echo the model's tool request into the history
    messages.append({
        "role": "assistant",
        "content": msg.content,
        "tool_calls": [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.function.name, "arguments": c.function.arguments},
            }
            for c in msg.tool_calls
        ],
    })

    # Run each requested tool and send the result back
    for call in msg.tool_calls:
        args = json.loads(call.function.arguments or "{}")
        result = TOOL_IMPL[call.function.name](**args)
        print(result)
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result),
        })

    # Second call: the model now sees the tool result and writes the answer
    response = client.chat.completions.create(
        messages=messages,
        model="openai/gpt-oss-120b",
        tools=TOOL_CALLS,
        reasoning_effort="low",
        stream=False,
    )
    print(response.choices[0].message.content)

else:
    print(msg.content)
