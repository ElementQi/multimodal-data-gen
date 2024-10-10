import json
import argparse
from openai import OpenAI
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rewrite conversations using OpenAI API."
    )
    parser.add_argument("--api_key", required=True, help="API key for OpenAI")
    parser.add_argument("--base_url", required=True, help="Base URL for OpenAI API")
    parser.add_argument(
        "--input_file",
        required=True,
        help="Input file containing the dataset in JSON format",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output file to save the rewritten conversations",
    )
    parser.add_argument("--model_name", required=True, help="OpenAI api model name")
    return parser.parse_args()


def rewrite_conversation(conversations, client, system_prompt, model_name):
    rewritten_conversations = []
    for idx, conversation in enumerate(conversations):
        message_to_rewrite = conversation["value"]
        is_human = conversation["from"] == "human"


        # Modify system prompt based on whether it's a question or answer
        role_specific_prompt = (
            system_prompt + "\nTreat this as a question and maintain the question format."
            if is_human
            else system_prompt + "\nTreat this as an answer and maintain an answering tone."
        )

        # API call to rewrite the conversation
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": role_specific_prompt},
                {"role": "user", "content": message_to_rewrite},
            ],
        )

        message_value = response.choices[0].message.content

        # Replace the original conversation with the rewritten one
        # For the first human message, ensure it includes "<image>"
        if idx == 0 and conversation["from"] == "human":
            if "<image>" in message_to_rewrite:
                message_value = f"<image>\n{message_value}"

        rewritten_conversations.append(
            {"from": conversation["from"], "value": message_value}
        )

    return rewritten_conversations


def main():
    args = parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    with open(args.input_file, "r") as file:
        data = json.load(file)

    system_prompt = """
    You are an AI visual assistant that is observing an image. Your task is to rewrite both the human questions and the assistant's responses in a more detailed, natural, and informative way. 
    For human entries (questions), rewrite them as questions while maintaining the question format and tone. For assistant responses, maintain a confident and informative tone.
    Ensure that the conversations sound natural and are based on visual reasoning from the image, while avoiding overly lengthy responses. The dialogue should retain its original meaning but become more engaging and insightful.
    """

    rewritten_data = []
    for item in tqdm(data):
        rewritten_conversations = rewrite_conversation(
            item["conversations"], client, system_prompt, args.model_name
        )
        rewritten_data.append(
            {
                "id": item["id"],
                "image": item["image"],
                "conversations": rewritten_conversations,
            }
        )

    with open(args.output_file, "w") as outfile:
        json.dump(rewritten_data, outfile, indent=2)

    print(f"Rewriting complete and saved to '{args.output_file}'")


if __name__ == "__main__":
    main()
