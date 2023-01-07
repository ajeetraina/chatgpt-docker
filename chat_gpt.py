import argparse
import torch

from transformers import GPT2Tokenizer, GPT2LMHeadModel

def main(args):
  # Load the model and tokenizer
  model = GPT2LMHeadModel.from_pretrained(args.model_name)
  tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)

  # Process input data and generate output
  input_text = "Hello, how are you?"
  input_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0)  # Batch size 1
  output = model.generate(input_ids)
  output_text = tokenizer.decode(output[0], skip_special_tokens=True)
  print(output_text)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name", type=str, required=True,
                      help="Name of the pre-trained model to use.")
  args = parser.parse_args()
  main(args)
