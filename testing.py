import time
from gpt4all import GPT4All
import torch
import asyncio

# Check for GPU availability and set the device
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda"
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner for better performance
else:
    print("CUDA is not available. Falling back to CPU.")
    device = "cpu"

# Define model name and path
model_name = "Meta-LLaMA-3-8B-Instruct.Q4_0.gguf"
model_path = "D:/Projects/Ror/ror/models"

try:
    # Start measuring total execution time
    total_start_time = time.time()

    # Initialize GPT4All model once to avoid reloading
    print("Loading the model, please wait...")
    start_loading_time = time.time()
    model = GPT4All(
        model_name=model_name, 
        model_path=model_path, 
        allow_download=False, 
        device=device
    )
    end_loading_time = time.time()
    print(f"Model loaded successfully! Loading time: {end_loading_time - start_loading_time:.2f} seconds\n")

    # Asynchronous function to generate a response
    async def generate_response(prompt):
        return model.generate(prompt, max_tokens=300)

    # Main asynchronous function to handle multiple prompts
    async def main():
        prompts = [
            "Explain how AI can help with optimizing training routine for professional athletes?",
            "Explain how AI helps small businesses in key areas like customer support?",
            "Explain how AI helps people with mental disorders?"
        ]
        tasks = [generate_response(prompt) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
        for i, response in enumerate(responses):
            print(f"\nResponse {i + 1}: {response}")

    # Run the asynchronous main function
    asyncio.run(main())

    # End measuring total execution time
    total_end_time = time.time()
    print(f"\nTotal execution time: {total_end_time - total_start_time:.2f} seconds")

except Exception as e:
    # Handle any errors gracefully
    print(f"An error occurred: {e}")
    exit(1)  # Exit with a non-zero status to indicate failure


###import time
###from gpt4all import GPT4All
###import torch
###
#### Check for GPU availability and set the device
###if torch.cuda.is_available():
###    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
###    device = "cuda"
###    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner for better performance
###else:
###    print("CUDA is not available. Falling back to CPU.")
###    device = "cpu"
###
#### Define model name and path
###model_name = "Meta-LLaMA-3-8B-Instruct.Q4_0.gguf"
###model_path = "D:/Projects/Ror/ror/models"
###
###try:
###    # Initialize GPT4All model once to avoid reloading
###    print("Loading the model, please wait...")
###    start_loading_time = time.time()
###    model = GPT4All(
###        model_name=model_name, 
###        model_path=model_path, 
###        allow_download=False, 
###        device=device
###    )
###    end_loading_time = time.time()
###    print(f"Model loaded successfully! Loading time: {end_loading_time - start_loading_time:.2f} seconds\n")
###
###    # Enter a loop for querying the model
###    while True:
###        # Prompt user for input
###        prompt = input("Enter your prompt (or type 'exit' to quit): ")
###        if prompt.lower() == "exit":
###            print("Exiting...")
###            break
###        
###        # Measure response generation time
###        start_time = time.time()
###        response = model.generate(
###            prompt, 
###            max_tokens=300, 
###            n_batch=16, 
###            top_k=50, 
###            top_p=0.9  # Optimized sampling parameters
###        )
###        end_time = time.time()
###
###        # Display the response and time taken
###        print("\nResponse:")
###        print(response)
###        print(f"\nTime taken for response: {end_time - start_time:.2f} seconds\n")
###
###except Exception as e:
###    # Handle any errors gracefully
###    print(f"An error occurred: {e}")
###    exit(1)  # Exit with a non-zero status to indicate failure
###