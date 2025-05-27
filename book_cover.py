import os
import base64
import time
import argparse # Added argparse
from crewai import Agent, Task, Crew
from crewai.tools import tool
import google.generativeai as genai
import google.generativeai as gai # Changed this line
from google.generativeai import types # Changed this line
from crewai.llm import LLM
from config import GEMINI_TEXT_MODEL, GEMINI_VISION_MODEL # Import from config


# Global variable to hold parsed arguments for tools
script_args = None

# These will be initialized in main_script_logic after API key checks
client = None
gemini_llm = None
prompt_generator_agent = None
image_generator_agent = None
crew = None

# Function to store image
def store_image(data, mime_type, filename="image.png"):
    try:
        with open(filename, 'wb') as f:
            f.write(data)
        return f"Image successfully saved to {filename}"
    except Exception as e:
        return f"Error saving image: {e}"

# Define custom tools
@tool("Generate Book Cover Prompt")
def generate_prompt(book_name: str, title: str, author: str) -> str:
    """Generate a highly specific and strict prompt for a book cover image, ensuring title and author inclusion."""
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(GEMINI_TEXT_MODEL) # Use imported constant
    try:
        response = model.generate_content(
            f"""
            You are an expert in crafting precise prompts for AI image generation. Generate a highly specific and strict prompt for a minimal, portrait-oriented book cover image for the book titled '{title}' from the work '{book_name}' with author '{author}'. Follow these rules:
            - The image MUST include the title '{title}' in bold, legible text, centered or positioned near the top for prominence.
            - If the author is provided and not empty, the image MUST include the author name '{author}' in legible text, placed clearly below the title, ensuring no overlap or obstruction.
            - If the author is empty or invalid, exclude the author name entirely; do NOT include any placeholder or default author.
            - The design MUST be minimal but visually appealing, with a simple, uncluttered background relevant to the theme of '{book_name}' (infer the theme if unclear, but keep it subtle).
            - The image MUST NOT be predominantly black, overly dark, or blurry; ensure clarity and vibrancy.
            - The design MUST be balanced: not too minimal (e.g., not just text on a plain background) and not too intricate (e.g., no excessive details or patterns).
            - The image MUST NOT contain any duplicate or additional text (e.g., subtitles, quotes, or random words) beyond the title and author (if provided).
            - Ensure the design is professional, portrait-oriented (taller than wide), and optimized for successful image generation.
            - Return only the prompt text, with no explanations or deviations.
            """
        )
        return response.text
    except Exception as e:
        return f"Error generating prompt: {str(e)}"

@tool("Generate and Validate Book Cover Image")
def generate_and_validate_image(prompt: str, title: str, author: str) -> str:
    """Generate and validate a book cover image iteratively, with strict validation to ensure all criteria are met."""
    max_attempts = 3
    last_error_detail = "No specific error detail captured." # Initialize error detail accumulator for comprehensive reporting.

    for attempt in range(max_attempts):
        output_path = None # Initialize output_path for this attempt. It's set upon successful image saving.
        # Generate image
        try:
            response = client.models.generate_content(
                model=GEMINI_VISION_MODEL, # Use vision model from config.
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["Text", "Image"] # Request both text and image modalities.
                ),
            )
            image_generated = False
            # Process response parts to find and save the image data.
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    img_data = part.inline_data
                    timestamp = int(time.time())
                    image_filename = f"book_cover_{timestamp}.png"
                    # Construct path using output_dir from globally stored script_args.
                    output_path = os.path.join(script_args.output_dir, image_filename)
                    os.makedirs(script_args.output_dir, exist_ok=True) # Ensure the output directory exists.
                    store_result = store_image(img_data.data, img_data.mime_type, output_path)
                    if "Error" in store_result:
                        # If image saving fails, this is a critical error; return immediately.
                        return store_result 
                    image_generated = True
                    break # Image found and saved, exit loop.
            
            if not image_generated:
                last_error_detail = "No image data in response after generation attempt." # Store specific error.
                print(f"Attempt {attempt + 1}: {last_error_detail} Retrying with a reinforced prompt...")
                # Reinforce prompt to include necessary text elements.
                prompt = f"{prompt} Strictly include the title '{title}' and author '{author}' (if provided) in legible text."
                continue # Move to the next attempt.
        except Exception as e:
            last_error_detail = f"Image generation exception: {str(e)}" # Store specific error.
            print(f"Attempt {attempt + 1}: {last_error_detail}. Retrying with a simplified prompt...")
            # Simplify prompt for the next attempt.
            prompt = f"{prompt} Simplify the design, but strictly include the title '{title}' and author '{author}' (if provided) in a portrait-oriented image."
            continue # Move to the next attempt.

        # Validate image
        try:
            # Check if image was successfully saved in the generation step.
            if not output_path or not os.path.exists(output_path): 
                last_error_detail = f"Validation error: Image file {output_path or 'unknown'} not found after generation." # Store specific error.
                print(f"Attempt {attempt + 1}: {last_error_detail}")
                # This indicates a problem in the image saving logic or path construction.
                # Retry generation as the image is missing for validation.
                continue 
            
            # Configure genai for validation model.
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel(GEMINI_TEXT_MODEL) # Use text model for validation.
            with open(output_path, 'rb') as img_file:
                image_data = img_file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            validation_prompt = f"""
            You are an expert in validating book cover images with strict criteria. Analyze the provided image and validate it against the following mandatory requirements. ALL criteria MUST be met for the image to be valid, and any deviation results in failure:
            1. The image contains the title '{title}' EXACTLY as provided, with no spelling mistakes, missing characters, or variations. If the title is missing or incorrect, the image is invalid.
            2. The title is prominently positioned (e.g., centered or near the top), highly legible (clear font, sufficient size, and contrast), and aesthetically balanced within the design. Poor positioning (e.g., too small, off-center, or obscured) invalidates the image.
            3. If an author name '{author}' is provided and not empty, the image contains the author name EXACTLY as specified, with no spelling mistakes or variations. If the author is empty, the image MUST NOT include any author name or placeholder. Any mismatch or unexpected author text invalidates the image.
            4. The author name (if present) is clearly positioned (e.g., directly below the title), highly legible, and not obscured, overlapping, or crowded by other elements. Poor positioning invalidates the image.
            5. The image contains NO duplicate or additional text (e.g., subtitles, quotes, random words, or extra titles/authors) beyond the specified title and author (if provided). ANY extra text, even minor, invalidates the image.
            6. The image is in portrait orientation (taller than wide). Landscape or square images are invalid.
            7. The design is visually appealing, professional, and suitable for a book cover, with a clear, vibrant appearance. The image MUST NOT be predominantly black, overly dark, or blurry.
            8. The design is minimal but balanced: it MUST NOT be too minimal (e.g., just text on a plain background with no visual elements) nor too intricate (e.g., excessive details, cluttered patterns, or complex imagery).
            If ANY criterion is not met, provide a detailed list of ALL issues with specific, actionable feedback (e.g., 'Title missing', 'Author misspelled as X', 'Image is blurry', 'Design too minimal'). If ALL criteria are met exactly, respond with "No issues" and nothing else.
            """
            response = model.generate_content([
                {"text": validation_prompt},
                {"inline_data": {"data": image_base64, "mime_type": "image/png"}}
            ])
            validation_result = response.text
            if validation_result == "No issues":
                # If validation passes, return the success message with the path.
                return f"Final image saved: {output_path}"
            else:
                # If validation fails, store the issues and update prompt for next attempt.
                last_error_detail = f"Validation issues: {validation_result}" # Store specific error.
                print(f"Attempt {attempt + 1}: {last_error_detail}. Regenerating image with feedback...")
                prompt = f"{prompt} Address the following issues: {validation_result}. Strictly include only the title '{title}' and author '{author}' (if provided) in a portrait-oriented, visually appealing, balanced minimal design."
        except Exception as e:
            # Catch any exception during the validation process.
            last_error_detail = f"Validation exception: {str(e)}" # Store specific error.
            print(f"Attempt {attempt + 1}: {last_error_detail}. Retrying with a reinforced prompt...")
            prompt = f"{prompt} Strictly include the title '{title}' and author '{author}' (if provided) in a clear, visually appealing, balanced minimal, portrait-oriented design."
    
    # If all attempts are exhausted, return a failure message with the last known error.
    return f"Failed to generate a valid image after {max_attempts} attempts. Last known error: {last_error_detail}"

# Run the crew
def parse_arguments(argv=None):
    """
    Parses command-line arguments for the book_cover.py script.

    Args:
        argv (list[str], optional): A list of command-line arguments.
                                    Defaults to None, which means sys.argv[1:] will be used.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        prog='book_cover.py', # Sets the program name, used in help messages.
        description="Generate a book cover."
    )
    # Define required arguments for book details.
    parser.add_argument(
        "--book_name", 
        type=str, 
        required=True, 
        help="The name of the book or series for thematic context."
    )
    parser.add_argument(
        "--title", 
        type=str, 
        required=True, 
        help="The specific title to be displayed on the cover."
    )
    parser.add_argument(
        "--author", 
        type=str, 
        required=True, 
        help="The author's name to be displayed on the cover."
    )
    # Define optional argument for specifying the output directory.
    parser.add_argument(
        "-d", "--output_dir", 
        type=str, 
        default="output", # Default directory for saving images.
        help="Directory to save the generated book cover image. Default: output/"
    )
    return parser.parse_args(argv)

def main_script_logic(args):
    """Main logic of the book_cover script after parsing arguments."""
    global script_args, client, gemini_llm, prompt_generator_agent, image_generator_agent, crew
    script_args = args # Assign the parsed args to the global var for tools

    # --- API Key Configuration and Resource Initialization ---
    gemini_api_key_local = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key_local:
        print("Error: The GEMINI_API_KEY environment variable is not set.")
        print("Please set it before running the script.")
        exit(1)
    os.environ["GOOGLE_API_KEY"] = gemini_api_key_local # Used by genai.configure within tools

    client = gai.Client(api_key=gemini_api_key_local)
    gemini_llm = LLM(
        model=GEMINI_TEXT_MODEL,
        api_key=gemini_api_key_local,
    )

    # --- Define Agents (now that LLM is initialized) ---
    prompt_generator_agent = Agent(
        role="Prompt Generator",
        goal="Generate a highly specific and strict prompt for a book cover image that guarantees the inclusion of the title and author (if provided), with no deviations or extra text.",
        backstory="You are an expert in crafting precise prompts for AI image generation, ensuring every detail is clear and optimized to produce professional book covers with exact text and positioning.",
        tools=[generate_prompt],
        llm=gemini_llm,
        verbose=True,
        instructions="""
        - Create a prompt for a book cover image based on the provided book name, title, and author.
        - Ensure the title is included in bold, legible text, centered or near the top for prominence.
        - If the author is provided and not empty, include the author name in legible text, clearly positioned below the title.
        - If the author is empty or invalid, exclude the author name entirely; do not add placeholders or defaults.
        - If the book name or title is ambiguous, infer a subtle, relevant theme, but keep the background minimal.
        - Specify that the design is visually appealing, not predominantly black or blurry, and balanced (not too minimal or too intricate).
        - Strictly prohibit duplicate or additional text (e.g., subtitles, quotes) beyond the title and author.
        - Specify a portrait-oriented, professional design optimized for successful image generation.
        - Avoid vague or ambiguous language that could lead to missing or incorrect text.
        """
    )

    image_generator_agent = Agent(
        role="Image Generator and Validator",
        goal="Generate and strictly validate a book cover image to ensure it meets all criteria exactly, iterating with precise feedback until successful.",
        backstory="You are an AI specialist in creating and validating book covers, with a meticulous eye for detail to ensure exact text, positioning, and design compliance.",
        tools=[generate_and_validate_image],
        llm=gemini_llm,
        verbose=True,
        instructions="""
        - Generate a book cover image using the provided prompt, ensuring the title and author (if provided) are included exactly as specified.
        - Validate the image against strict criteria: exact title presence and positioning, exact author presence and positioning (if provided), no extra text, portrait orientation, visually appealing, and balanced minimal design.
        - If validation fails, analyze the issues and modify the prompt with specific feedback to correct them (e.g., add missing title, fix spelling, enhance clarity).
        - Handle errors (e.g., no image generated, API failures) by reinforcing the prompt to include the required elements and retrying.
        - Iterate until a valid image is produced or the maximum attempts are reached, providing detailed feedback for each attempt.
        - Do not accept an image as valid unless ALL criteria are met exactly.
        """
    )

    # --- Define Tasks (now that agents are initialized) ---
    # Note: Task descriptions use f-strings for book_name, title, author.
    # These will be resolved when crew.kickoff is called with these inputs.
    generate_prompt_task = Task(
        description="""
        Generate a highly specific and strict prompt for a minimal, portrait-oriented book cover image for the book '{book_name}' with title '{title}' and author '{author}'. 
        Ensure the prompt mandates:
        - The title '{title}' in bold, legible text, centered or near the top.
        - The author '{author}' in legible text below the title, only if provided and not empty; otherwise, exclude the author.
        - No duplicate or additional text beyond the specified title and author.
        - A visually appealing, balanced minimal design with a simple, relevant background, not predominantly black or blurry.
        Handle edge cases by excluding invalid or empty authors and inferring a subtle theme if the book name is unclear.
        """,
        expected_output="A precise prompt that ensures the book cover image includes only the specified title and author (if provided), with correct positioning, no extra text, and an appealing design.",
        agent=prompt_generator_agent,
    )

    generate_image_task = Task(
        description="""
        Using the generated prompt, create a book cover image that is minimal, portrait-oriented, visually appealing, and contains exactly the title '{title}' and author '{author}' (if provided) with no spelling mistakes. 
        Strictly validate the image to ensure:
        - The title is present, correct, and prominently positioned.
        - The author name is present, correct, and well-positioned (if provided); absent if not provided.
        - No duplicate or additional text is present.
        - The image is portrait-oriented, clear, vibrant, and balanced in minimalism.
        Iterate with precise feedback to address any issues, retrying up to a maximum number of attempts. 
        Handle errors by reinforcing the prompt to include the required elements.
        """,
        expected_output="The file path of the saved book cover image or a detailed error message if generation fails after multiple attempts.",
        agent=image_generator_agent,
        context=[generate_prompt_task],
    )

    # --- Define Crew (now that tasks are initialized) ---
    crew = Crew(
        agents=[prompt_generator_agent, image_generator_agent],
        tasks=[generate_prompt_task, generate_image_task],
        process="sequential",
        verbose=True,
    )

    try:
        # Kick off the crew execution with the provided inputs.
        result = crew.kickoff(inputs={
            "book_name": args.book_name,
            "title": args.title,
            "author": args.author
        })
        # Check the result from the crew to determine success or failure for user feedback.
        if result and result.startswith("Final image saved:"): # Success condition.
            print(f"Image Generation Process Succeeded: {result}")
        elif result: # If there's a result but it's not the success message, it's an error/failure.
            print(f"Image Generation Process Failed: {result}")
        else: # Handle cases where the result might be None or empty.
            print("Image Generation Process Failed: No result returned from crew.")
            
    except Exception as e:
        print(f"Error running crew: {str(e)}")

if __name__ == "__main__":
    parsed_args = parse_arguments()
    main_script_logic(parsed_args)
