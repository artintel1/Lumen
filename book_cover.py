import os
import base64
import time
from crewai import Agent, Task, Crew
from crewai.tools import tool
import google.generativeai as genai
from google import genai as gai
from google.genai import types
from crewai.llm import LLM

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")  # Replace with your actual Gemini API key

#this client is used for image genration later
client = gai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

#gemini LLM
gemini_llm = LLM(
    model="gemini/gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

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
    model = genai.GenerativeModel('gemini-2.0-flash')
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
    for attempt in range(max_attempts):
        # Generate image
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["Text", "Image"]
                ),
            )
            image_generated = False
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    img_data = part.inline_data
                    timestamp = int(time.time())
                    output_path = f"output/book_cover_{timestamp}.png"
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    store_result = store_image(img_data.data, img_data.mime_type, output_path)
                    if "Error" in store_result:
                        return store_result
                    image_generated = True
                    break
            if not image_generated:
                print(f"Attempt {attempt + 1}: No image data in response. Retrying with a reinforced prompt...")
                prompt = f"{prompt} Strictly include the title '{title}' and author '{author}' (if provided) in legible text."
                continue
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error generating image: {str(e)}. Retrying with a simplified prompt...")
            prompt = f"{prompt} Simplify the design, but strictly include the title '{title}' and author '{author}' (if provided) in a portrait-oriented image."
            continue

        # Validate image
        try:
            if not os.path.exists(output_path):
                return f"Attempt {attempt + 1}: Validation error: Image file {output_path} not found."
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel('gemini-2.0-flash')
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
                return f"Final image saved: {output_path}"
            else:
                print(f"Attempt {attempt + 1}: Issues found - {validation_result}. Regenerating image with feedback...")
                prompt = f"{prompt} Address the following issues: {validation_result}. Strictly include only the title '{title}' and author '{author}' (if provided) in a portrait-oriented, visually appealing, balanced minimal design."
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error validating image: {str(e)}. Retrying with a reinforced prompt...")
            prompt = f"{prompt} Strictly include the title '{title}' and author '{author}' (if provided) in a clear, visually appealing, balanced minimal, portrait-oriented design."
    
    return f"Failed to generate a valid image after {max_attempts} attempts. Consider revising the input or checking model capabilities."

# Define agents with strict instructions
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

# Define tasks with strict descriptions
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

# Define crew
crew = Crew(
    agents=[prompt_generator_agent, image_generator_agent],
    tasks=[generate_prompt_task, generate_image_task],
    process="sequential",
    verbose=True,
)

# Run the crew
def main():
    book_name = "Blotzmann Machines"
    title = "Boltzmann Machines"
    author = "Jeffery Hinton"
    try:
        result = crew.kickoff(inputs={"book_name": book_name, "title": title, "author": author})
        print(f"Result for '{book_name}': {result}")
    except Exception as e:
        print(f"Error running crew: {str(e)}")

if __name__ == "__main__":
    main()
