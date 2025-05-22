import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import textwrap
import fitz # Import PyMuPDF

load_dotenv()

# --- Configure API Keys and LLM ---
# Ensure API keys are loaded from .env
if not os.getenv("SERPER_API_KEY"):
    raise ValueError("SERPER_API_KEY not found in environment variables.")
if not os.getenv("GEMINI_API_KEY"):
     raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Initialize Gemini LLM
gemini_llm = LLM(
    model="gemini/gemini-2.0-flash", # Using a model potentially better for longer text
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.8 # Adjust temperature as needed for creativity/factualness
)

# --- Initialize Tools ---
serper_tool = SerperDevTool()

# --- Define Agents ---

# 1. Narration Generator Agent
narration_agent = Agent(
    role='Podcast Narration Script Writer',
    goal=textwrap.dedent("""
        Generate a comprehensive and engaging audio-only podcast narration script
        for a single host based on provided book content.
        The script should be between 5000 and 10000 words, equivalent to 15 to 25 minutes
        of spoken audio. Focus on storytelling and clear explanations suitable for
        a listening audience.
    """),
    backstory=textwrap.dedent("""
        You are a professional scriptwriter with expertise in adapting written
        material into compelling audio narratives. You understand how to structure
        content for a single host podcast format, keeping the listener engaged
        without visuals.
    """),
    verbose=True,
    allow_delegation=False,
    llm=gemini_llm
)

# 2. Fact-Checking Identifier Agent
fact_identifier_agent = Agent(
    role='Claim Identifier for Fact-Checking',
    goal=textwrap.dedent("""
        Analyze a given narration script and identify only the specific factual claims,
        names, or events that absolutely require external
        verification for accuracy. Be highly selective to minimize unnecessary lookups.
        Output a clear list of these potential claims.
    """),
    backstory=textwrap.dedent("""
        You are a meticulous content analyst trained to spot information that is
        prone to error or requires confirmation from authoritative sources.
        You are cost-conscious and understand the importance of targeted fact-checking.
    """),
    verbose=True,
    allow_delegation=False,
    llm=gemini_llm
)

# 3. Senior Fact Checker Agent
fact_checker_agent = Agent(
    role='Senior Web-Based Fact Checker',
    goal=textwrap.dedent("""
        Verify the factual accuracy of a list of claims using reliable web search
        results via the SerperDevTool. For each claim, provide a clear judgment
        (True, False, Partially True, or Undetermined) and cite the most relevant
        source URLs found.
    """),
    backstory=textwrap.dedent("""
        You are a highly skilled fact checker with extensive experience using search
        engines to validate information. You prioritize credible sources and
        synthesize findings efficiently to provide clear verification results.
    """),
    verbose=True,
    allow_delegation=False,
    tools=[serper_tool],
    llm=gemini_llm
)

# 4. Narrative Corrector Agent
corrector_agent = Agent(
    role='Podcast Script Corrector and Documentarian',
    goal=textwrap.dedent("""
        Revise a podcast narration script based on provided fact-checking results.
        Correct any inaccuracies identified. Additionally, create a separate, detailed
        log of all corrections made, including the original incorrect statement,
        the corrected statement, and the citation(s) that supported the correction.
    """),
    backstory=textwrap.dedent("""
        You are an editor and documentarian ensuring the final output is factually
        sound and transparent about any changes made. You can seamlessly integrate
        corrections into the narrative flow and maintain accurate records.
    """),
    verbose=True,
    allow_delegation=False,
    llm=gemini_llm
)

# --- Define Tasks ---

# Function to read PDF content
def read_pdf(file_path):
    """Reads text content from a PDF file."""
    text = ""
    try:
        document = fitz.open(file_path)
        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            text += page.get_text()
        return text
    except fitz.FileDataError:
        print(f"Error: Cannot open file {file_path}. It might not be a valid PDF.")
        return None
    except Exception as e:
        print(f"Error reading PDF file {file_path}: {e}")
        return None

# Get book path from user
book_path = input("Please enter the path to the book file: ")
book_content = read_pdf(book_path)

if book_content is None:
    exit() # Exit if PDF reading failed

# Task 1: Generate Narration Script
generate_narration_task = Task(
    description=textwrap.dedent(f"""
        Generate a detailed audio-only podcast narration script for a single host
        based on the following book content.
        The script should be between 5000 and 10000 words, formatted for easy reading
        aloud. Ensure it flows well and is engaging for a listener.

        Book Content:
        ---
        {book_content} # Use the content read from the PDF
        ---

        Your Output: A single, comprehensive script for a 15-25 minute audio podcast narration (5000-10000 words).
    """),
    expected_output="A single, comprehensive script for a 15-25 minute audio podcast narration (5000-10000 words).",
    agent=narration_agent,
)

# Task 2: Identify Claims for Verification
identify_claims_task = Task(
    description=textwrap.dedent("""
        Analyze the provided podcast narration script. Extract a precise list of
        only the factual claims, statistics, dates, names, or events that are
        critical to the narrative's accuracy and absolutely require external fact-checking.
        Be highly selective to minimize unnecessary lookups.

        Narration Script:
        ---
        {output_of_generate_narration_task} # This will be populated by the output of the previous task
        ---

        Your Output: A bulleted list of claims to be fact-checked. If no claims require checking, state "No claims require verification."
    """),
    expected_output="A bulleted list of specific claims extracted from the narration that need fact-checking, or a statement indicating none are found.",
    agent=fact_identifier_agent,
)

# Task 3: Verify Identified Claims
verify_claims_task = Task(
    description=textwrap.dedent("""
        For each claim in the provided list of claims, use the SerperDevTool
        to search for supporting or contradicting information.
        Synthesize the search results to determine the accuracy of each claim.

        List of Claims to Verify:
        ---
        {output_of_identify_claims_task} # This will be populated by the output of the previous task
        ---

        Your Output: For each claim, provide the original claim, your judgment on its accuracy (True, False, Partially True, or Undetermined), and cite the top 1-3 relevant source URLs from your search results. Format this as a clear list.
    """),
    expected_output="A list of verification results, where each item includes the claim, accuracy judgment (True/False/Partially True/Undetermined), and source URLs.",
    agent=fact_checker_agent,
)

# Task 4: Correct Narration and Document Changes
correct_narration_task = Task(
    description=textwrap.dedent("""
        Review the original narration script and the fact-checking verification results.
        Make necessary corrections to the narration script based on the verification findings.
        Ensure the corrected narration flows naturally.

        Additionally, create a separate, detailed log of *only* the corrections made.
        For each correction in the log, include:
        - The original incorrect statement from the script.
        - The corrected statement.
        - The source URL(s) that supported the correction.

        Original Narration Script:
        ---
        {output_of_generate_narration_task} # This will be the output from generate_narration_task
        ---

        Fact-Checking Results:
        ---
        {output_of_verify_claims_task} # This will be the output from verify_claims_task
        ---

        Your Output: First, provide the fully corrected podcast narration script. Then, after a clear separator (e.g., "--- Corrections Log ---"), provide the detailed log of changes made.
    """),
    expected_output="The corrected podcast narration script, followed by a separator and a detailed log of all corrections with sources.",
    agent=corrector_agent,
)

# --- Create the Crew ---
podcast_fact_checking_crew = Crew(
    agents=[
        narration_agent,
        fact_identifier_agent,
        fact_checker_agent,
        corrector_agent
    ],
    tasks=[
        generate_narration_task,
        identify_claims_task,
        verify_claims_task,
        correct_narration_task
    ],
    process=Process.sequential,
    verbose=True
)

# --- Run the Crew ---
print("## Running Podcast Narration Fact-Checking Crew")

# The kickoff method starts the process.
# The book_content is already included in the description of the first task.
result = podcast_fact_checking_crew.kickoff()

print("\n## Crew Run Complete")
print("--- Final Output (Corrected Narration and Corrections Log) ---")
print(result)

# --- Save the Final Output to a Markdown File ---
output_filename = "corrected_narration_and_log.md"
print(f"\nSaving the final output to {output_filename}...")

try:
    # Open the file in write mode ('w') and specify utf-8 encoding
    with open(output_filename, "w", encoding="utf-8") as f:
        # Write the entire result string to the file
        f.write(result.raw)
    print(f"Successfully saved output to {output_filename}")

except IOError as e:
    print(f"Error saving file {output_filename}: {e}")
