<div align="center">
<pre>
L       U   U  M   M  EEEEE  N   N
L       U   U  MM MM  E      NN  N
L       U   U  M M M  EEE    N N N
L       U   U  M   M  E      N  NN
LLLLL   UUUUU  M   M  EEEEE  N   N
</pre>
</div>

# AI-Powered Content Generation for Books

## Overview

This repository contains a suite of Python scripts designed to transform books into various content formats using AI. It helps you generate engaging podcast/Ted Talk scripts, create custom cover images for books, and extract key ideas to understand a book's core concepts quickly.

## Features

*   **Podcast/Ted Talk Script Generation:** Creates detailed, fact-checked, and corrected narration scripts suitable for audio-only formats.
*   **Custom Book Cover Image Generation:** Generates unique, minimal, and visually appealing book cover images based on book title and author.
*   **Key Idea Extraction:** Identifies and ranks the top key ideas from a book to provide a quick understanding of its main themes.

## File Structure

*   `narration.py`: Generates, fact-checks, and corrects narration scripts for podcasts or Ted Talks.
*   `book_cover.py`: Creates custom book cover images using AI image generation.
*   `key_idea_finder.py`: Reads a book (PDF or TXT), extracts text chunks, embeds them, clusters them to find themes, summarizes these themes, and ranks them by relevance to the entire book.
*   `config.py`: Contains configuration for model names (e.g., `GEMINI_TEXT_MODEL`, `EMBEDDING_MODEL`) used by the scripts.
*   `utils.py`: Provides shared utility functions, such as file reading (PDF, TXT) and text chunking logic, used by various scripts.
*   `requirements.txt`: Lists all the necessary Python packages for running the scripts.
*   `output/`: Default directory created by `book_cover.py` to store generated images.
*   `tests/`: Contains unit tests for the project. Includes `test_cli.py` for CLI argument parsing and `test_utils.py` for utility functions. Sample files for testing are in `tests/sample_files/`.

## Prerequisites

*   **Python 3.x:** Ensure you have Python 3 installed (developed with 3.10+).
*   **API Keys & Configuration:**
    *   **Environment Variables:**
        *   `GEMINI_API_KEY`: For accessing Google's Gemini models (used in all scripts). Obtain this from Google AI Studio.
        *   `SERPER_API_KEY`: For web search capabilities used in the fact-checking part of `narration.py`. Get this from [Serper.dev](https://serper.dev).
        These scripts load API keys from environment variables. You can set them in your system or create a `.env` file in the root of the project with the following content:
        ```
        GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
        SERPER_API_KEY="YOUR_SERPER_API_KEY"
        ```
        The scripts use `python-dotenv` to load these variables and include checks to ensure they are set before running core logic.
    *   **Model Configuration (`config.py`):**
        Model names (e.g., for different Gemini versions or embedding models) are centralized in `config.py`. You can modify these constants if needed, for example, to use a newer model version.
*   **Command-Line Arguments**: The scripts now use `argparse` for handling command-line arguments, providing a more flexible way to specify inputs and outputs.

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## How to Run

Make sure you have set up your API keys as environment variables as described in the "Prerequisites" section.

### 1. Narration Script Generation (`narration.py`)

This script generates a podcast/Ted Talk style narration from a book.

*   **Usage:**
    ```bash
    python narration.py <path_to_book_file> [--output_file <output_path.md>]
    ```
    *   `<path_to_book_file>`: (Required) Path to the input book file (PDF or TXT).
    *   `--output_file <output_path.md>` or `-o <output_path.md>`: (Optional) Path to save the final corrected narration and log. Defaults to `corrected_narration_and_log.md` in the current directory.
*   **Example:**
    ```bash
    python narration.py "my_book.pdf" -o "narration_output/my_book_narration.md"
    ```
*   **Output:** A Markdown file containing the final corrected narration script and a log of any fact-checking corrections made, saved to the path specified by `--output_file`.

### 2. Custom Book Cover Generation (`book_cover.py`)

This script generates a custom cover image for a book.

*   **Usage:**
    ```bash
    python book_cover.py --book_name "Name of the Book Series or Work" --title "Actual Book Title" --author "Author's Name" [--output_dir <directory_path>]
    ```
    *   `--book_name <name>`: (Required) The name of the book series or general work (used for thematic context).
    *   `--title <title>`: (Required) The specific title to be displayed on the book cover.
    *   `--author <author>`: (Required) The author's name to be displayed on the book cover.
    *   `--output_dir <directory_path>` or `-d <directory_path>`: (Optional) Directory where the generated image will be saved. Defaults to `output/`.
*   **Example:**
    ```bash
    python book_cover.py --book_name "The Expanse" --title "Leviathan Wakes" --author "James S.A. Corey" -d "generated_covers/"
    ```
*   **Output:** An image file (e.g., `book_cover_{timestamp}.png`) will be saved in the specified output directory. The script creates this directory if it doesn't exist.

### 3. Key Idea Extraction (`key_idea_finder.py`)

This script extracts and ranks key ideas from a book.

*   **Usage:**
    ```bash
    python key_idea_finder.py <path_to_book_file> [--output_file <output_path.md>]
    ```
    *   `<path_to_book_file>`: (Required) Path to the input book file (PDF or TXT).
    *   `--output_file <output_path.md>` or `-o <output_path.md>`: (Optional) Path to save the ranked key ideas in Markdown format. If not provided, output is printed to the console.
*   **Example:**
    ```bash
    python key_idea_finder.py "path/to/my_document.pdf" -o "key_ideas/document_ideas.md"
    ```
*   **Output:** If `--output_file` is specified, a Markdown file containing the ranked key ideas, their summaries, and similarity scores. Otherwise, this information is printed to the console.

## Running Tests

The project includes unit tests to verify functionality.
To run tests:
```bash
python -m unittest discover tests
```
Or run a specific test file:
```bash
python -m unittest tests/test_utils.py
python -m unittest tests/test_cli.py
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please feel free to:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Open a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
