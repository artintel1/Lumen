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
*   `requirements.txt`: Lists all the necessary Python packages for running the scripts.
*   `output/`: This directory will be created by `book_cover.py` to store generated images. (Note: You might need to create this directory manually if it doesn't exist before running `book_cover.py` for the first time, or the script should handle its creation, which it does by using `os.makedirs`).

## Prerequisites

*   **Python 3.x:** Ensure you have Python 3 installed.
*   **API Keys:**
    *   `GEMINI_API_KEY`: For accessing Google's Gemini models (used in all scripts). You can obtain this from Google AI Studio.
    *   `SERPER_API_KEY`: For web search capabilities used in the fact-checking part of `narration.py`. You can get this from [Serper.dev](https://serper.dev).
*   **Setting Environment Variables:**
    These scripts load API keys from environment variables. You can set them in your system or create a `.env` file in the root of the project with the following content:
    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    SERPER_API_KEY="YOUR_SERPER_API_KEY"
    ```
    The scripts use `python-dotenv` to load these variables.

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

Make sure you have set up your API keys as environment variables.

### 1. Narration Script Generation (`narration.py`)

This script generates a podcast/Ted Talk style narration from a book.

*   **Usage:**
    Run the script from your terminal. It will prompt you to enter the path to the book file (PDF).
    ```bash
    python narration.py
    ```
*   **Input:** Path to a book PDF file.
*   **Output:** A Markdown file named `corrected_narration_and_log.md` will be created in the root directory. This file contains the final corrected narration script and a log of any fact-checking corrections made.

### 2. Custom Book Cover Generation (`book_cover.py`)

This script generates a custom cover image for a book.

*   **Usage:**
    Currently, the book title and author are hardcoded in the `main()` function of `book_cover.py`.
    ```python
    # Inside book_cover.py
    def main():
        book_name = "Blotzmann Machines" # You can change this
        title = "Boltzmann Machines"    # You can change this
        author = "Jeffery Hinton"       # You can change this
        # ... rest of the function
    ```
    To generate a cover for a different book, you'll need to modify these variables directly in the script.
    Run the script:
    ```bash
    python book_cover.py
    ```
*   **Output:** An image file (e.g., `book_cover_{timestamp}.png`) will be saved in the `output/` directory. The script creates this directory if it doesn't exist.

### 3. Key Idea Extraction (`key_idea_finder.py`)

This script extracts and ranks key ideas from a book.

*   **Usage:**
    Run the script from your terminal. It will prompt you to enter the path to the book file (PDF or TXT).
    ```bash
    python key_idea_finder.py
    ```
*   **Input:** Path to a book file (PDF or TXT).
*   **Output:** The script will print the ranked key ideas, their summaries, and similarity scores directly to the console.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please feel free to:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Open a Pull Request.

## License

This project is open-source. Please refer to the `LICENSE` file if one is added to the repository. (Currently, no license file is specified).
