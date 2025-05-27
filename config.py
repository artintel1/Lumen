# Configuration for model names used across the application.
# These constants allow for easy updates to model versions or types.

# GEMINI_TEXT_MODEL: Used for general text generation and understanding tasks,
# such as summarization, narration scripting, and text-based validation.
GEMINI_TEXT_MODEL = "gemini-pro"

# GEMINI_VISION_MODEL: Used for tasks involving image generation or multimodal understanding.
# Specifically, this is leveraged by book_cover.py for generating cover images.
# It was previously "gemini-2.0-flash-exp" and has been standardized to "gemini-pro-vision".
GEMINI_VISION_MODEL = "gemini-pro-vision"

# EMBEDDING_MODEL: Used for generating numerical representations (embeddings) of text segments.
# These embeddings are crucial for tasks like semantic search and clustering in key_idea_finder.py.
EMBEDDING_MODEL = "models/text-embedding-004"
