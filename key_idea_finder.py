import os
import google.generativeai as genai
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import time
import warnings
import fitz  # PyMuPDF
import math # Import math for checking nan

# --- Configuration ---
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
# Gemini 1.5 Flash is good for summarization tasks - check Google AI docs for latest stable name
SUMMARIZATION_MODEL_NAME = "models/gemini-2.0-flash"
NUM_CLUSTERS = 7 # Adjust as needed (desired number of key ideas)
MIN_CHUNK_LENGTH = 400
# Number of closest chunks to the centroid to use for summarization
NUM_CLOSEST_CHUNKS_TO_SUMMARIZE = 7
# Maximum combined length of chunks to send for summarization (to avoid hitting token limits)
MAX_SUMMARY_INPUT_LENGTH = 50000 # Adjust based on model limits and typical chunk size

# --- Helper Functions ---

def read_book(filepath):
    """Reads text content from a PDF or plain text file."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None

    # Handle PDF file using PyMuPDF (fitz)
    if filepath.lower().endswith('.pdf'):
        try:
            print(f"Attempting to extract text from PDF: {filepath}")
            doc = fitz.open(filepath)
            full_text = ""
            num_pages = doc.page_count
            print(f"PDF has {num_pages} pages.")
            for i, page in enumerate(doc):
                # Use 'text' for standard text extraction
                # 'html', 'xml', 'json', 'text' are options
                text = page.get_text("text")
                if text:
                    full_text += text + "\n"
            doc.close()
            if not full_text.strip():
                 print("Warning: No text could be extracted from the PDF.")
                 return None
            print(f"Successfully extracted text from PDF.")
            return full_text
        except fitz.fitz.PasswordError:
             print(f"Error: The PDF file '{filepath}' is password-protected.")
             return None
        except Exception as e:
            print(f"Error extracting text from PDF '{filepath}': {e}")
            return None
    # Handle Plain Text File
    else:
        try:
            print(f"Reading plain text file: {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            try:
                print(f"UTF-8 decoding failed ({e}), trying latin-1 encoding...")
                with open(filepath, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e2:
                 print(f"Error reading plain text file with UTF-8 and latin-1: {e2}")
                 return None

def chunk_text_by_paragraph(text, min_length=MIN_CHUNK_LENGTH):
    """Chunks the text by paragraphs and filters short chunks."""
    # Split by multiple newlines (more likely to be paragraphs)
    paragraphs = text.split('\n\n')
    # If very few paragraphs, try splitting by single newlines
    if len(paragraphs) < 10 and len(text.split('\n')) > 20: # Heuristic: check if single newlines are more common
        print("Few paragraphs found with '\\n\\n', trying single '\\n' splits.")
        paragraphs = text.split('\n')
    elif len(paragraphs) < 10:
         print("Few paragraphs found even with single '\\n' splits or short text overall.")


    chunks = [p.strip() for p in paragraphs if len(p.strip()) >= min_length]

    # Fallback: if no chunks found, try splitting by sentences (less ideal but better than nothing)
    if not chunks and len(text.strip()) >= min_length:
         print(f"Warning: No text chunks found by paragraph splitting/filtering (min length {min_length}). Attempting sentence splitting.")
         # Basic sentence splitting (can be improved)
         import re
         sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text.strip())
         # Combine sentences until min_length is reached
         current_chunk = ""
         for sentence in sentences:
             if len(current_chunk) + len(sentence) + 1 >= min_length:
                 if current_chunk: # Only add if not empty
                    chunks.append(current_chunk.strip())
                 current_chunk = sentence + " "
             else:
                 current_chunk += sentence + " "
         if current_chunk and len(current_chunk.strip()) >= min_length:
              chunks.append(current_chunk.strip())
         elif current_chunk and chunks: # Append remaining if it's not too short and we have other chunks
              chunks[-1] += " " + current_chunk.strip() # Append to last chunk

    if not chunks:
        print(f"Warning: No valid text chunks found after splitting/filtering (min length {min_length}).")
        return []

    print(f"Chunked text into {len(chunks)} sections (min length {min_length}).")
    return chunks

def get_embeddings(texts, model_name, api_key):
    """Generates embeddings using the Gemini API. Handles both single string and list of strings."""
    if not api_key:
        raise ValueError("API key not found. Set the GOOGLE_API_KEY environment variable.")
    genai.configure(api_key=api_key)
    print(f"Using embedding model: {model_name}")

    is_single_text = isinstance(texts, str)
    texts_list = [texts] if is_single_text else texts

    if not texts_list:
         print("Error: No text provided for embedding.")
         return None

    embeddings = []
    batch_size = 100
    retries = 3
    delay = 5

    for i in range(0, len(texts_list), batch_size):
        batch_texts = texts_list[i:i + batch_size]
        print(f"Generating embeddings for batch {i // batch_size + 1}/{ (len(texts_list) + batch_size - 1) // batch_size }...")
        for attempt in range(retries):
            try:
                result = genai.embed_content(
                    model=model_name,
                    content=batch_texts,
                    task_type="RETRIEVAL_DOCUMENT" # Or "SEMANTIC_SIMILARITY" or "AQA" depending on task
                )
                # Ensure the result has the expected structure
                if result and 'embedding' in result and isinstance(result['embedding'], list):
                    embeddings.extend(result['embedding'])
                else:
                    print(f"Warning: Unexpected embedding result structure for batch {i // batch_size + 1}. Result keys: {result.keys() if result else 'None'}")
                    # Optionally, inspect result['embedding'] structure if present
                    # if 'embedding' in result and result['embedding']: print(f"First element type: {type(result['embedding'][0])}")
                    pass # Decide how to handle this - skip batch?

                print(f"Successfully processed batch {i // batch_size + 1}.")
                time.sleep(0.5) # Small delay
                break
            except Exception as e:
                print(f"Error embedding batch {i // batch_size + 1}, attempt {attempt + 1}/{retries}: {e}")
                if "API key not valid" in str(e): return None
                if "rate limit" in str(e).lower():
                     print("Rate limit hit. Waiting longer before retrying.")
                     time.sleep(delay * (attempt + 2)) # Exponential backoff
                elif attempt < retries - 1: time.sleep(delay)
                else: print(f"Failed to embed batch {i // batch_size + 1}. Skipping.")
                if "400 Bad Request" in str(e) and "text is too long" in str(e).lower():
                     print("Error: Input text is too long for the embedding model.")
                     return None # Fatal error for this batch/input


    if not embeddings:
         print("Error: No embeddings were generated.")
         return None

    # Handle cases where the number of generated embeddings doesn't match input texts
    if len(embeddings) != len(texts_list):
        print(f"Warning: Embeddings count ({len(embeddings)}) != text count ({len(texts_list)}). Some embeddings may have failed.")
        # A more robust implementation might return a list of (text, embedding) or None for failed ones

    # Convert to numpy array
    embeddings_np = np.array(embeddings, dtype=np.float32) # Use float32 for potentially smaller memory footprint

    print(f"Generated {len(embeddings)} embeddings with shape {embeddings_np.shape}.")

    # If the input was a single string, return a single embedding vector
    if is_single_text:
        return embeddings_np[0] if embeddings_np.shape[0] > 0 else None
    # Otherwise, return the array of embeddings
    return embeddings_np


def cluster_embeddings(embeddings, n_clusters):
    """Performs K-Means clustering."""
    if embeddings is None or embeddings.shape[0] < n_clusters:
        print(f"Error: Cannot cluster. Need >= {n_clusters} data points, got {embeddings.shape[0] if embeddings is not None else 0}.")
        # Adjust n_clusters if embeddings count is less but > 0
        if embeddings is not None and embeddings.shape[0] > 0:
            print(f"Adjusting n_clusters to {embeddings.shape[0]} as there are fewer embeddings than requested clusters.")
            n_clusters = embeddings.shape[0]
        else:
             return None # Cannot cluster with no data

    print(f"Performing K-Means clustering with {n_clusters} clusters...")
    # Explicitly set n_init to 'auto' or an integer (e.g., 10) to avoid warning in future sklearn versions
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    with warnings.catch_warnings():
         warnings.simplefilter("ignore", category=UserWarning)
         kmeans.fit(embeddings)
    print("Clustering complete.")
    return kmeans

def generate_summaries_for_clusters(kmeans, embeddings, original_texts, api_key,
                                    summarization_model_name, num_closest=5, max_input_len=15000):
    """
    Generates a summary for each cluster using Gemini, based on the text
    of the N closest chunks to the centroid.
    """
    if kmeans is None or embeddings is None or len(original_texts) != embeddings.shape[0]:
        print("Error: Invalid input for summarizing clusters.")
        return []

    if not api_key:
        print("Error: API key not provided for summarization.")
        return []

    print(f"\nInitializing summarization model: {summarization_model_name}")
    genai.configure(api_key=api_key)
    try:
        # Configure safety settings to be less restrictive if needed, otherwise defaults are used
        # safety_settings = [
        #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        # ]
        model = genai.GenerativeModel(summarization_model_name)#, safety_settings=safety_settings)
    except Exception as e:
        print(f"Error initializing generative model: {e}")
        return []

    summaries = []
    n_clusters = kmeans.n_clusters
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    print(f"Generating summaries for {n_clusters} clusters using top {num_closest} chunks each...")

    for i in range(n_clusters):
        print(f"\n--- Processing Cluster {i+1}/{n_clusters} ---")
        cluster_indices = np.where(labels == i)[0]

        if len(cluster_indices) == 0:
            print("Skipping empty cluster.")
            summaries.append({"cluster_id": i, "summary": "Cluster {i+1}: Empty cluster.", "centroid_embedding": centroids[i]})
            continue

        cluster_embeddings = embeddings[cluster_indices]
        cluster_texts = [original_texts[idx] for idx in cluster_indices]
        centroid = centroids[i]

        # Calculate distances within the cluster
        distances = euclidean_distances(cluster_embeddings, centroid.reshape(1, -1)).flatten()

        # Get indices of the N closest points *within the cluster_indices array*
        num_to_select = min(num_closest, len(distances))
        # Check if distances is empty or num_to_select is 0
        if num_to_select == 0 or len(distances) == 0:
             print(f"Warning: No distances calculated or no points to select for cluster {i+1}.")
             summaries.append({"cluster_id": i, "summary": f"Cluster {i+1}: Could not select closest chunks for summarization.", "centroid_embedding": centroids[i]})
             continue

        closest_indices_in_cluster = np.argsort(distances)[:num_to_select]

        # Get the actual text of the closest chunks
        closest_texts = [cluster_texts[idx] for idx in closest_indices_in_cluster]

        # Combine text for summarization prompt
        combined_text = "\n\n---\n\n".join(closest_texts)

        # Truncate if combined text is too long
        if len(combined_text) > max_input_len:
            print(f"Warning: Combined text for cluster {i+1} exceeds max length ({max_input_len}). Truncating.")
            combined_text = combined_text[:max_input_len]

        if not combined_text.strip():
             print(f"Warning: No text content found for closest chunks in cluster {i+1}. Skipping summarization.")
             summaries.append({"cluster_id": i, "summary": f"Cluster {i+1}: No text content found for summarization.", "centroid_embedding": centroids[i]})
             continue

        # Define the prompt for the Gemini model
        prompt = f"""The following are text excerpts from a larger document that seem to relate to a common theme or topic:

--- START EXCERPTS ---
{combined_text}
--- END EXCERPTS ---

Based *only* on the excerpts provided above, please synthesize and concisely summarize the core idea, topic, or theme they collectively represent in a single, well-written paragraph. Focus on identifying the central subject discussed across these varied pieces of text.
"""

        # Call the Gemini API
        summary_text_content = f"Error generating summary." # Default error message
        retries = 2
        delay = 5
        for attempt in range(retries):
            try:
                print(f"Sending {len(combined_text)} characters to Gemini for summarization (attempt {attempt+1})...")
                # Add generation config if needed (e.g., temperature=0.5 for less randomness)
                # generation_config = genai.types.GenerationConfig(temperature=0.5)
                response = model.generate_content(prompt)#, generation_config=generation_config)

                # Accessing the text might depend on the library version and model response structure
                if response.parts:
                     generated_summary = "".join(part.text for part in response.parts)
                elif hasattr(response, 'text'):
                     generated_summary = response.text
                else:
                     # Try accessing _result if other attributes fail (less stable)
                     # generated_summary = response._result.candidates[0].content.parts[0].text
                     print("Warning: Could not extract text using standard attributes '.parts' or '.text'.")
                     generated_summary = "[Could not extract summary text]"


                if generated_summary.strip():
                    summary_text_content = generated_summary.strip()
                    print("Summary generated successfully.")
                else:
                    # Handle cases where the model might return an empty response (e.g., due to safety filters)
                     block_reason = response.prompt_feedback.block_reason if (hasattr(response, 'prompt_feedback') and response.prompt_feedback) else "Unknown"
                     # Check if candidates exist before accessing safety_ratings
                     safety_ratings = response.candidates[0].safety_ratings if (hasattr(response, 'candidates') and response.candidates) else "N/A"

                     print(f"Warning: Gemini returned an empty response or refused. Block Reason: {block_reason}. Safety Ratings: {safety_ratings}")
                     summary_text_content = f"Could not generate summary (Model response empty/blocked. Reason: {block_reason})."

                break # Success
            except Exception as e:
                print(f"Error calling Gemini API for cluster {i+1} (attempt {attempt+1}): {e}")
                if attempt < retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"Failed to generate summary for cluster {i+1} after {retries} attempts.")
                    summary_text_content = f"Failed to generate summary after multiple attempts."

        summaries.append({
            "cluster_id": i,
            "summary": summary_text_content,
            "centroid_embedding": centroids[i] # Store centroid embedding for ranking
        })
        time.sleep(1) # Add a small delay between API calls

    return summaries

def rank_ideas_by_book_similarity(key_idea_summaries, book_embedding):
    """
    Ranks key idea summaries based on the cosine similarity of their cluster
    centroid embedding to the whole book embedding.
    """
    if not key_idea_summaries or book_embedding is None:
        print("Cannot rank ideas: Missing summaries or book embedding.")
        return []

    ranked_ideas = []
    print("\nCalculating similarity of cluster centroids to whole book embedding...")

    # Reshape book_embedding to be a 2D array (1 sample, embedding dimension)
    book_embedding_2d = book_embedding.reshape(1, -1)

    for idea_info in key_idea_summaries:
        centroid_embedding = idea_info.get("centroid_embedding")
        summary = idea_info.get("summary", "No Summary Provided")

        if centroid_embedding is None:
            print(f"Warning: Missing centroid embedding for cluster ID {idea_info.get('cluster_id', 'N/A')}. Skipping ranking for this idea.")
            similarity = -1 # Assign a low similarity so it ranks last
        else:
            try:
                 # Ensure embeddings are numpy arrays before calculating similarity
                 centroid_embedding_np = np.array(centroid_embedding).reshape(1, -1)
                 # Check for NaN or Inf values
                 if np.isnan(centroid_embedding_np).any() or np.isinf(centroid_embedding_np).any() or np.isnan(book_embedding_2d).any() or np.isinf(book_embedding_2d).any():
                     print(f"Warning: NaN or Inf found in embeddings for cluster ID {idea_info.get('cluster_id', 'N/A')}. Cannot calculate similarity.")
                     similarity = -1
                 else:
                    # Calculate cosine similarity
                    # The function returns a 2D array [[similarity_score]], so take the first element
                    similarity = cosine_similarity(centroid_embedding_np, book_embedding_2d)[0][0]
            except Exception as e:
                 print(f"Error calculating similarity for cluster ID {idea_info.get('cluster_id', 'N/A')}: {e}")
                 similarity = -1 # Assign low similarity on error

        ranked_ideas.append({
            "cluster_id": idea_info.get("cluster_id", "N/A"),
            "summary": summary,
            "similarity": similarity
        })

    # Sort by similarity in descending order
    # Handle potential -1 similarities (from errors/missing embeddings) by putting them at the end
    ranked_ideas.sort(key=lambda x: x['similarity'] if x['similarity'] != -1 else -float('inf'), reverse=True)

    print("Ranking complete.")
    return ranked_ideas


# --- Main Execution ---
if __name__ == "__main__":
    book_path = input("Enter the path to the book file (PDF or TXT): ")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\nError: GOOGLE_API_KEY environment variable not set.")
        exit(1)

    # 1. Read Book
    print("-" * 20)
    book_text = read_book(book_path)

    if book_text:
        # 2. Get Embedding of the Whole Book
        print("-" * 20)
        print("Generating embedding for the entire book...")
        book_embedding = get_embeddings(book_text, EMBEDDING_MODEL_NAME, api_key)

        if book_embedding is None:
            print("\nFailed to generate embedding for the whole book. Cannot proceed with ranking.")
            # Decide if you want to exit or continue without ranking
            exit(1) # Exit if ranking is essential

        # 3. Chunk Text
        print("-" * 20)
        text_chunks = chunk_text_by_paragraph(book_text)

        if text_chunks:
            # 4. Get Embeddings for Chunks
            print("-" * 20)
            embeddings_array = get_embeddings(text_chunks, EMBEDDING_MODEL_NAME, api_key)

            if embeddings_array is not None and embeddings_array.shape[0] > 0:
                # Adjust NUM_CLUSTERS if needed
                actual_num_clusters = min(NUM_CLUSTERS, embeddings_array.shape[0])
                if actual_num_clusters < NUM_CLUSTERS:
                    print(f"\nWarning: Using {actual_num_clusters} clusters (was {NUM_CLUSTERS}) due to fewer chunks ({embeddings_array.shape[0]}).")
                if actual_num_clusters == 0:
                     print("\nError: No embeddings available. Cannot cluster.")
                     exit(1)

                # 5. Cluster Embeddings
                print("-" * 20)
                kmeans_result = cluster_embeddings(embeddings_array, actual_num_clusters)

                if kmeans_result:
                    # 6. Generate Summaries for Clusters (includes centroid embeddings)
                    print("-" * 20)
                    key_idea_summaries_with_embeddings = generate_summaries_for_clusters(
                        kmeans_result,
                        embeddings_array,
                        text_chunks,
                        api_key,
                        SUMMARIZATION_MODEL_NAME,
                        num_closest=NUM_CLOSEST_CHUNKS_TO_SUMMARIZE,
                        max_input_len=MAX_SUMMARY_INPUT_LENGTH
                    )

                    if key_idea_summaries_with_embeddings:
                        # 7. Rank Key Ideas
                        print("-" * 20)
                        ranked_ideas = rank_ideas_by_book_similarity(key_idea_summaries_with_embeddings, book_embedding)

                        # 8. Print Ranked Key Ideas
                        print("\n" + "=" * 40)
                        print(f" Ranked Key Ideas by Similarity to Whole Book ")
                        print("=" * 40)
                        if ranked_ideas:
                            for rank, idea_info in enumerate(ranked_ideas):
                                print(f"\n--- Rank {rank + 1} (Similarity: {idea_info['similarity']:.4f}) ---")
                                print(idea_info['summary'])
                                print("-" * 40)
                        else:
                            print("\nNo ranked ideas to display (ranking failed or no summaries generated).")
                    else:
                        print("\nNo summaries could be generated for the clusters. Cannot rank.")
                else:
                    print("\nClustering step failed.")
            else:
                print("\nEmbedding generation failed or yielded no results. Cannot proceed.")
        else:
            print("\nNo valid text chunks found. Cannot proceed.")
    else:
        print("\nFailed to read/extract text from file. Cannot proceed.")

    print("\nScript finished.")
