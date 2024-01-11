from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.evaluation import load_evaluator


def main():
    # Get embedding for a word.
    embedding_function = HuggingFaceEmbeddings()
    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {vector}")
    print(f"Vector length: {len(vector)}")

    # Compare vector of two words
    evaluator = load_evaluator("embedding_distance",embeddings = embedding_function)
    words = ("apple", "iphone")
    x = evaluator.evaluate_strings(prediction=words[0], reference=words[1])
    print(f"Comparing ({words[0]}, {words[1]}): {x}")


if __name__ == "__main__":
    main()
