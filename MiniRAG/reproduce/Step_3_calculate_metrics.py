import pandas as pd

if __name__ == "__main__":
    minirag_eval = pd.read_csv("./logs/qa_output_minirag_evaluation_gpt-4o-mini.csv")
    naive_eval = pd.read_csv("./logs/qa_output_naive_evaluation_gpt-4o-mini.csv")

    print("MiniRAG Evaluation Metrics:")
    print(minirag_eval[['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']].mean())

    print("\nNaive RAG Evaluation Metrics:")
    print(naive_eval[['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']].mean())