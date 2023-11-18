import torch
import time
import numpy as np
import matplotlib.pyplot as plt

# Function to perform similarity search
def similarity_search(query_vector, embedding_set, k=5, metric_type="cosine"):

    if metric_type == "cosine":
        distances = torch.nn.functional.cosine_similarity(embedding_set, query_vector)
    elif metric_type == "euclidean":
        distances = torch.cdist(embedding_set, query_vector.unsqueeze(0))
    else:
        raise ValueError("Unsupported metric type")

    distances = distances.detach().cpu().numpy()
    indices = np.argsort(distances)[:k]

    return distances[indices], indices

# Function to measure search time
def measure_search_time(embedding_set, metric_type, use_gpu=True):
    query_vector = embedding_set[0, :]  # Use the first vector as a query

    if use_gpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    embedding_set = torch.tensor(embedding_set).to(device)
    query_vector = torch.tensor(query_vector).to(device)

    start_time = time.time()
    _, _ = similarity_search(query_vector, embedding_set, k=5, metric_type=metric_type)
    elapsed_time = time.time() - start_time
    device_name = "GPU" if use_gpu and torch.cuda.is_available() else "CPU"

    return elapsed_time, device_name

def plot_metrics(metrics):
    # Create a plot
    fig, ax = plt.subplots()

    # Define line styles and markers
    line_styles = ['-', '--', '-.']
    markers = ['o', 's', '^']

    for i, metric_type in enumerate(["cosine", "euclidean"]):
        for j, device_name in enumerate(["CPU", "GPU"]):
            for embedding_dim in [128, 256, 512]:
                mask = (
                    (np.array(metrics["metric_type"]) == metric_type)
                    & (np.array(metrics["device_name"]) == device_name)
                    & (np.array(metrics["embedding_dim"]) == embedding_dim)
                )
                label = f"{metric_type} - {device_name} - Embedding Dim {embedding_dim}"

                x_values = np.array(metrics["num_embeddings"])[mask]
                y_values = np.array(metrics["search_time"])[mask]

                # Use different line styles and markers for differentiation
                style = line_styles[i % len(line_styles)]
                marker = markers[j % len(markers)]

                ax.plot(x_values, y_values, label=label, linestyle=style, marker=marker)

    ax.set_xlabel("Number of Embeddings")
    ax.set_ylabel("Search Time (seconds)")
    ax.set_xticks(np.unique(metrics["num_embeddings"]))  # Set explicit x-axis ticks
    ax.legend(prop={'size': 8}, borderaxespad=0.1)
    plt.title("Search Time vs Number of Embeddings")
    plt.savefig("search_complexity_plot.png")
    plt.show()

def plot_space_metrics(space_metrics):
    # Create a plot for space complexity
    fig, ax = plt.subplots()

    for embedding_dim in [128, 256, 512]:
        mask = np.array(space_metrics["embedding_dim"]) == embedding_dim
        label = f"Embedding Dim {embedding_dim}"
        ax.plot(np.array(space_metrics["num_embeddings"])[mask], np.array(space_metrics["space"])[mask], label=f"{label}")

    ax.set_xlabel("Number of Embeddings")
    ax.set_ylabel("Space (MB)")
    ax.set_xticks(np.unique(space_metrics["num_embeddings"]))  # Set explicit x-axis ticks
    ax.legend()
    plt.title("Space Complexity vs Number of Embeddings")
    plt.savefig("space_complexity_plot.png")
    plt.show()

# Main function
def main():
    run_space_test = True
    run_search_test = True

    metrics = {"num_embeddings": [], "embedding_dim": [], "metric_type": [], "device_name": [], "search_time": []}
    space_metrics = {"num_embeddings": [], "embedding_dim": [], "space": []}

    if run_space_test:
        print("Running Space Complexity Tests")
        for num_embeddings in [1000, 100000, 1000000]:
            for embedding_dim in [128, 256, 512]:
                print(f"Number of Embeddings: {num_embeddings}, Embedding Dimension: {embedding_dim}")

                # Generate random embeddings for testing
                embedding_set = np.random.rand(num_embeddings, embedding_dim).astype(np.float32)
                space = embedding_set.nbytes / (1024 ** 2) # Convert to MB

                space_metrics["num_embeddings"].append(num_embeddings)
                space_metrics["embedding_dim"].append(embedding_dim)
                space_metrics["space"].append(space)

                print(f"Space: {space:.4f} MB")

        plot_space_metrics(space_metrics)

    if run_search_test:
        print("Running Search Tests")
        for num_embeddings in [1000, 100000, 1000000]:
            for embedding_dim in [10, 128, 256, 512]: # Note: We add a dummy test of 10 to handle any loading problems of the functions into memory
                print(f"Number of Embeddings: {num_embeddings}, Embedding Dimension: {embedding_dim}")

                # Generate random embeddings for testing
                embedding_set = np.random.rand(num_embeddings, embedding_dim).astype(np.float32)

                for metric_type in ["cosine", "euclidean"]:
                    for use_gpu in [True, False]:
                        torch.cuda.empty_cache() # Empty the cache before each run

                        search_time, device_name = measure_search_time(embedding_set, metric_type, use_gpu)

                        metrics["num_embeddings"].append(num_embeddings)
                        metrics["embedding_dim"].append(embedding_dim)
                        metrics["metric_type"].append(metric_type)
                        metrics["device_name"].append(device_name)
                        metrics["search_time"].append(search_time)

                        print(f"Metric: {metric_type}, Device: {device_name}, Search Time: {search_time:.4f} seconds")

        plot_metrics(metrics)
    
if __name__ == "__main__":
    main()