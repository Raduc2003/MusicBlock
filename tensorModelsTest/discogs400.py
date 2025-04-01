import json
import essentia.standard as es
import numpy as np
import matplotlib.pyplot as plt

audio = es.MonoLoader(filename="manelis.mp3", sampleRate=16000, resampleQuality=4)()

embedding_model = es.TensorflowPredictEffnetDiscogs(graphFilename="discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
embeddings = embedding_model(audio)

model = es.TensorflowPredict2D(graphFilename="genre_discogs400-discogs-effnet-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall:0")
predictions = model(embeddings)

# Get top 10 predicted genres
N = 10

# Average predictions over time frames (if predictions are per-frame)
if len(predictions.shape) > 1 and predictions.shape[0] > 1:
    predictions_mean = np.mean(predictions, axis=0)
else:
    predictions_mean = predictions[0]  # Just use the first frame if only one exists

# Get top N indices
top_indices = (-predictions_mean).argsort()[:N]
top_values = predictions_mean[top_indices]

# Load metadata
metadata = json.load(open("genre_discogs400-discogs-effnet-1.json", "r"))
# Get labels for top genres
top_labels = [metadata["classes"][i] for i in top_indices]

# Create bar chart
plt.figure(figsize=(12, 6))
plt.bar(range(N), top_values, color='skyblue')
plt.xticks(range(N), top_labels, rotation=45, ha='right')
plt.xlabel('Genres')
plt.ylabel('Probability')
plt.title('Top 10 Predicted Music Genres')
plt.tight_layout()

# Display the plot
plt.show()

# Print numerical results
print("\nTop predicted genres:")
for i in range(N):
    print(f"{top_labels[i]}: {top_values[i]:.4f}")