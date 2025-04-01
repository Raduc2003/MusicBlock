import json
import essentia.standard as es
import numpy as np
import matplotlib.pyplot as plt
musicnn_metadata = json.load(open('msd-musicnn-1.json', 'r'))

# for k, v in musicnn_metadata.items():
#     print('{}: {}'.format(k , v))

sr = 16000
audio = es.MonoLoader(filename='test.mp3', sampleRate=sr)()

# musicnn_preds = es.TensorflowPredictMusiCNN(graphFilename='msd-musicnn-1.pb')(audio)a

# classes = musicnn_metadata['classes']

# plt.matshow(musicnn_preds.T)
# plt.title('taggram')
# plt.yticks(np.arange(len(classes)), classes)
# plt.gca().xaxis.set_ticks_position('bottom')
# plt.show()
danceability_preds = es.TensorflowPredictMusiCNN(graphFilename='danceability-musicnn-msd-2.pb')(audio)

danceability_metadata = json.load(open('danceability-musicnn-msd-2.json', 'r'))['classes']

# Average predictions over the time axis
danceability_preds = np.mean(danceability_preds, axis=0)

print('{}: {}%'.format(danceability_metadata[0] , danceability_preds[0] * 100))
