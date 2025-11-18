# Dots Classifier (Client-side)

This is a minimal static demo of a Dots Classifier that generates synthetic images of white circular dots on a dark background, then trains a small CNN in-browser using TensorFlow.js.

Features included:
- Client-side training with Adam optimizer (learning rate 0.001)
- Generator options: overlap allow/prevent, dot count range (1–9), image size, dot radius range, seed
- Training controls: batch size, epochs (1–8), steps/epoch, val steps/epoch
- Preview of the first image in the latest batch
- Live chart plotting training vs validation accuracy per batch (Chart.js)

How to run:
Open `index.html` in a modern browser (Chrome/Edge/Firefox). The page uses CDNs for TensorFlow.js and Chart.js — an internet connection is required.

Notes and caveats:
- This is a starting scaffold — performance depends on browser and device.
- Model saving/loading and advanced UX are not implemented yet.
# quiz2.1