# Neural Network Digit Recognition

A handwritten digit recognizer built from scratch with **Python and NumPy** (no TensorFlow or PyTorch). Draw a digit on a **10 × 10** grid and see the model's live prediction in your browser.

## Architecture

**100 → 64 → 32 → 10** — 100 input pixels, two hidden layers (64 and 32 neurons), and 10 digit outputs (0–9).

![Neural network architecture](image.png)

## Run the website with Docker Compose

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/), then run from the repository root:

```bash
docker compose up -d --build
```

Open **http://localhost:5000**. To stop the application:

```bash
docker compose down
```

The repository needs `models/model.npz` for predictions to work.

## Run with a VS Code Dev Container

1. Install Docker Desktop and the **Dev Containers** extension in VS Code.
2. Open the repository in VS Code. Press **Ctrl+Shift+P** → **Dev Containers: Reopen in Container**. Wait for setup to install `requirements.txt`.
3. Open **Terminal → New Terminal** inside the container and run:

   ```bash
   python server.py
   ```

4. In VS Code's **Ports** tab, forward **5000** if needed. Open **http://localhost:5000**.
5. Stop Flask with **Ctrl+C**.

**Reconnect to an existing container:** In VS Code, use **Dev Containers: Attach to Running Container...** and select `neural-network-dev`. Then open the project folder and run `python server.py`. Attaching does not rerun `postCreateCommand`.

**Without VS Code**, you can open a shell in the existing container (if it is running):

```bash
docker exec -it -u vscode -w /workspaces/NeuralNetwork-Python neural-network-dev bash
python server.py
```

This `docker exec` approach does **not** publish port 5000 to Windows. For browser access without VS Code forwarding, use Docker Compose above.

## Run locally without Docker

With Python 3.12 installed:

```bash
python -m pip install -r requirements.txt
python server.py
```

Open **http://localhost:5000**.

## Train or guess from the terminal

```bash
python train.py   # Train and save the model
python guess.py   # Paste 100 pixel values to predict a digit
```

The trained model is saved as `models/model.npz`. The browser drawing tool is `pixel_drawer.html`; it supports drawing, erasing, clearing, copying pixel values, and saving PNGs. The Flask server (`server.py`) serves the page and runs predictions.

**Note:** Output percentages are sigmoid scores, not calibrated probabilities. Accuracy depends on the training data and drawing style.
