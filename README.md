# README

## Installation

Follow these steps to set up the project and install dependencies:

1. **Install Python**
   Ensure you have Python `>=3.11, <3.13` installed. Download it here: [Python Downloads](https://www.python.org/downloads/).

2. **Clone the repository**

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

3. **Install dependencies**
   You can use either `pip` or `poetry`:

   * **Using pip**:

     ```bash
     pip install -r requirements.txt
     ```
   * **Using poetry**:

     ```bash
     poetry install
     ```

4. **Verify installation**
   Run the main script to check that everything is set up correctly:

   ```bash
   python .\src\krzycz_trybson\pipeline.py -h
   ```

   or if installed as a package:

   ```bash
   python -m krzycz_trybson.pipeline -h
   ```

   This should display the script's help message.

---

## Usage

Run the pipeline with the desired input:

### 1. Local Video File

```bash
python .\src\krzycz_trybson\pipeline.py --input video.mp4 --output ./results
```

or as a package:

```bash
python -m krzycz_trybson.pipeline --input video.mp4 --output ./results
```

### 2. YouTube Video

```bash
python .\src\krzycz_trybson\pipeline.py --youtube-url "https://www.youtube.com/watch?v=..." --output ./results
```

or as a package:

```bash
python -m krzycz_trybson.pipeline --youtube-url "https://www.youtube.com/watch?v=..." --output ./results
```

Replace `video.mp4` with your file path and `./results` with your desired output directory.

---

## Optional Flags

### Disable Autocorrection

By default, the pipeline performs transcription autocorrection using a large language model. This can sometimes slow down processing or cause delays if the Ollama server is overloaded. To skip this step, use the `--no-autocorrect` flag:

```bash
python .\src\krzycz_trybson\pipeline.py --input video.mp4 --output ./results --no-autocorrect
```

or as a package:

```bash
python -m krzycz_trybson.pipeline --input video.mp4 --output ./results --no-autocorrect
```

---

## Limitations and Disclaimer

* **YouTube Downloads**: The YouTube functionality relies on the `yt-dlp` library. Some videos may fail to download due to age restrictions, region locks, or changes in YouTube’s policies. Updating `yt-dlp` usually resolves these issues:

  ```bash
  pip install --upgrade yt-dlp
  ```

  or with poetry:

  ```bash
  poetry update yt-dlp
  ```

* **Disclaimer**: This project is provided "as is" without warranties. The authors are not responsible for any issues such as data loss or system crashes. Use at your own risk.

---

## Model Card

Model card can be found [here](https://github.com/BredaUniversityADSAI/fae2-nlpr-group-group-9/blob/74772a70a22b6aeba6f86ef4424f9be86eadfe46/task11/readme.md).

## Requirements List
Full list of dependencies can be found [here](./requirements.txt).
