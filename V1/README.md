# RonoLM

Local-first conversational AI with SQLite memories, local sentence embeddings,
and Step 9 cross-thread Topic Unit/Topic Episode recall.

## Run

```bash
cd V1
python3 -m pip install -e .
ollama serve
ollama pull llama3.2:3b
python3 -m ronolm.cli
```

The idempotent SQLite migration runs on CLI startup. It preserves the Step 8
`memory_embeddings` table and adds the Step 9 topic tables.

Both text and voice turns use `ronolm.chat_service.process_user_message`, so
voice transcripts pass through the same message storage, memory, topic recall,
Ollama, and episode-update pipeline.

## Local voice

Install the optional local audio dependencies:

```bash
sudo apt install libportaudio2
python3 -m pip install -e '.[voice]'
```

Export the values in `.env.example`, setting `PIPER_MODEL_PATH` to an installed
Piper `.onnx` voice model. If Piper is outside `PATH`, also set
`PIPER_EXECUTABLE` to its absolute path. RonoLM reads these values from the
process environment; it does not load `.env` files automatically.

```bash
set -a
source .env
set +a
python3 -m ronolm.voice_cli
```

Push Enter once to start a turn and again to stop recording. The conversation
view stays minimal: input prompt, live listening meter, transcription status,
heard text, and the RonoLM response. Voice commands are `/text`, `/mute`,
`/unmute`, `/repeat`, `/mic`, `/debug`, `/quit`, and `/exit`.

Run without speech playback:

```bash
python3 -m ronolm.voice_cli --no-tts
```

Test the adapters independently:

```bash
python3 -m ronolm.audio.stt path/to/audio.wav
python3 -m ronolm.audio.tts "Hello, this is RonoLM."
python3 -m ronolm.audio.tts "Hello, this is RonoLM." --output tmp/audio/test.wav
```

Input and generated speech files use `TTS_OUTPUT_DIR` and are deleted after
use when `DELETE_TEMP_AUDIO=true`. RonoLM stores only the transcript text.

Useful inspection commands:

```text
/topics
/episodes [topic_id]
/topic <topic_id>
/episode <episode_id>
/thread-topics [thread_id]
/route <text>
/thread-retrieve <text>
```

## Test

```bash
cd V1
python3 -m unittest discover -s tests -v
```
