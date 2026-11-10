Submission package. Zip the *contents* of this directory (agent.yaml at the zip root):

    cd submission && zip -r ../submission.zip . -x README.md

Local eval (server from scripts/serve_vllm.sh must be up):

    OPENAI_BASE_URL=http://127.0.0.1:8000/v1 swegemma eval --tasks data/gemma4/tasks.jsonl \
      --snapshots-dir data/gemma4/snapshots --submission-dir submission --results-dir runs/eval/<name> \
      --sandbox docker --max-tool-calls 40 --max-time-minutes 5 --task-id httpx_3672
