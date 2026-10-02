from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct',revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306',local_dir='models/qwen',ignore_patterns=['*.msgpack','*.h5','*.ot'])
