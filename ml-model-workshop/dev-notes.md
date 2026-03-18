## Dependency Issue Fixes
- Must install in this order to avoid protobuf depednecy conflict between onnx and descript-audio-codec
- Also should include `--prefer-binary` flag to avoid cmake incompatibilities

```sh
pip install onnx onnxruntime onnxscript --prefer-binary
pip install torch torchaudio --prefer-binary
pip install descript-audio-codec
pip install "protobuf>=4.25.1" --force-reinstall
```

- Because onnx was installed in a previous step, pip will leave it installed. It will downgrade Protobuf, yell at you with red text, and finish. ONNX will survive on the older Protobuf, DAC will have the exact version it needs, and CMake won't execute a single time.

- UPDATE: actually needed to upgrade to `protobuf>=4.25.1` with that last pip install command...

```sh
pip install torchcodec
```

- `torchcodec` required to run `test_models.py`