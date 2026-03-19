## Dependency Issue Fixes
- Must install in this order to avoid protobuf depednecy conflict between onnx and descript-audio-codec
- Also should include `--prefer-binary` flag to avoid cmake incompatibilities wherever possible

```sh
pip install onnx onnxruntime onnxscript --prefer-binary
pip install torch torchaudio --prefer-binary
pip install descript-audio-codec
pip install "protobuf>=4.25.1" --force-reinstall
```

- onnx and descript-audio-codec require conflicting versions of protobuf. That last pip install step will be (allegedly) incompatible for descript-audio-codec. After running that install step you'll see console warnings/errors but actually everything will be fine. I went around in circles trying to reconcile the dependency collisions and this was the only way I could get things to play nicely.

```sh
pip install torchcodec
```

- `torchcodec` required to run `test_models.py`