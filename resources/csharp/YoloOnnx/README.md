# C# corner (Week 9B): YOLO ONNX inference

A minimal .NET 8 console program that runs the conveyor detector trained and exported in
`notebooks/W09_data_centric/W09B_train_deploy_detector.ipynb` (`best.onnx`), using
[ONNX Runtime](https://onnxruntime.ai/docs/get-started/with-csharp.html) and [OpenCvSharp](https://github.com/shimat/opencvsharp).

It performs the same steps as the Python notebook: letterbox, tensor conversion, inference, decoding, undoing the
letterbox and class-aware NMS.

```bash
cd resources/csharp/YoloOnnx
dotnet build
dotnet run -- ../../../data/w09b/best.onnx ../../../data/w09b/scene.jpg 0.25
```

- **Windows**: works out of the box (`OpenCvSharp4.runtime.win`).
- **Linux**: add the native runtime for your distribution, e.g. `dotnet add package OpenCvSharp4_.runtime.ubuntu.20.04-x64`.
- **GPU**: replace `Microsoft.ML.OnnxRuntime` by `Microsoft.ML.OnnxRuntime.Gpu` and create the session with
  `SessionOptions.MakeSessionOptionWithCudaProvider()`.

Continuous integration only checks that this project **builds**; run it locally.
