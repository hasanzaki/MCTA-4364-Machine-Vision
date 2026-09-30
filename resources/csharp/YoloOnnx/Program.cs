// Week 9B - C# corner: YOLO (Ultralytics export) inference with ONNX Runtime and OpenCvSharp.
// The steps are exactly those of the Python notebook: letterbox -> NCHW float tensor -> run -> decode -> undo letterbox -> NMS.
using System.Diagnostics;
using System.Globalization;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;

string modelPath = args.Length > 0 ? args[0] : "best.onnx";
string imagePath = args.Length > 1 ? args[1] : "scene.jpg";
float confThreshold = args.Length > 2 ? float.Parse(args[2], CultureInfo.InvariantCulture) : 0.25f;
const float NmsIou = 0.7f;
string[] names = { "hex_nut", "washer", "bolt", "flange" };

using var session = new InferenceSession(modelPath);
var input = session.InputMetadata.First();
int size = input.Value.Dimensions[2];                       // square input, e.g. 480 or 640

using Mat image = Cv2.ImRead(imagePath);
if (image.Empty())
    throw new FileNotFoundException($"Cannot read {imagePath}");

// 1. Letterbox: keep the aspect ratio, pad with grey (114) to size x size.
float scale = (float)size / Math.Max(image.Width, image.Height);
int newW = (int)Math.Round(image.Width * scale), newH = (int)Math.Round(image.Height * scale);
int padX = (size - newW) / 2, padY = (size - newH) / 2;
using var resized = new Mat();
Cv2.Resize(image, resized, new Size(newW, newH));
using var canvas = new Mat(size, size, MatType.CV_8UC3, new Scalar(114, 114, 114));
using (var roi = new Mat(canvas, new Rect(padX, padY, newW, newH)))
    resized.CopyTo(roi);

// 2. BGR uint8 HWC -> RGB float NCHW in [0, 1].
var tensor = new DenseTensor<float>(new[] { 1, 3, size, size });
var pixels = canvas.GetGenericIndexer<Vec3b>();
for (int y = 0; y < size; y++)
{
    for (int x = 0; x < size; x++)
    {
        Vec3b p = pixels[y, x];
        tensor[0, 0, y, x] = p.Item2 / 255f;                // R
        tensor[0, 1, y, x] = p.Item1 / 255f;                // G
        tensor[0, 2, y, x] = p.Item0 / 255f;                // B
    }
}

// 3. Inference. Output shape: [1, 4 + classes, candidates], boxes as (cx, cy, w, h) in letterbox pixels.
var timer = Stopwatch.StartNew();
using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor(input.Key, tensor) });
timer.Stop();
Tensor<float> output = results.First().AsTensor<float>();
int numClasses = output.Dimensions[1] - 4;
int numCandidates = output.Dimensions[2];

// 4. Decode: best class per candidate, confidence threshold, undo the letterbox.
var boxes = new List<Rect>();
var scores = new List<float>();
var classIds = new List<int>();
for (int i = 0; i < numCandidates; i++)
{
    int bestClass = 0;
    float bestScore = 0f;
    for (int c = 0; c < numClasses; c++)
    {
        float s = output[0, 4 + c, i];
        if (s > bestScore) { bestScore = s; bestClass = c; }
    }
    if (bestScore < confThreshold) continue;
    float cx = output[0, 0, i], cy = output[0, 1, i], w = output[0, 2, i], h = output[0, 3, i];
    float x0 = (cx - w / 2 - padX) / scale, y0 = (cy - h / 2 - padY) / scale;
    boxes.Add(new Rect((int)Math.Round(x0), (int)Math.Round(y0), (int)Math.Round(w / scale), (int)Math.Round(h / scale)));
    scores.Add(bestScore);
    classIds.Add(bestClass);
}

// 5. Class-aware NMS: shift boxes of different classes apart so they never suppress each other.
var shifted = boxes.Select((b, i) => new Rect(b.X + classIds[i] * 10000, b.Y, b.Width, b.Height)).ToList();
CvDnn.NMSBoxes(shifted, scores, confThreshold, NmsIou, out int[] keep);

Console.WriteLine($"{keep.Length} objects in {timer.Elapsed.TotalMilliseconds:F1} ms (model input {size} x {size})");
foreach (int k in keep)
{
    string label = classIds[k] < names.Length ? names[classIds[k]] : classIds[k].ToString();
    Rect b = boxes[k];
    Console.WriteLine($"  {label,-8} {scores[k]:F2}  x={b.X} y={b.Y} w={b.Width} h={b.Height}");
    Cv2.Rectangle(image, b, Scalar.LimeGreen, 2);
    Cv2.PutText(image, $"{label} {scores[k]:F2}", new Point(b.X, Math.Max(b.Y - 4, 12)), HersheyFonts.HersheySimplex, 0.5, Scalar.LimeGreen, 1);
}
Cv2.ImWrite("detections.jpg", image);
Console.WriteLine("Saved detections.jpg");
