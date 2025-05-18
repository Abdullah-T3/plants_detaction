import 'dart:io';
import 'dart:typed_data';
import 'dart:math' as Math;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  File? _image;
  final picker = ImagePicker();
  String result = "";
  bool isLoading = false;
  Interpreter? interpreter;
  
  // Map class indices to plant names
  final List<String> plantNames = [
    'Lettuce', // Class 0
    'Cactus',  // Class 1
    'Banana',  // Class 2
    'Mint'     // Class 3
  ];

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  Future<void> loadModel() async {
    try {
      interpreter = await Interpreter.fromAsset('assets/model.tflite');
      print('Model loaded successfully');

      // Fix: Get both input and output shapes correctly
      print('Input shape: ${interpreter!.getInputTensor(0).shape}');
      print('Output shape: ${interpreter!.getOutputTensor(0).shape}');
      
      // Print model details
      print('Input tensor shape: ${interpreter!.getInputTensor(0).shape}');
      print('Output tensor shape: ${interpreter!.getOutputTensor(0).shape}');
      
      // Verify the model has the correct number of output classes
      final outputShape = interpreter!.getOutputTensor(0).shape;
      if (outputShape[1] != plantNames.length) {
        print('WARNING: Model output classes (${outputShape[1]}) does not match plantNames length (${plantNames.length})');
      }
    } catch (e) {
      print('Error loading model: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error: TensorFlow Lite cannot load the model.'),
          backgroundColor: Colors.red,
          duration: const Duration(seconds: 10),
        ),
      );
    }
  }

  Future<void> getImage(ImageSource source) async {
    final pickedFile = await picker.pickImage(source: source);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
      });
      classifyImage(_image!);
    } else {
      print('No image selected.');
    }
  }

Future<void> classifyImage(File image) async {
  setState(() {
    isLoading = true;
  });

  try {
    final imageBytes = await image.readAsBytes();
    final decodedImage = img.decodeImage(imageBytes);

    if (decodedImage == null) {
      throw Exception('Failed to decode image');
    }

    final resizedImage = img.copyResize(decodedImage, width: 299, height: 299);

    // Create a Float32List for input tensor
    var inputShape = interpreter!.getInputTensor(0).shape;
    print('Input shape: $inputShape');
    
    // Create a properly shaped input tensor (4D: [1, height, width, channels])
    var input = List.generate(
      1,
      (_) => List.generate(
        299,
        (_) => List.generate(
          299,
          (_) => List.filled(3, 0.0),
        ),
      ),
    );
    
  
    for (var y = 0; y < 299; y++) {
      for (var x = 0; x < 299; x++) {
        final pixel = resizedImage.getPixelSafe(x, y);
        input[0][y][x][0] = img.getRed(pixel) / 255.0;
        input[0][y][x][1] = img.getGreen(pixel) / 255.0;
        input[0][y][x][2] = img.getBlue(pixel) / 255.0;
      }
    }
    
    // Print sample of preprocessed pixels for debugging
    print('Sample of preprocessed pixels:');
    print('Pixel at (0,0): [${input[0][0][0][0]}, ${input[0][0][0][1]}, ${input[0][0][0][2]}]');
    print('Pixel at (150,150): [${input[0][150][150][0]}, ${input[0][150][150][1]}, ${input[0][150][150][2]}]');
    
    // Get output shape and prepare output tensor
    var outputShape = interpreter!.getOutputTensor(0).shape;
    print('Output shape: $outputShape');
    
    // Create output tensor based on actual output shape
    var output = List.filled(1, List.filled(plantNames.length, 0.0));
    
    // Run inference
    interpreter!.run(input, output);
    
    // Debug the raw output
    print('Raw output tensor: $output');
    
    print('Running inference...');
    print('Model output: $output');
    
    // Find the class with highest probability
    int predictedIndex = 0;
    double maxProb = output[0][0];
    
    // Create a list of (class index, probability) pairs
    List<MapEntry<int, double>> classProbabilities = [];
    
    // Debug output to see raw values
    print('Raw output values:');
    for (int i = 0; i < output[0].length; i++) {
      double prob = output[0][i];
      print('Class $i: $prob');
      classProbabilities.add(MapEntry(i, prob));
      
      // Update the max probability if current is higher
      if (prob > maxProb) {
        maxProb = prob;
        predictedIndex = i;
      }
    }
    
    // Debug raw output values
    print('Raw output values:');
    for (int i = 0; i < output[0].length; i++) {
      print('Class $i (${plantNames[i]}): ${output[0][i]}');
    }
    
    // Use raw output values directly
    classProbabilities.clear();
    for (int i = 0; i < output[0].length; i++) {
      // Take absolute value to handle negative outputs
      double value = output[0][i].abs();
      classProbabilities.add(MapEntry(i, value));
      print('Using absolute value - Class $i (${plantNames[i]}): $value');
    }
    
    // Sort by probability in descending order
    classProbabilities.sort((a, b) => b.value.compareTo(a.value));
    
    // Only display the top class prediction
    int topClassIndex = classProbabilities[0].key;
    double topProbability = classProbabilities[0].value;
    
    // Debug the top prediction
    print('Top predicted class: $topClassIndex (${plantNames[topClassIndex]}) with probability: $topProbability');
    
    // Format the probability as a percentage for better readability
    String formattedProb = (topProbability * 100).toStringAsFixed(2) + "%";
    
    // Get the plant name for the top class
    String plantName = topClassIndex < plantNames.length ? plantNames[topClassIndex] : "Unknown Plant";
    
    // Build the result string with the plant name and probability
    String resultText = "Detected: $plantName\nConfidence: $formattedProb";
    
    setState(() {
      result = resultText;
      isLoading = false;
    });
  } catch (e) {
    print('Error classifying image: $e');
    setState(() {
      result = "Error: $e";
      isLoading = false;
    });
  }
}
 
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Plant Identification'),
        backgroundColor: Colors.green,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _image == null
                ? Container(
                    height: 250,
                    width: double.infinity,
                    color: Colors.grey[300],
                    child: const Icon(
                      Icons.image,
                      size: 100,
                      color: Colors.grey,
                    ),
                  )
                : Container(
                    height: 250,
                    width: double.infinity,
                    decoration: BoxDecoration(
                      image: DecorationImage(
                        image: FileImage(_image!),
                        fit: BoxFit.contain,
                      ),
                    ),
                  ),
            const SizedBox(height: 20),
            isLoading
                ? const CircularProgressIndicator()
                : Text(
                    result,
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                    textAlign: TextAlign.center,
                  ),
            const SizedBox(height: 20),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed: () => getImage(ImageSource.camera),
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('Camera'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.green,
                    foregroundColor: Colors.white,
                  ),
                ),
                ElevatedButton.icon(
                  onPressed: () => getImage(ImageSource.gallery),
                  icon: const Icon(Icons.photo_library),
                  label: const Text('Gallery'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.green,
                    foregroundColor: Colors.white,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    interpreter?.close();
    super.dispose();
  }
}
