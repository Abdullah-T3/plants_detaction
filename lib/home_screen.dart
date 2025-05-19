import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'Responsive/enums/device_type.dart';
import 'Responsive/ui_component/info_widget.dart';

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
  'Aloe vera', // Class 0
  'Banana',  // Class 1
  'Mint',  // Class 2
  'lettuce'    // Class 3
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
        input[0][y][x][0] = img.getRed(pixel).toDouble();
        input[0][y][x][1] = img.getGreen(pixel).toDouble();
        input[0][y][x][2] = img.getBlue(pixel).toDouble();
      }
    }
 
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
    
    int predictedIndex = 0;
    double maxProb = output[0][0];
    
    List<MapEntry<int, double>> classProbabilities = [];
    
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
    String formattedProb;
    String resultText;
    // Format the probability as a percentage for better readability
    if(topProbability * 100 < 60) {
      resultText = "Unrecognized photo";
    } else {
      formattedProb = (topProbability * 100).toStringAsFixed(2) + "%";
      // Get the plant name for the top class
      String plantName = topClassIndex < plantNames.length ? plantNames[topClassIndex] : "Unknown Plant";
      
      // Create a more detailed result text showing max probability class
      resultText = "Detected: $plantName\nConfidence: $formattedProb\n\nMax Probability Class: $topClassIndex (${plantNames[topClassIndex]})";
      
      // Debug information
      print('Max Probability Class: $topClassIndex (${plantNames[topClassIndex]}) with confidence: $formattedProb');
    }

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
        title: const Text('Plantify' , style: TextStyle(color: Colors.white),),
        backgroundColor: Colors.green,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: InfoWidget(
          builder: (context, deviceInfo) {
            // Adjust UI based on device type
            final isTabletOrDesktop = deviceInfo.deviceType == DeviceType.tablet ||
                deviceInfo.deviceType == DeviceType.desktop;
            final imageHeight = isTabletOrDesktop ? 350.0 : 250.0;
            final fontSize = isTabletOrDesktop ? 22.0 : 18.0;
            final iconSize = isTabletOrDesktop ? 120.0 : 100.0;
            final buttonSpacing = isTabletOrDesktop ? 30.0 : 20.0;

            return SingleChildScrollView(

              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  _image == null
                      ? Container(
                    height: imageHeight,
                    width: double.infinity,
                    color: Colors.grey[300],
                    child: Icon(
                      Icons.image,
                      size: iconSize,
                      color: Colors.grey,
                    ),
                  )
                      : Container(
                    height: imageHeight,
                    width: double.infinity,
                    decoration: BoxDecoration(
                      image: DecorationImage(
                        image: FileImage(_image!),
                        fit: BoxFit.contain,
                      ),
                    ),
                  ),
                  SizedBox(height: buttonSpacing),
                  isLoading
                      ? const CircularProgressIndicator()
                      : Text(
                    result,
                    style: TextStyle(
                      fontSize: fontSize,
                      fontWeight: FontWeight.bold,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  SizedBox(height: buttonSpacing),
                  isTabletOrDesktop && deviceInfo.orientation == Orientation.landscape
                      ? Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      _buildButton(Icons.camera_alt, 'Camera', () => getImage(ImageSource.camera)),
                      SizedBox(width: buttonSpacing),
                      _buildButton(Icons.photo_library, 'Gallery', () => getImage(ImageSource.gallery)),
                    ],
                  )
                      : Column(
                    children: [
                      _buildButton(Icons.camera_alt, 'Camera', () => getImage(ImageSource.camera)),
                      SizedBox(height: buttonSpacing / 2),
                      _buildButton(Icons.photo_library, 'Gallery', () => getImage(ImageSource.gallery)),
                    ],
                  ),
                ],
              ),
            );
          },
        ),
      ),
    );
  }

  // Helper method to build consistent buttons
  Widget _buildButton(IconData icon, String label, VoidCallback onPressed) {
    return ElevatedButton.icon(
      onPressed: onPressed,
      icon: Icon(icon),
      label: Text(label),
      style: ElevatedButton.styleFrom(
        backgroundColor: Colors.green,
        foregroundColor: Colors.white,
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
      ),
    );
  }

  @override
  void dispose() {
    interpreter?.close();
    super.dispose();
  }
}
