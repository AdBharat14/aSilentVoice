import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:picovoice_flutter/picovoice_error.dart';
import 'package:picovoice_flutter/picovoice_manager.dart';
import 'package:rhino_flutter/rhino.dart';


class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final String accessKey =
      "vTOSlvjB0JX/YFEvd7s4idEVeXmOxWKdZGUowafSF2ru4k6zUxQd5A==";
  final String keywordAsset = "./assets/zoro.ppn";
  final String contextAsset = "./assets/mangaReader.rhn";

  final FlutterTts flutterTts = FlutterTts();


  bool _isListening = false;
  PicovoiceManager? _picovoiceManager;
  
  @override
  void initState() {
    super.initState();
    _initPicovoice();
  }

  void _initPicovoice() async {
    try {
      _picovoiceManager = await PicovoiceManager.create(accessKey, keywordAsset,
          _wakeWordCallback, contextAsset, _inferenceCallback,
          processErrorCallback: _errorCallback);
      await _picovoiceManager?.start();
    } on PicovoiceActivationException {
      _errorCallback(
          PicovoiceActivationException("AccessKey activation error."));
    } on PicovoiceActivationLimitException {
      _errorCallback(PicovoiceActivationLimitException(
          "AccessKey reached its device limit."));
    } on PicovoiceActivationRefusedException {
      _errorCallback(PicovoiceActivationRefusedException("AccessKey refused."));
    } on PicovoiceActivationThrottledException {
      _errorCallback(PicovoiceActivationThrottledException(
          "AccessKey has been throttled."));
    } on PicovoiceException catch (ex) {
      _errorCallback(ex);
    }
  }

  _wakeWordCallback() {
    setState(() {
      _isListening = true;
    });
    // _speak("Hello,I am zoro.");
  }

  _inferenceCallback(RhinoInference inference) {
    if(inference.isUnderstood!){
      Map<String , String>? slots = inference.slots!;
      if(inference.intent == "open_manga"){
        String? manga = slots["manga"];
        _speak("Opening ${manga!} manga");

      }
    }else{
      _speak("I did not understand your command. Please repeat.");
    }
    setState(() {
      _isListening = false;
    });
  }

  _errorCallback(PicovoiceException error) {
    print(error.message);
  }

  _speak(String content) async{
    await flutterTts.setLanguage('en-US');
    await flutterTts.speak(content);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
                child: _isListening
                    ? Icon(
                        Icons.mic,
                        size: 100,
                        color: Theme.of(context).focusColor,
                      )
                    : Icon(
                        Icons.mic_none,
                        size: 100,
                        color: Theme.of(context).primaryColor,
                      )),
          ],
        ),
      ),
    );
  }
}