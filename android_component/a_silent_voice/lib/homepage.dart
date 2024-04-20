import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:picovoice_flutter/picovoice_error.dart';
import 'package:picovoice_flutter/picovoice_manager.dart';
import 'package:rhino_flutter/rhino.dart';
import 'package:cloud_firestore/cloud_firestore.dart';




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
  bool isSpeaking = false;
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

  Future<void> _findAndSpeak(String manga) async {
    print(manga);
  QuerySnapshot<Map<String, dynamic>> querySnapshot = await FirebaseFirestore.instance
      .collection('mangas')
      .where('name', isEqualTo: manga) // Use the 'manga' parameter here
      .get();
  
  if (querySnapshot.size > 0) {
    String mainText = querySnapshot.docs[0].data()["main_text"].toString();
    await _speak(mainText, 0.2);
  } else {
    print('Manga not found');
  }
}

  _inferenceCallback(RhinoInference inference) async {
    if(inference.isUnderstood!){
      Map<String , String>? slots = inference.slots!;
      if(inference.intent == "open_manga"){
        String? manga = slots["manga"];
        manga = manga!.replaceAll(' ', '');
        await _speak("Opening ${manga!} manga", 0.5);
        _findAndSpeak(manga);
      }
      else if(inference.intent == "stop"){
        await flutterTts.stop();
      }
    }else{
      _speak("I did not understand your command. Please repeat.", 0.5);
    }
    setState(() {
      _isListening = false;
    });
  }

  _errorCallback(PicovoiceException error) {
    print(error.message);
  }

  _speak(String content, var rate) async{
    if (isSpeaking) {
      // Wait for the current speech to finish before starting the new one
      await Future.delayed(Duration(seconds: 1));
      return _speak(content, rate);
    }
    isSpeaking = true;
    await flutterTts.setQueueMode(1);
    await flutterTts.setLanguage('en-US');
    await flutterTts.setSpeechRate(rate);
    await flutterTts.speak(content);
    isSpeaking = false;
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