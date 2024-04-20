
import 'package:a_silent_voice/homepage.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';



void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'A_Silent_Voice',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        primaryColor: Colors.grey,
        focusColor: Colors.blue,
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

