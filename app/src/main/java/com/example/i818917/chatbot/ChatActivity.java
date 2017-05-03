
package com.example.i818917.chatbot;

import android.content.Intent;
import android.speech.RecognizerIntent;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.ToggleButton;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.List;
import java.util.Locale;

public class ChatActivity extends AppCompatActivity {

  private TensorFlowInferenceInterface inferenceInterface;

  static {
    System.loadLibrary("tensorflow_inference");
  }

  private static final int SPEECH_REQUEST_CODE = 0;
  TextToSpeech mTTS;
  ToggleButton mTB;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_chat);

    inferenceInterface = new TensorFlowInferenceInterface();
    //inferenceInterface.initializeTensorFlow(getAssets(), MODEL_FILE);

    mTTS = new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
      @Override
      public void onInit(int status) {
        mTTS.setLanguage(Locale.UK);
      }
    });
    final ChatActivity that = this;
    mTTS.setOnUtteranceProgressListener(new UtteranceProgressListener() {
      @Override
      public void onStart(String utteranceId) {

      }

      @Override
      public void onDone(String utteranceId) {
        that.displaySpeechRecognizer();
      }

      @Override
      public void onError(String utteranceId) {

      }
    });

    mTB = (ToggleButton) findViewById(R.id.toggleButton);
    mTB.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v) {
        displaySpeechRecognizer();
      }
    });
  }

  private void displaySpeechRecognizer() {
    if (!mTB.isChecked()){
      return;
    }
    Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
    intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
    // Start the activity, the intent will be populated with the speech text
    startActivityForResult(intent, SPEECH_REQUEST_CODE);
  }

  @Override
  protected void onActivityResult(int requestCode, int resultCode,
                                  Intent data) {
    if (requestCode == SPEECH_REQUEST_CODE && resultCode == RESULT_OK) {
      List<String> results = data.getStringArrayListExtra(
          RecognizerIntent.EXTRA_RESULTS);
      String spokenText = results.get(0);
      mTTS.speak(spokenText,TextToSpeech.QUEUE_ADD, null, "message");
      return;
    }
    super.onActivityResult(requestCode, resultCode, data);
  }

}
