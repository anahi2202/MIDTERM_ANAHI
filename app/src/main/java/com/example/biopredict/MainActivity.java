package com.example.biopredict;

import android.content.res.AssetFileDescriptor;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    EditText inputFieldRBC;
    EditText inputFieldPCV;
    EditText inputFieldMCV;
    EditText inputFieldMCH;

    Button predictBtn;

    TextView resultTV;

    Interpreter interpreter;

    // Min and max values for normalization
    float minRBC = 1.36f;
    float maxRBC = 6.9f;
    float minPCV = 13.1f;
    float maxPCV = 56.9f;
    float minMCV = 55.7f;
    float maxMCV = 124.1f;
    float minMCH = 14.7f;
    float maxMCH = 41.4f;

    // Min and max values for output normalization
    float minHGB = 4.2f;
    float maxHGB = 19.6f;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        try {
            interpreter = new Interpreter(loadModelFile());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // Referencia a los campos de entrada
        inputFieldRBC = findViewById(R.id.editTextRBC);
        inputFieldPCV = findViewById(R.id.editTextPCV);
        inputFieldMCV = findViewById(R.id.editTextMCV);
        inputFieldMCH = findViewById(R.id.editTextMCH);

        predictBtn = findViewById(R.id.button);
        resultTV = findViewById(R.id.textView);

        predictBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Obtención de datos de entrada
                String rbcInput = inputFieldRBC.getText().toString();
                String pcvInput = inputFieldPCV.getText().toString();
                String mcvInput = inputFieldMCV.getText().toString();
                String mchInput = inputFieldMCH.getText().toString();

                // Conversión a float
                float rbcValue = Float.parseFloat(rbcInput);
                float pcvValue = Float.parseFloat(pcvInput);
                float mcvValue = Float.parseFloat(mcvInput);
                float mchValue = Float.parseFloat(mchInput);

                // Normalización de los valores
                float normRBC = (rbcValue - minRBC) / (maxRBC - minRBC);
                float normPCV = (pcvValue - minPCV) / (maxPCV - minPCV);
                float normMCV = (mcvValue - minMCV) / (maxMCV - minMCV);
                float normMCH = (mchValue - minMCH) / (maxMCH - minMCH);

                // Preparar los datos de entrada normalizados
                float[][] inputs = new float[1][4];
                inputs[0][0] = normRBC;
                inputs[0][1] = normPCV;
                inputs[0][2] = normMCV;
                inputs[0][3] = normMCH;

                // Hacer predicción
                float normalizedPrediction = doInference(inputs);

                // Desnormalizar el valor predicho
                float predictedHGB = normalizedPrediction * (maxHGB - minHGB) + minHGB;

                resultTV.setText("Predicted HGB: " + predictedHGB);
            }
        });

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }

    // Método para realizar la inferencia
    public float doInference(float[][] input) {
        float[][] output = new float[1][1];
        interpreter.run(input, output);

        return output[0][0]; // Devuelve el valor normalizado
    }

    // Método para cargar el modelo TFLite
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor assetFileDescriptor = this.getAssets().openFd("MIDTERMANAHI_linear.tflite");
        FileInputStream fileInputStream = new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel = fileInputStream.getChannel();
        long startOffset = assetFileDescriptor.getStartOffset();
        long length = assetFileDescriptor.getLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, length);
    }
}
