package com.xlzhen.codeformer;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.MemoryFormat;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.channels.FileChannel;
import java.util.Objects;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        new Thread(()->{
            File file = new File(getFilesDir().getAbsolutePath() + "/codeformer.ptl");
            if (!file.exists()) {
                Log.v("codeformer model", "not exist");
                copyAssets();
            }
            Module module = LiteModuleLoader.load(getFilesDir().getAbsolutePath() + "/codeformer.ptl");

            Log.v("module", module.toString());
            try {
                for (String name : Objects.requireNonNull(getAssets().list("cropped_faces"))) {
                    Bitmap bitmap = BitmapFactory.decodeStream(getAssets().open("cropped_faces/" + name));

                    Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap, new float[]{0.5f, 0.5f, 0.5f}
                            , new float[]{0.5f, 0.5f, 0.5f}, MemoryFormat.CHANNELS_LAST);

                    Log.v("inputTensor", inputTensor.toString());
                    IValue value = module.forward(IValue.from(inputTensor));
                    IValue[] tensors = value.toTuple();
                    Log.v("value", value.toString());
                    float[] array1 = tensors[0].toTensor().getDataAsFloatArray();

                    Bitmap outBitmap = floatArrayToBitmap(array1, 512, 512);
                    Log.v("output", outBitmap.toString());
                    saveBitmapToFile(outBitmap, name);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }).start();



    }

    private void saveBitmapToFile(Bitmap bitmap, String name) throws IOException {
        File f = new File(Objects.requireNonNull(getExternalCacheDir()).getAbsolutePath() + "/" + name);
        f.createNewFile();
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 0 /*ignored for PNG*/, bos);
        byte[] bitmapdata = bos.toByteArray();

        FileOutputStream fos = new FileOutputStream(f);
        fos.write(bitmapdata);
        fos.flush();
        fos.close();
    }

    private Bitmap floatArrayToBitmap(float[] floatArray, int width, int height) {

        // Create empty bitmap in RGBA format
        Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[width * height * 4];

        // mapping smallest value to 0 and largest value to 255
        float maxValue = 1.0f;
        float minValue = -1.0f;
        for (float value : floatArray) {
            if (value > maxValue) {
                maxValue = value;
            }
            if(value < minValue){
                minValue = value;
            }
        }
        float delta = maxValue - minValue;

        // copy each value from float array to RGB channels and set alpha channel
        for (int i = 0; i < width * height; i++){
            int r = conversion(minValue,delta,floatArray[i]);
            int g = conversion(minValue,delta,floatArray[i + width * height]);
            int b = conversion(minValue,delta,floatArray[i + 2 * width * height]);
            pixels[i] =Color.rgb(r, g, b);
        }
        bmp.setPixels(pixels, 0, width, 0, 0, width, height);

        return bmp;
    }

    private int conversion(float minValue,float delta,float data){
        return (int)((data - minValue) / delta * 255.0f);
    }
    private void copyAssets() {
        AssetManager assetManager = getAssets();

        InputStream in = null;
        OutputStream out = null;
        try {
            in = assetManager.open("codeformer.ptl");
            File outFile = new File(getFilesDir().getAbsolutePath() + "/", "codeformer.ptl");
            out = new FileOutputStream(outFile);
            copyFile(in, out);
            in.close();
            in = null;
            out.flush();
            out.close();
            out = null;
        } catch (IOException e) {
            Log.e("tag", "Failed to copy asset file: codeformer.ptl", e);
        }

    }

    private void copyFile(InputStream in, OutputStream out) throws IOException {
        byte[] buffer = new byte[8192];
        int read;
        while ((read = in.read(buffer)) != -1) {
            out.write(buffer, 0, read);
        }
    }
}