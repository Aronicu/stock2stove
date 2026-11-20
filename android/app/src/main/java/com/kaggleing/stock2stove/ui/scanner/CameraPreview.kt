package com.kaggleing.stock2stove.ui.scanner

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import java.io.ByteArrayOutputStream
import java.util.Base64
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.roundToInt

object CameraFrameProcessor {
    fun processBitmapToJpeg(
        bitmap: Bitmap,
        rotationDegrees: Int = 0,
        targetWidth: Int = 640,
        targetHeight: Int = 640,
        quality: Int = 80
    ): ByteArray {
        val rotated = if (rotationDegrees != 0) {
            val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } else {
            bitmap
        }

        val scaled = if (rotated.width != targetWidth || rotated.height != targetHeight) {
            Bitmap.createScaledBitmap(rotated, targetWidth, targetHeight, true)
        } else {
            rotated
        }

        val outputStream = ByteArrayOutputStream()
        scaled.compress(Bitmap.CompressFormat.JPEG, quality, outputStream)
        return outputStream.toByteArray()
    }

    fun encodeBase64(bytes: ByteArray): String {
        return Base64.getEncoder().encodeToString(bytes)
    }

    fun cropJpegBase64(
        jpegBytes: ByteArray,
        bboxNorm: List<Float>,
        maxSide: Int = 128,
        quality: Int = 78
    ): String? {
        if (jpegBytes.isEmpty() || bboxNorm.size < 4) {
            return null
        }
        return try {
            val fullBitmap = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size) ?: return null
            val w = fullBitmap.width
            val h = fullBitmap.height
            if (w <= 0 || h <= 0) {
                return null
            }

            val rawX1 = bboxNorm[0].coerceIn(0f, 1f)
            val rawY1 = bboxNorm[1].coerceIn(0f, 1f)
            val rawX2 = bboxNorm[2].coerceIn(0f, 1f)
            val rawY2 = bboxNorm[3].coerceIn(0f, 1f)

            val x1 = minOf(rawX1, rawX2)
            val y1 = minOf(rawY1, rawY2)
            val x2 = maxOf(rawX1, rawX2)
            val y2 = maxOf(rawY1, rawY2)

            val px1 = (x1 * w).toInt().coerceIn(0, w - 1)
            val py1 = (y1 * h).toInt().coerceIn(0, h - 1)
            val px2 = (x2 * w).roundToInt().coerceIn(px1 + 1, w)
            val py2 = (y2 * h).roundToInt().coerceIn(py1 + 1, h)

            val cw = px2 - px1
            val ch = py2 - py1
            if (cw <= 0 || ch <= 0) {
                return null
            }

            val cropped = Bitmap.createBitmap(fullBitmap, px1, py1, cw, ch)
            val scaled = if (maxSide > 0 && maxOf(cw, ch) > maxSide) {
                val scale = maxSide.toFloat() / maxOf(cw, ch).toFloat()
                val nw = (cw * scale).roundToInt().coerceAtLeast(1)
                val nh = (ch * scale).roundToInt().coerceAtLeast(1)
                Bitmap.createScaledBitmap(cropped, nw, nh, true)
            } else {
                cropped
            }

            val outputStream = ByteArrayOutputStream()
            scaled.compress(Bitmap.CompressFormat.JPEG, quality, outputStream)
            val outBytes = outputStream.toByteArray()
            if (outBytes.isEmpty()) {
                null
            } else {
                encodeBase64(outBytes)
            }
        } catch (_: Throwable) {
            null
        }
    }

    fun decodeBase64Bitmap(b64: String?): Bitmap? {
        if (b64.isNullOrBlank()) {
            return null
        }
        return try {
            var cleaned = b64.trim()
            if (cleaned.startsWith("data:")) {
                val commaIdx = cleaned.indexOf(',')
                if (commaIdx >= 0) {
                    cleaned = cleaned.substring(commaIdx + 1)
                }
            }
            val bytes = Base64.getDecoder().decode(cleaned)
            if (bytes.isEmpty()) {
                null
            } else {
                BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            }
        } catch (_: Throwable) {
            null
        }
    }

    fun shouldCaptureFrame(currentTimeMs: Long, lastFrameTimeMs: Long, targetFps: Int): Boolean {
        if (targetFps <= 0) return true
        val minIntervalMs = 1000L / targetFps
        return (currentTimeMs - lastFrameTimeMs) >= minIntervalMs
    }
}

@Composable
fun CameraPreview(
    modifier: Modifier = Modifier,
    targetFps: Int = 8,
    onFrameCaptured: (ByteArray, String) -> Unit = { _, _ -> }
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
    val lastFrameTime = remember { AtomicLong(0L) }
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCameraPermission = granted
    }

    LaunchedEffect(Unit) {
        if (!hasCameraPermission) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            cameraExecutor.shutdown()
        }
    }

    if (!hasCameraPermission) {
        Box(
            modifier = modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(text = "Camera permission is required to scan", color = Color.White)
                Button(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) {
                    Text(text = "Grant Camera Access")
                }
            }
        }
        return
    }

    AndroidView(
        factory = { ctx ->
            val previewView = PreviewView(ctx).apply {
                implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            }
            val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
            cameraProviderFuture.addListener({
                val cameraProvider = cameraProviderFuture.get()
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

                val imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()

                imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    val now = System.currentTimeMillis()
                    val last = lastFrameTime.get()
                    if (CameraFrameProcessor.shouldCaptureFrame(now, last, targetFps)) {
                        lastFrameTime.set(now)
                        try {
                            val bitmap = imageProxy.toBitmap()
                            val jpegBytes = CameraFrameProcessor.processBitmapToJpeg(
                                bitmap = bitmap,
                                rotationDegrees = imageProxy.imageInfo.rotationDegrees,
                                targetWidth = 640,
                                targetHeight = 640
                            )
                            val b64 = CameraFrameProcessor.encodeBase64(jpegBytes)
                            onFrameCaptured(jpegBytes, b64)
                        } catch (_: Throwable) {
                        }
                    }
                    imageProxy.close()
                }

                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                try {
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        cameraSelector,
                        preview,
                        imageAnalysis
                    )
                } catch (_: Throwable) {
                }
            }, ContextCompat.getMainExecutor(ctx))
            previewView
        },
        modifier = modifier.fillMaxSize()
    )
}
