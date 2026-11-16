package com.kaggleing.stock2stove

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import com.kaggleing.stock2stove.data.repository.AlphaTelemetryLogger
import com.kaggleing.stock2stove.ui.theme.Stock2StoveTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        AlphaTelemetryLogger.getInstance(applicationContext).logEvent("APP_LIFECYCLE", "INFO", "Stock2Stove application started")
        setContent {
            Stock2StoveTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    Stock2StoveApp()
                }
            }
        }
    }
}
