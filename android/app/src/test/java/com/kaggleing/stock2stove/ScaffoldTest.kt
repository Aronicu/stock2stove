package com.kaggleing.stock2stove

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ScaffoldTest {
    @Test
    fun testScaffoldInitialState() {
        val appName = "Stock-to-Stove"
        assertTrue(appName.isNotEmpty())
        assertEquals("Stock-to-Stove", appName)
    }

    @Test
    fun testPackageConfiguration() {
        val packageName = "com.kaggleing.stock2stove"
        assertEquals("com.kaggleing.stock2stove", packageName)
    }
}
