package com.kaggleing.stock2stove.data.api

import com.kaggleing.stock2stove.data.model.HealthResponse
import com.kaggleing.stock2stove.data.model.RecipeGenerateRequest
import com.kaggleing.stock2stove.data.model.RecipeGenerateResponse
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST

interface InferenceApiService {
    @GET("api/v1/health")
    suspend fun getHealth(): HealthResponse

    @POST("api/v1/recipes/generate")
    suspend fun generateRecipes(
        @Body request: RecipeGenerateRequest
    ): RecipeGenerateResponse
}
