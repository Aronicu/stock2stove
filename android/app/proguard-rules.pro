-keepattributes *Annotation*, InnerClasses, Signature, EnclosingMethod
-keepattributes RuntimeVisibleAnnotations, RuntimeVisibleParameterAnnotations

-dontnote kotlinx.serialization.AnnotationsKt

-keepclassmembers class kotlinx.serialization.json.** {
    *** Companion;
}
-keepclasseswithmembers class kotlinx.serialization.json.** {
    kotlinx.serialization.KSerializer serializer(...);
}

-keep,includedescriptorclasses class com.kaggleing.stock2stove.data.model.**$$serializer { *; }
-keepclassmembers class com.kaggleing.stock2stove.data.model.** {
    *** Companion;
}
-keepclasseswithmembers class com.kaggleing.stock2stove.data.model.** {
    kotlinx.serialization.KSerializer serializer(...);
}

-keep,includedescriptorclasses class com.kaggleing.stock2stove.data.repository.LogEntry$$serializer { *; }
-keepclassmembers class com.kaggleing.stock2stove.data.repository.LogEntry {
    *** Companion;
}
-keepclasseswithmembers class com.kaggleing.stock2stove.data.repository.LogEntry {
    kotlinx.serialization.KSerializer serializer(...);
}

-keepclassmembers,allowshrinking,allowobfuscation interface * {
    @retrofit2.http.* <methods>;
}
-keep,allowobfuscation,allowshrinking interface retrofit2.Call
-keep,allowobfuscation,allowshrinking class retrofit2.Response
-keep,allowobfuscation,allowshrinking class kotlin.coroutines.Continuation

-dontwarn org.codehaus.mojo.animal_sniffer.IgnoreJRERequirement
-dontwarn javax.annotation.**
-dontwarn kotlin.Unit
-dontwarn retrofit2.Platform$Java8
-dontwarn okhttp3.internal.platform.**
-dontwarn org.conscrypt.**
-dontwarn org.bouncycastle.**
-dontwarn org.openjsse.**
