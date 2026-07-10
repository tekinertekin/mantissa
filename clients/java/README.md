# Java client

Uses [JNA](https://github.com/java-native-access/jna). Download `jna.jar` (JNA
5.x), then:

```sh
javac -cp jna.jar Demo.java
java  -cp .:jna.jar -Djna.library.path=../../dist Demo    # Windows: use ;
```

`Native.load("mantissa", ...)` finds `libmantissa.{so,dylib}` / `mantissa.dll`
on `jna.library.path` — point it at the folder with your downloaded library
(rename to that base name if needed).
