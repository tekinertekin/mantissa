# C# client

Uses P/Invoke (`[DllImport]`). `DllImport("mantissa")` resolves to
`libmantissa.so` / `libmantissa.dylib` / `mantissa.dll`, so **rename the
downloaded release asset to that base name** and put it next to the binary (or
on the library path).

```sh
cp ../../dist/libmantissa.dylib ./libmantissa.dylib   # macOS example
dotnet run
```

Requires the .NET SDK.
