# C++ client

Loads the shared library at runtime (`dlopen`/`LoadLibrary`) — only the library
file is needed, no headers.

```sh
c++ -std=c++17 demo.cpp -o demo          # Linux: add -ldl
./demo ../../dist/libmantissa.dylib      # or the file you downloaded
```

The path can also come from `$MANTISSA_LIB`, or defaults to the release asset
name for your OS.
