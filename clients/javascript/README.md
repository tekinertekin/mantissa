# JavaScript (Node.js) client

Uses [koffi](https://koffi.dev), a maintained FFI for Node.

```sh
npm install
node demo.js ../../dist/libmantissa.dylib      # or the file you downloaded
```

The library path can also come from `$MANTISSA_LIB`, or defaults to the release
asset name for your OS. Requires Node 18+ and a library built for the same
architecture as your Node binary.
