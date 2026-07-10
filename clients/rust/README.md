# Rust client

Uses the [`libloading`](https://crates.io/crates/libloading) crate to load the
library at runtime.

```sh
cargo run -- ../../dist/libmantissa.dylib      # or the file you downloaded
```

The path can also come from `$MANTISSA_LIB`, or defaults to the release asset
name for your OS.
